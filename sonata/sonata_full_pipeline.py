import numpy as np
import open3d as o3d
import sonata
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', DEVICE)
import torch.nn as nn
import gc
import os
import sys
import argparse
from scipy.spatial import cKDTree

# LAS processing imports
import laspy
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import eigh

try:
    import flash_attn
except ImportError:
    flash_attn = None

# ============================================================================
# LAS CONVERTER FUNCTIONS
# ============================================================================

def compute_normals_vectorized(points, indices):
    """Fully vectorized normal computation - fastest method"""
    # print("Computing normals using vectorized operations...")
    n_points, k = indices.shape
    
    # Get all neighborhoods at once: shape (n_points, k, 3)
    neighborhoods = points[indices]
    
    # Compute centroids for all neighborhoods: shape (n_points, 1, 3)
    centroids = np.mean(neighborhoods, axis=1, keepdims=True)
    
    # Center all neighborhoods: shape (n_points, k, 3)
    centered = neighborhoods - centroids
    
    # Compute covariance matrices for all points at once
    # print("Computing covariance matrices...")
    cov_matrices = np.einsum('nij,nik->njk', centered, centered) / (k - 1)
    
    # print("Computing eigenvalues and eigenvectors...")
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrices)
    
    # Normal is the eigenvector corresponding to smallest eigenvalue
    normals = eigenvectors[:, :, 0]  # shape (n_points, 3)
    
    # print("Orienting normals...")
    # Ensure consistent orientation
    point_cloud_center = np.mean(points, axis=0)
    to_center = points - point_cloud_center
    
    # Check if normal points towards center (dot product < 0)
    dot_products = np.sum(normals * to_center, axis=1)
    flip_mask = dot_products < 0
    normals[flip_mask] *= -1
    
    # print("Vectorized normal computation completed!")
    return normals

def compute_normals_fast(points, k=20, method='vectorized'):
    """Compute surface normals for point cloud using vectorized operations"""
    print(f"Computing normals using {k} nearest neighbors...")
    
    # Use KNN to find local neighborhoods
    # print("Building KNN index...")
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(points)
    # print("Finding neighborhoods...")
    distances, indices = nbrs.kneighbors(points)
    
    return compute_normals_vectorized(points, indices)

def subsample_by_distance(points, colors=None, intensity=None, distance=0.010):
    """Subsample point cloud by minimum distance between points"""
    print(f"Subsampling with minimum distance: {distance}m")
    
    n_points = len(points)
    selected_indices = []
    
    # Use spatial hashing for efficiency
    grid_size = distance
    grid = {}
    
    print("Processing points for subsampling:")
    for i in range(n_points):
        if i % max(1, n_points // 100) == 0 or i % 10000 == 0:
            progress = (i / n_points) * 100
            selected_count = len(selected_indices)
            print(f"  Progress: {progress:.1f}% ({i}/{n_points}) - Selected: {selected_count}", end='\r')
        
        point = points[i]
        
        # Convert point to grid coordinates
        grid_x = int(point[0] / grid_size)
        grid_y = int(point[1] / grid_size)
        grid_z = int(point[2] / grid_size)
        
        # Check surrounding grid cells for nearby points
        too_close = False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    cell = (grid_x + dx, grid_y + dy, grid_z + dz)
                    if cell in grid:
                        for existing_idx in grid[cell]:
                            if np.linalg.norm(points[existing_idx] - point) < distance:
                                too_close = True
                                break
                        if too_close:
                            break
                    if too_close:
                        break
                if too_close:
                    break
        
        if not too_close:
            # Add point to grid
            cell_key = (grid_x, grid_y, grid_z)
            if cell_key not in grid:
                grid[cell_key] = []
            grid[cell_key].append(i)
            selected_indices.append(i)
    
    selected_indices = np.array(selected_indices)
    final_count = len(selected_indices)
    reduction_percent = final_count / n_points * 100
    
    print(f"  Progress: 100.0% ({n_points}/{n_points}) - Selected: {final_count}")
    print(f"Subsampling completed! Reduced from {n_points} to {final_count} points ({reduction_percent:.1f}%)")
    
    # Subsample all arrays
    subsampled_points = points[selected_indices]
    subsampled_colors = colors[selected_indices] if colors is not None else None
    subsampled_intensity = intensity[selected_indices] if intensity is not None else None
    
    return subsampled_points, subsampled_colors, subsampled_intensity, selected_indices

def las_to_npy(las_file_path, output_dir="output", save_npz=True, subsample_distance=0.010, normal_method='vectorized', sonata_format=False):
    """Convert LAS file to NPY files and NPZ format"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading LAS file: {las_file_path}")
    
    # Read LAS file
    with laspy.open(las_file_path) as las_file:
        las = las_file.read()
    
    print(f"Loaded {len(las.points)} points")
    
    # Extract coordinates (XYZ)
    coordinates = np.vstack([las.x, las.y, las.z]).T
    # print(f"Coordinates shape: {coordinates.shape}")
    
    # Extract RGB colors
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        colors = np.vstack([
            (las.red / 65535.0 * 255).astype(np.uint8),
            (las.green / 65535.0 * 255).astype(np.uint8),
            (las.blue / 65535.0 * 255).astype(np.uint8)
        ]).T
        # print(f"Colors shape: {colors.shape}")
    else:
        print("Warning: No RGB color information found in LAS file")
        colors = None
    
    # Extract intensity
    if hasattr(las, 'intensity'):
        intensity = las.intensity
        # print(f"Intensity shape: {intensity.shape}")
    else:
        print("Warning: No intensity information found in LAS file")
        intensity = None
    
    # Apply subsampling if requested
    if subsample_distance and subsample_distance > 0:
        # print(f"Subsampling with minimum distance: {subsample_distance}m")
        coordinates, colors, intensity, subsample_indices = subsample_by_distance(
            coordinates, colors, intensity, subsample_distance
        )
        print(f"After subsampling: {coordinates.shape[0]} points")
    
    # Compute surface normals
    normals = compute_normals_fast(coordinates, method=normal_method)
    print(f"Normals shape: {normals.shape}")
    
    # Save to NPY files
    coord_file = os.path.join(output_dir, "coordinates.npy")
    np.save(coord_file, coordinates)
    # print(f"Saved coordinates to: {coord_file}")
    
    if colors is not None:
        color_file = os.path.join(output_dir, "colors.npy")
        np.save(color_file, colors)
        # print(f"Saved colors to: {color_file}")
    
    normal_file = os.path.join(output_dir, "normals.npy")
    np.save(normal_file, normals)
    # print(f"Saved normals to: {normal_file}")
    
    if intensity is not None:
        intensity_file = os.path.join(output_dir, "intensity.npy")
        np.save(intensity_file, intensity)
        # print(f"Saved intensity to: {intensity_file}")
    
    # Save as single NPZ file
    if save_npz:
        if sonata_format:
            npz_data = {
                'coord': coordinates.astype(np.float32),
                'normal': normals.astype(np.float32)
            }
            if colors is not None:
                npz_data['color'] = colors.astype(np.float32)
            if intensity is not None:
                npz_data['intensity'] = intensity.astype(np.float32)
        else:
            npz_data = {
                'coordinates': coordinates,
                'normals': normals
            }
            if colors is not None:
                npz_data['colors'] = colors
            if intensity is not None:
                npz_data['intensity'] = intensity
            
        npz_file = os.path.join(output_dir, "complete.npz")
        np.savez_compressed(npz_file, **npz_data)
        print(f"Saved complete data to: {npz_file}")
        
        if sonata_format:
            print(f"NPZ saved in Sonata-compatible format with keys: {list(npz_data.keys())}")
    
    return {
        'coordinates': coordinates,
        'colors': colors,
        'normals': normals,
        'intensity': intensity
    }

def run_las_converter(las_file="alpha.las", subsample_distance=0.010, output_dir="output"):
    """Execute LAS conversion"""
    print("\n" + "="*60)
    print("LAS FILE CONVERSION")
    print("="*60)

    normal_method = 'vectorized'

    print(f"Processing LAS file: {las_file}")
    print(f"Normal method: {normal_method} (fixed)")
    print(f"Subsample distance: {subsample_distance}m")
    
    try:
        # Convert LAS file with Sonata format for compatibility
        data = las_to_npy(las_file, output_dir=output_dir, save_npz=True,
                         subsample_distance=subsample_distance,
                         normal_method=normal_method, sonata_format=True)
        print("\nLAS conversion completed successfully!")
        
        # Print some statistics
        print(f"\nData summary:")
        print(f"Number of points: {len(data['coordinates'])}")
        print(f"Coordinate range:")
        print(f"  X: {data['coordinates'][:, 0].min():.2f} to {data['coordinates'][:, 0].max():.2f}")
        print(f"  Y: {data['coordinates'][:, 1].min():.2f} to {data['coordinates'][:, 1].max():.2f}")
        print(f"  Z: {data['coordinates'][:, 2].min():.2f} to {data['coordinates'][:, 2].max():.2f}")
        
        if data['colors'] is not None:
            print(f"Color range: {data['colors'].min()} to {data['colors'].max()}")
        
        # print(f"Normal vector lengths (should be ~1.0): {np.linalg.norm(data['normals'], axis=1).mean():.3f}")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: File '{las_file}' not found!")
        print("Usage:")
        print(f"  python {sys.argv[0]} <las_file> [subsample_distance]")
        print(f"  python {sys.argv[0]} alpha.las 0.010")
        print(f"  python {sys.argv[0]} alpha.las 0.005")
        return False
    except Exception as e:
        print(f"Error processing LAS file: {e}")
        print("Make sure you have the required packages installed:")
        print("pip install laspy scikit-learn scipy numpy")
        return False

# ============================================================================
# ROUND 1: FLOOR/CEILING DETECTION 
# ============================================================================

# ScanNet Meta data - UPDATED with ceiling class
VALID_CLASS_IDS_20 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
CLASS_LABELS_20 = ("wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture", "counter", "desk", "curtain", "refrigerator", "shower curtain", "toilet", "sink", "bathtub", "otherfurniture")

# Extended color map with ceiling class 22
SCANNET_COLOR_MAP_EXTENDED = {
    0: (0.0, 0.0, 0.0), 1: (174.0, 199.0, 232.0), 2: (152.0, 223.0, 138.0), 3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0), 5: (188.0, 189.0, 34.0), 6: (140.0, 86.0, 75.0), 7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0), 9: (197.0, 176.0, 213.0), 10: (148.0, 103.0, 189.0), 11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0), 13: (128.0, 128.0, 128.0), 14: (247.0, 182.0, 210.0), 15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0), 17: (140.0, 57.0, 197.0), 18: (202.0, 185.0, 52.0), 19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0), 21: (92.0, 193.0, 61.0), 22: (255.0, 255.0, 150.0), 23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0), 25: (91.0, 163.0, 138.0), 26: (153.0, 98.0, 156.0), 27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0), 29: (100.0, 125.0, 154.0), 30: (178.0, 127.0, 135.0), 31: (146.0, 111.0, 194.0),
    32: (146.0, 111.0, 194.0), 33: (44.0, 160.0, 44.0), 34: (112.0, 128.0, 144.0), 35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0), 37: (213.0, 92.0, 176.0), 38: (94.0, 106.0, 211.0), 39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0)
}

# Create extended class color array
CLASS_COLOR_EXTENDED = []
for i in range(41):
    if i in SCANNET_COLOR_MAP_EXTENDED:
        CLASS_COLOR_EXTENDED.append(SCANNET_COLOR_MAP_EXTENDED[i])
    else:
        CLASS_COLOR_EXTENDED.append((128.0, 128.0, 128.0))

class SegHead(nn.Module):
    def __init__(self, backbone_out_channels, num_classes):
        super(SegHead, self).__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x):
        return self.seg_head(x)

def create_spatial_chunks(coord, max_points_per_chunk=300000, overlap_ratio=0.5):
    """Create overlapping spatial chunks"""
    print(f"Creating spatial chunks from {len(coord)} points with max {max_points_per_chunk} points per chunk...")
    
    min_coords = coord.min(axis=0)
    max_coords = coord.max(axis=0)
    bbox_size = max_coords - min_coords
    overlap_dist = bbox_size * overlap_ratio
    
    chunks = []
    chunk_indices = []
    
    largest_dim = np.argmax(bbox_size)
    estimated_chunks_needed = max(1, len(coord) // max_points_per_chunk)
    
    if largest_dim == 0:  # X dimension
        num_segments = max(2, int(np.ceil(estimated_chunks_needed * 1.5)))
        chunk_width = bbox_size[0] / num_segments
        current_x = min_coords[0]
        
        while current_x < max_coords[0]:
            x_min = current_x - (overlap_dist[0] if current_x > min_coords[0] else 0)
            x_max = current_x + chunk_width + overlap_dist[0]
            
            mask = (coord[:, 0] >= x_min) & (coord[:, 0] <= x_max)
            indices = np.where(mask)[0]
            
            if len(indices) > 0:
                if len(indices) > max_points_per_chunk:
                    num_splits = int(np.ceil(len(indices) / max_points_per_chunk))
                    for split_idx in range(num_splits):
                        start_idx = split_idx * max_points_per_chunk
                        end_idx = min((split_idx + 1) * max_points_per_chunk, len(indices))
                        split_indices = indices[start_idx:end_idx]
                        chunks.append(coord[split_indices])
                        chunk_indices.append(split_indices)
                else:
                    chunks.append(coord[indices])
                    chunk_indices.append(indices)
            
            current_x += chunk_width
    
    print(f"Created {len(chunks)} spatial chunks")
    return chunks, chunk_indices

def merge_overlapping_predictions(original_coord, chunk_indices, chunk_predictions, original_normal, overlap_ratio=0.5):
    """Merge predictions from overlapping chunks"""
    print("Merging overlapping predictions...")
    
    final_predictions = np.zeros(len(original_coord), dtype=np.float32)
    prediction_weights = np.zeros(len(original_coord), dtype=np.float32)
    
    for chunk_idx, (indices, pred) in enumerate(zip(chunk_indices, chunk_predictions)):
        chunk_coord = original_coord[indices]
        chunk_center = chunk_coord.mean(axis=0)
        distances = np.linalg.norm(chunk_coord - chunk_center, axis=1)
        max_distance = distances.max()
        
        weights = np.exp(-(distances / max_distance) ** 2)
        
        final_predictions[indices] += pred * weights
        prediction_weights[indices] += weights
    
    valid_mask = prediction_weights > 0
    final_predictions[valid_mask] /= prediction_weights[valid_mask]
    
    merged_predictions = np.round(final_predictions).astype(np.int32)
    
    print("Prediction merging completed")
    return merged_predictions

def apply_spatial_consistency(coords, predictions, normals):
    """Floor and ceiling detection"""
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        DBSCAN = None
    
    cleaned_predictions = predictions.copy()
    
    print("Applying floor/ceiling detection...")
    
    # Clear existing floor and ceiling classifications
    cleaned_predictions[cleaned_predictions == 2] = 0   # Floor → background
    cleaned_predictions[cleaned_predictions == 22] = 0  # Ceiling → background
    
    vertical_up = np.array([0, 0, 1])
    vertical_down = np.array([0, 0, -1])
    
    dot_with_up = np.dot(normals, vertical_up)
    dot_with_down = np.dot(normals, vertical_down)
    
    threshold = 0.94
    floor_candidates = np.where(dot_with_down > threshold)[0]
    ceiling_candidates = np.where(dot_with_up > threshold)[0]
    
    print(f"Floor candidates: {len(floor_candidates)}")
    print(f"Ceiling candidates: {len(ceiling_candidates)}")
    
    if len(floor_candidates) > 0:
        large_floor_points = filter_by_area(coords[floor_candidates], floor_candidates, 
                                          min_area_sqm=5.0, max_neighbor_dist=0.075, 
                                          use_dbscan=(DBSCAN is not None))
        if len(large_floor_points) > 0:
            cleaned_predictions[large_floor_points] = 2
            print(f"Classified {len(large_floor_points)} points as floors")
    
    if len(ceiling_candidates) > 0:
        large_ceiling_points = filter_by_area(coords[ceiling_candidates], ceiling_candidates,
                                            min_area_sqm=5.0, max_neighbor_dist=0.075,
                                            use_dbscan=(DBSCAN is not None))
        if len(large_ceiling_points) > 0:
            cleaned_predictions[large_ceiling_points] = 22
            print(f"Classified {len(large_ceiling_points)} points as ceilings")
    
    return cleaned_predictions

def filter_by_area(candidate_coords, candidate_indices, min_area_sqm=2.0, max_neighbor_dist=0.15, use_dbscan=True):
    """Filter surface candidates by connected area"""
    if len(candidate_coords) < 50:
        return np.array([], dtype=int)
    
    if use_dbscan:
        try:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=max_neighbor_dist, min_samples=10).fit(candidate_coords)
            labels = clustering.labels_
        except ImportError:
            labels = simple_clustering(candidate_coords, max_neighbor_dist)
    else:
        labels = simple_clustering(candidate_coords, max_neighbor_dist)
    
    large_area_points = []
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]
    
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_points = candidate_coords[cluster_mask]
        cluster_indices = candidate_indices[cluster_mask]
        
        if len(cluster_points) < 20:
            continue
            
        area_sqm = estimate_surface_area(cluster_points)
        
        if area_sqm >= min_area_sqm:
            large_area_points.extend(cluster_indices)
    
    return np.array(large_area_points, dtype=int)

def simple_clustering(points, max_distance):
    """Simple fallback clustering"""
    tree = cKDTree(points)
    labels = np.full(len(points), -1, dtype=int)
    current_label = 0
    
    for i in range(len(points)):
        if labels[i] != -1:
            continue
            
        cluster_points = [i]
        labels[i] = current_label
        
        to_check = [i]
        while to_check:
            point_idx = to_check.pop()
            neighbors = tree.query_ball_point(points[point_idx], max_distance)
            
            for neighbor in neighbors:
                if labels[neighbor] == -1:
                    labels[neighbor] = current_label
                    cluster_points.append(neighbor)
                    to_check.append(neighbor)
        
        if len(cluster_points) >= 10:
            current_label += 1
        else:
            for idx in cluster_points:
                labels[idx] = -1
    
    return labels

def estimate_surface_area(points_3d):
    """Estimate surface area using 2D projection and convex hull"""
    if len(points_3d) < 3:
        return 0.0
    
    try:
        from scipy.spatial import ConvexHull
        points_2d = points_3d[:, :2]
        unique_points = np.unique(points_2d, axis=0)
        
        if len(unique_points) < 3:
            return 0.0
            
        hull = ConvexHull(unique_points)
        area_sqm = hull.volume
        
        return area_sqm
        
    except Exception:
        min_coords = points_3d.min(axis=0)
        max_coords = points_3d.max(axis=0)
        bbox_area = (max_coords[0] - min_coords[0]) * (max_coords[1] - min_coords[1])
        return bbox_area * 0.7

def process_single_chunk_round1(model, seg_head, transform, chunk_coords, chunk_colors, chunk_normals, chunk_idx, total_chunks):
    """Process a single chunk for Round 1"""
    # Progress bar display
    progress = (chunk_idx + 1) / total_chunks * 100
    bar_length = 30
    filled_length = int(bar_length * (chunk_idx + 1) // total_chunks)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f"\rRound 1 Progress: |{bar}| {progress:.1f}% ({chunk_idx + 1}/{total_chunks}) chunks", end='', flush=True)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        point_data = {
            'coord': chunk_coords,
            'color': chunk_colors,
            'normal': chunk_normals
        }
        
        transformed_point = transform(point_data)
        
        with torch.inference_mode():
            for key in transformed_point.keys():
                if isinstance(transformed_point[key], torch.Tensor):
                    transformed_point[key] = transformed_point[key].to(DEVICE, non_blocking=True)
            
            result = model(transformed_point)
            
            while "pooling_parent" in result.keys():
                torch.cuda.empty_cache()
                parent = result.pop("pooling_parent")
                inverse = result.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, result.feat[inverse]], dim=-1)
                result = parent
            
            features = result.feat
            logits = seg_head(features)
            predictions = logits.argmax(dim=-1).cpu().numpy()
            processed_coords = result.coord.cpu().numpy()
            
            tree = cKDTree(processed_coords)
            distances, nn_indices = tree.query(chunk_coords, k=1)
            final_predictions = predictions[nn_indices]
            
            del result, features, logits
            torch.cuda.empty_cache()
        
        return final_predictions
        
    except Exception as e:
        print(f"\nError in chunk {chunk_idx + 1}: {e}")
        return np.ones(len(chunk_coords), dtype=np.int32) * 2

def export_floor_ceiling_and_remaining(original_coord, original_color, original_normal, predictions, output_dir="output"):
    """Export floor/ceiling points and remaining points as separate NPZ files"""
    print("\n=== EXPORTING FLOOR/CEILING AND REMAINING POINTS ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    floor_ceiling_mask = (predictions == 2) | (predictions == 22)
    remaining_mask = ~floor_ceiling_mask
    
    floor_ceiling_indices = np.where(floor_ceiling_mask)[0]
    remaining_indices = np.where(remaining_mask)[0]
    
    print(f"Total points: {len(original_coord)}")
    print(f"Floor/Ceiling points: {len(floor_ceiling_indices)} ({len(floor_ceiling_indices)/len(original_coord)*100:.1f}%)")
    print(f"Remaining points: {len(remaining_indices)} ({len(remaining_indices)/len(original_coord)*100:.1f}%)")
    
    floor_ceiling_file = os.path.join(output_dir, "floor_ceiling.npz")
    np.savez_compressed(floor_ceiling_file,
                       coord=original_coord[floor_ceiling_indices],
                       color=original_color[floor_ceiling_indices], 
                       normal=original_normal[floor_ceiling_indices],
                       predictions=predictions[floor_ceiling_indices],
                       original_indices=floor_ceiling_indices)
    
    remaining_file = os.path.join(output_dir, "remaining_points.npz")
    np.savez_compressed(remaining_file,
                       coord=original_coord[remaining_indices],
                       color=original_color[remaining_indices],
                       normal=original_normal[remaining_indices], 
                       predictions=predictions[remaining_indices],
                       original_indices=remaining_indices)
    
    # Create visualizations
    floor_ceiling_colors = np.array(CLASS_COLOR_EXTENDED)[predictions[floor_ceiling_indices]] / 255.0
    pcd_floor_ceiling = o3d.geometry.PointCloud()
    pcd_floor_ceiling.points = o3d.utility.Vector3dVector(original_coord[floor_ceiling_indices])
    pcd_floor_ceiling.colors = o3d.utility.Vector3dVector(floor_ceiling_colors)
    
    floor_ceiling_ply = os.path.join(output_dir, "floor_ceiling.ply")
    o3d.io.write_point_cloud(floor_ceiling_ply, pcd_floor_ceiling)
    
    remaining_colors = np.array(CLASS_COLOR_EXTENDED)[predictions[remaining_indices]] / 255.0
    pcd_remaining = o3d.geometry.PointCloud()
    pcd_remaining.points = o3d.utility.Vector3dVector(original_coord[remaining_indices])
    pcd_remaining.colors = o3d.utility.Vector3dVector(remaining_colors)
    
    remaining_ply = os.path.join(output_dir, "remaining_points.ply")
    o3d.io.write_point_cloud(remaining_ply, pcd_remaining)
    
    print("Showing Floor/Ceiling points...")
    print('Viewer skipped')
    
    print("Showing Remaining points...")
    print('Viewer skipped')
    
    return floor_ceiling_file, remaining_file

def run_round1(output_dir="output"):
    """Execute Round 1: Floor/Ceiling Detection"""
    print("Starting Round 1 execution...")

    torch.cuda.empty_cache()
    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.7)
        except Exception:
            pass
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    sonata.utils.set_seed(24525867)

    try:
        model = sonata.model.load("ckpt/sonata.pth").to(DEVICE)
        print("Model loaded from local checkpoint")
    except Exception:
        custom_config = dict(
            enc_patch_size=[512 for _ in range(5)],
            enable_flash=False,
        )
        model = sonata.load("sonata", repo_id="facebook/sonata", custom_config=custom_config).to(DEVICE)
        print("Model loaded from online")
    
    # Load segmentation head
    try:
        ckpt = sonata.load("ckpt/sonata_linear_prob_head_sc.pth", ckpt_only=True)
    except Exception:
        ckpt = sonata.load("sonata_linear_prob_head_sc", repo_id="facebook/sonata", ckpt_only=True)
    
    seg_head = SegHead(**ckpt["config"]).to(DEVICE)
    seg_head.load_state_dict(ckpt["state_dict"])
    
    # Create transform
    config = [
        dict(type="CenterShift", apply_z=True),
        dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train", 
             return_grid_coord=True, return_inverse=True),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "color", "inverse"), 
             feat_keys=("coord", "color", "normal")),
    ]
    transform = sonata.transform.Compose(config)
    
    # Load data
    npz_data = np.load(os.path.join(output_dir, "complete.npz"))
    print("Available keys in NPZ file:", list(npz_data.keys()))

    original_coord = npz_data["coord"].astype(np.float32)
    original_color = npz_data["color"].astype(np.float32)
    original_normal = npz_data["normal"].astype(np.float32)
    
    print(f"Total point cloud size: {len(original_coord)}")
    
    # Create chunks
    max_points_per_chunk = 300000
    coord_chunks, chunk_indices = create_spatial_chunks(original_coord, max_points_per_chunk)
    
    # Process each chunk
    model.eval()
    seg_head.eval()
    
    all_chunk_predictions = []
    
    print("=== PROCESSING CHUNKS ===")
    for i, (coord_chunk, indices) in enumerate(zip(coord_chunks, chunk_indices)):
        chunk_colors = original_color[indices]
        chunk_normals = original_normal[indices]
        
        chunk_pred = process_single_chunk_round1(model, seg_head, transform, 
                                                coord_chunk, chunk_colors, chunk_normals, i, len(coord_chunks))
        
        if len(chunk_pred) != len(indices):
            if len(chunk_pred) < len(indices):
                most_common = np.bincount(chunk_pred).argmax() if len(chunk_pred) > 0 else 2
                chunk_pred = np.pad(chunk_pred, (0, len(indices) - len(chunk_pred)), 
                                  'constant', constant_values=most_common)
            else:
                chunk_pred = chunk_pred[:len(indices)]
        
        all_chunk_predictions.append(chunk_pred)
    
    # Print newline after progress bar
    print()  
    
    # Merge predictions
    predictions = merge_overlapping_predictions(original_coord, chunk_indices, all_chunk_predictions, original_normal)
    
    # Apply spatial consistency
    predictions_with_structure = apply_spatial_consistency(original_coord, predictions, original_normal)
    
    print("\nROUND 1 COMPLETED!")
    
    # Export floor/ceiling and remaining points
    floor_ceiling_file, remaining_file = export_floor_ceiling_and_remaining(
        original_coord, original_color, original_normal, predictions_with_structure,
        output_dir=output_dir
    )
    
    return model, seg_head, transform

# ============================================================================
# ROUND 2: SONATA CLASSIFICATION
# ============================================================================

# ScanNet-20 constants
SCANNET_COLOR_MAP_20 = {0: (0.0, 0.0, 0.0), 1: (174.0, 199.0, 232.0), 2: (152.0, 223.0, 138.0), 3: (31.0, 119.0, 180.0), 4: (255.0, 187.0, 120.0), 5: (188.0, 189.0, 34.0), 6: (140.0, 86.0, 75.0), 7: (255.0, 152.0, 150.0), 8: (214.0, 39.0, 40.0), 9: (197.0, 176.0, 213.0), 10: (148.0, 103.0, 189.0), 11: (196.0, 156.0, 148.0), 12: (23.0, 190.0, 207.0), 14: (247.0, 182.0, 210.0), 15: (66.0, 188.0, 102.0), 16: (219.0, 219.0, 141.0), 17: (140.0, 57.0, 197.0), 18: (202.0, 185.0, 52.0), 19: (51.0, 176.0, 203.0), 20: (200.0, 54.0, 131.0), 21: (92.0, 193.0, 61.0), 22: (78.0, 71.0, 183.0), 23: (172.0, 114.0, 82.0), 24: (255.0, 127.0, 14.0), 25: (91.0, 163.0, 138.0), 26: (153.0, 98.0, 156.0), 27: (140.0, 153.0, 101.0), 28: (158.0, 218.0, 229.0), 29: (100.0, 125.0, 154.0), 30: (178.0, 127.0, 135.0), 32: (146.0, 111.0, 194.0), 33: (44.0, 160.0, 44.0), 34: (112.0, 128.0, 144.0), 35: (96.0, 207.0, 209.0), 36: (227.0, 119.0, 194.0), 37: (213.0, 92.0, 176.0), 38: (94.0, 106.0, 211.0), 39: (82.0, 84.0, 163.0), 40: (100.0, 85.0, 144.0)}
CLASS_COLOR_20 = [SCANNET_COLOR_MAP_20[id] for id in VALID_CLASS_IDS_20]

def create_simple_chunks(coord, max_points_per_chunk=300000):
    """Create simple chunks without overlap"""
    print(f"Creating simple chunks from {len(coord)} points with max {max_points_per_chunk} points per chunk...")
    
    num_chunks = int(np.ceil(len(coord) / max_points_per_chunk))
    chunks = []
    chunk_indices = []
    
    for i in range(num_chunks):
        start_idx = i * max_points_per_chunk
        end_idx = min((i + 1) * max_points_per_chunk, len(coord))
        
        indices = np.arange(start_idx, end_idx)
        chunk_coords = coord[indices]
        
        chunks.append(chunk_coords)
        chunk_indices.append(indices)
    
    return chunks, chunk_indices

def process_single_chunk_round2(model, seg_head, transform, chunk_coords, chunk_colors, chunk_normals, chunk_idx, total_chunks):
    """Process a single chunk with pure Sonata inference"""
    # Progress bar display
    progress = (chunk_idx + 1) / total_chunks * 100
    bar_length = 30
    filled_length = int(bar_length * (chunk_idx + 1) // total_chunks)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f"\rRound 2 Progress: |{bar}| {progress:.1f}% ({chunk_idx + 1}/{total_chunks}) chunks", end='', flush=True)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        point_data = {
            'coord': chunk_coords,
            'color': chunk_colors,
            'normal': chunk_normals
        }
        
        transformed_point = transform(point_data)
        
        with torch.inference_mode():
            for key in transformed_point.keys():
                if isinstance(transformed_point[key], torch.Tensor):
                    transformed_point[key] = transformed_point[key].to(DEVICE, non_blocking=True)
            
            result = model(transformed_point)
            
            while "pooling_parent" in result.keys():
                torch.cuda.empty_cache()
                parent = result.pop("pooling_parent")
                inverse = result.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, result.feat[inverse]], dim=-1)
                result = parent
            
            features = result.feat
            logits = seg_head(features)
            predictions = logits.argmax(dim=-1).cpu().numpy()
            processed_coords = result.coord.cpu().numpy()
            
            tree = cKDTree(processed_coords)
            distances, nn_indices = tree.query(chunk_coords, k=1)
            final_predictions = predictions[nn_indices]
            
            del result, features, logits
            torch.cuda.empty_cache()
        
        return final_predictions
        
    except Exception as e:
        print(f"\nError in chunk {chunk_idx + 1}: {e}")
        return np.ones(len(chunk_coords), dtype=np.int32) * 19

def simple_merge_predictions(original_coord, chunk_indices, chunk_predictions):
    """Simple merge - no overlap, just concatenate predictions"""
    print("Merging chunk predictions...")
    
    final_predictions = np.zeros(len(original_coord), dtype=np.int32)
    
    for indices, pred in zip(chunk_indices, chunk_predictions):
        if len(pred) != len(indices):
            if len(pred) < len(indices):
                most_common = np.bincount(pred).argmax() if len(pred) > 0 else 19
                pred = np.pad(pred, (0, len(indices) - len(pred)), 
                             'constant', constant_values=most_common)
            else:
                pred = pred[:len(indices)]
        
        final_predictions[indices] = pred
    
    return final_predictions

def save_as_ply(coord, color, predictions, filename):
    """Save point cloud with semantic colors as PLY file"""
    print(f"Saving PLY file: {filename}")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    
    semantic_colors = np.zeros((len(coord), 3))
    for i, pred in enumerate(predictions):
        if pred < len(CLASS_COLOR_20):
            semantic_colors[i] = np.array(CLASS_COLOR_20[pred]) / 255.0
        else:
            semantic_colors[i] = [0.0, 0.0, 0.0]
    
    pcd.colors = o3d.utility.Vector3dVector(semantic_colors)
    
    success = o3d.io.write_point_cloud(filename, pcd)
    return success

def save_original_color_ply(coord, color, filename):
    """Save point cloud with original colors as PLY file"""
    print(f"Saving original color PLY file: {filename}")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    
    if color.max() > 1.0:
        normalized_colors = color / 255.0
    else:
        normalized_colors = color
    
    pcd.colors = o3d.utility.Vector3dVector(normalized_colors)
    
    success = o3d.io.write_point_cloud(filename, pcd)
    return success

def save_as_ply_with_priority(coord, color, predictions, filename, num_floor_ceiling_points):
    """Save point cloud with priority-based colors"""
    print(f"Saving PLY file with priority-based colors: {filename}")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    
    semantic_colors = np.zeros((len(coord), 3))
    
    for i, pred in enumerate(predictions):
        if i < num_floor_ceiling_points:
            # Floor/ceiling points - apply custom colors ONLY for floor(2) and ceiling(22)
            if pred == 2:  # Floor from floor/ceiling data gets custom green
                semantic_colors[i] = np.array([152.0, 223.0, 138.0]) / 255.0
            elif pred == 22:  # Ceiling from floor/ceiling data gets custom yellow
                semantic_colors[i] = np.array([255.0, 255.0, 150.0]) / 255.0
            else:
                if pred < len(CLASS_COLOR_20):
                    semantic_colors[i] = np.array(CLASS_COLOR_20[pred]) / 255.0
                else:
                    semantic_colors[i] = [0.0, 0.0, 0.0]
        else:
            # Sonata classified points - ALWAYS use standard ScanNet-20 colors
            if pred < len(CLASS_COLOR_20):
                semantic_colors[i] = np.array(CLASS_COLOR_20[pred]) / 255.0
            else:
                semantic_colors[i] = [0.0, 0.0, 0.0]
    
    pcd.colors = o3d.utility.Vector3dVector(semantic_colors)
    
    success = o3d.io.write_point_cloud(filename, pcd)
    return success

def create_class_legend_file(filename="class_legend.txt"):
    """Create a text file with class labels and their corresponding colors"""
    print(f"Creating class legend file: {filename}")
    
    with open(filename, 'w') as f:
        f.write("ScanNet-20 Class Legend\n")
        f.write("=======================\n\n")
        f.write("Class ID | Class Name        | RGB Color\n")
        f.write("---------|-------------------|----------\n")
        
        for i, (class_name, color) in enumerate(zip(CLASS_LABELS_20, CLASS_COLOR_20)):
            f.write(f"{i:8d} | {class_name:17s} | ({color[0]:3.0f}, {color[1]:3.0f}, {color[2]:3.0f})\n")
        
        f.write("\nVisualization Notes:\n")
        f.write("- semantic_classification.ply: Points colored by predicted class\n")
        f.write("- original_colors.ply: Points with original RGB colors\n")
        f.write("- Use MeshLab, CloudCompare, or similar tools to view PLY files\n")

def run_round2(model, seg_head, transform, output_dir="output"):
    """Execute Round 2: Sonata Classification"""
    print("=== ROUND 2: PURE SONATA CLASSIFICATION ===")

    # Load remaining points data
    npz_data = np.load(os.path.join(output_dir, "remaining_points.npz"))
    
    original_coord = npz_data["coord"].astype(np.float32)
    original_color = npz_data["color"].astype(np.float32)
    original_normal = npz_data["normal"].astype(np.float32)
    
    print(f"Total remaining points to classify: {len(original_coord)}")
    
    # Create simple chunks
    max_points_per_chunk = 300000
    coord_chunks, chunk_indices = create_simple_chunks(original_coord, max_points_per_chunk)
    
    # Process each chunk with pure Sonata
    model.eval()
    seg_head.eval()
    
    all_chunk_predictions = []
    
    for i, (coord_chunk, indices) in enumerate(zip(coord_chunks, chunk_indices)):
        chunk_colors = original_color[indices]
        chunk_normals = original_normal[indices]
        
        chunk_pred = process_single_chunk_round2(model, seg_head, transform, 
                                                coord_chunk, chunk_colors, chunk_normals, i, len(coord_chunks))
        
        all_chunk_predictions.append(chunk_pred)
    
    # Print newline after progress bar
    print()
    
    # Simple merge
    final_predictions = simple_merge_predictions(original_coord, chunk_indices, all_chunk_predictions)
    
    # Print class distribution
    print("\n=== RAW SONATA CLASSIFICATION RESULTS ===")
    unique_classes, counts = np.unique(final_predictions, return_counts=True)
    total_points = len(final_predictions)
    
    for class_id, count in zip(unique_classes, counts):
        if class_id < len(CLASS_LABELS_20):
            class_name = CLASS_LABELS_20[class_id]
            percentage = (count / total_points) * 100
            print(f"Class {class_id:2d} ({class_name:15s}): {count:8d} points ({percentage:5.1f}%)")
    
    # Save results
    np.savez(os.path.join(output_dir, "raw_sonata_classification.npz"),
             coord=original_coord,
             color=original_color,
             normal=original_normal,
             predictions=final_predictions,
             class_names=CLASS_LABELS_20)

    # Save PLY files
    save_as_ply(original_coord, original_color, final_predictions,
                os.path.join(output_dir, "semantic_classification.ply"))

    save_original_color_ply(original_coord, original_color,
                           os.path.join(output_dir, "original_colors.ply"))

    create_class_legend_file(os.path.join(output_dir, "class_legend.txt"))
    
    # === MERGE WITH FLOOR_CEILING.NPZ ===
    print("\n=== MERGING WITH FLOOR_CEILING DATA ===")
    
    try:
        # Load floor/ceiling data
        floor_ceiling_data = np.load(os.path.join(output_dir, "floor_ceiling.npz"))
        
        floor_coord = floor_ceiling_data["coord"].astype(np.float32)
        floor_color = floor_ceiling_data["color"].astype(np.float32)
        floor_predictions = floor_ceiling_data["predictions"].astype(np.int32)
        
        print(f"Floor/ceiling points: {len(floor_coord)}")
        print(f"Sonata classified points: {len(original_coord)}")
        
        # Apply specific class mapping for floor/ceiling during merge
        modified_floor_predictions = floor_predictions.copy()
        
        unique_floor_classes = np.unique(floor_predictions)
        
        if len(unique_floor_classes) == 2:
            min_class = unique_floor_classes.min()
            max_class = unique_floor_classes.max()
            
            modified_floor_predictions[floor_predictions == min_class] = 2  # Floor
            modified_floor_predictions[floor_predictions == max_class] = 22  # Ceiling
        else:
            modified_floor_predictions[:] = 2
        
        # Merge coordinates, colors, and predictions
        merged_coord = np.vstack([floor_coord, original_coord])
        merged_color = np.vstack([floor_color, original_color])
        merged_predictions = np.concatenate([modified_floor_predictions, final_predictions])
        
        print(f"Total merged points: {len(merged_coord)}")
        
        # Save merged results
        np.savez(os.path.join(output_dir, "merged_classification.npz"),
                 coord=merged_coord,
                 color=merged_color,
                 predictions=merged_predictions,
                 class_names=CLASS_LABELS_20,
                 num_floor_ceiling_points=len(floor_coord),
                 num_sonata_points=len(original_coord))
        
        # Save merged PLY files
        save_as_ply_with_priority(merged_coord, merged_color, merged_predictions,
                                os.path.join(output_dir, "merged_semantic_classification.ply"),
                                num_floor_ceiling_points=len(floor_coord))

        save_original_color_ply(merged_coord, merged_color,
                               os.path.join(output_dir, "merged_original_colors.ply"))
        
        print("\nPLY files saved. Use MeshLab or CloudCompare for visualization.")

    except FileNotFoundError:
        print(f"ERROR: '{os.path.join(output_dir, 'floor_ceiling.npz')}' not found!")
    except Exception as e:
        print(f"ERROR during merging: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Complete LAS Processing and Two-Round Classification Pipeline")
    parser.add_argument("las_file", help="Path to input LAS file")
    parser.add_argument("--subsample-distance", type=float, default=0.010,
                        help="Minimum distance between points for subsampling (default: 0.010)")
    parser.add_argument("--output-dir", default="output",
                        help="Output directory (default: output)")
    args = parser.parse_args()

    print("="*80)
    print("COMPLETE LAS PROCESSING AND TWO-ROUND CLASSIFICATION PIPELINE")
    print("="*80)

    # === LAS CONVERSION ===
    print("\nStarting LAS conversion...")
    conversion_success = run_las_converter(
        las_file=args.las_file,
        subsample_distance=args.subsample_distance,
        output_dir=args.output_dir,
    )

    if not conversion_success:
        print("LAS conversion failed. Exiting pipeline.")
        sys.exit(1)

    print("\n" + "="*50)
    print("Starting Round 1: Floor/Ceiling Detection")
    print("="*50)

    # === ROUND 1 EXECUTION ===
    model, seg_head, transform = run_round1(output_dir=args.output_dir)

    print("\n" + "="*50)
    print("Starting Round 2: Sonata Classification")
    print("="*50)

    # === ROUND 2 EXECUTION ===
    run_round2(model, seg_head, transform, output_dir=args.output_dir)

    print("\n" + "="*80)
    print("COMPLETE PIPELINE FINISHED!")
    print("="*80)
    print("Complete pipeline finished successfully!")
    print("Check the merged PLY files for the complete classified scene.")
