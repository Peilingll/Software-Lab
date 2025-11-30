# Save as: script/05_compute_bbox_corrected.py
# (This REPLACES your old file)
# NEW: Adds --visualize-dir and cleans noise from each wall.
# (REFINED) Replaced statistical outlier with DBSCAN for cleaning.

import open3d as o3d
import numpy as np
import json
import glob
import os
import sys
import argparse
import csv

def compute_wall_bbox(ply_path, plane_params=None, clean_pcd=True, visualize_dir=None, wall_id=0):
    """
    Calculate the bounding box and orientation of a wall.
    NEW: Includes cleaning and optional visualization.

    Returns:
        dict: Contains center, dimensions, rotation_matrix, normal
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    
    if len(pcd.points) < 10:
        return None
    
    # --- (REFINED) START: Use DBSCAN to find the main cluster ---
    if clean_pcd:
        
        labels = np.array(pcd.cluster_dbscan(eps=0.2, min_points=20, print_progress=False))
        
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        if len(counts) == 0:
            return None 
            
        main_cluster_label = unique_labels[np.argmax(counts)]
        
        pcd = pcd.select_by_index(np.where(labels == main_cluster_label)[0])
    # --- (REFINED) END: DBSCAN cluster step ---

    if len(pcd.points) < 10:
        return None

    # Calculate Oriented Bounding Box (OBB)
    obb = pcd.get_oriented_bounding_box()
    
    # Extract parameters
    center = obb.center.tolist()
    extent = obb.extent.tolist()  # [width, height, depth]
    R = obb.R  # 3x3 rotation matrix
    
    # Sort dimensions to identify thickness
    dims_sorted = sorted(enumerate(extent), key=lambda x: x[1])
    thickness_idx = dims_sorted[0][0]
    
    bbox_data = {
        'center': center,
        'extent': extent,
        'rotation_matrix': R.tolist(),
        'thickness_idx': thickness_idx,
        'num_points': len(pcd.points)
    }
    
    if plane_params:
        bbox_data['plane_normal'] = plane_params['normal']
        bbox_data['plane_d'] = plane_params['d']

    # --- NEW: Save visualization ---
    if visualize_dir:
        try:
            os.makedirs(visualize_dir, exist_ok=True) 
            pcd_path = os.path.join(visualize_dir, f"wall_{wall_id}_cleaned.ply")
            o3d.io.write_point_cloud(pcd_path, pcd, write_ascii=True)
            bbox_corners = np.asarray(obb.get_box_points())
            bbox_pcd = o3d.geometry.PointCloud()
            bbox_pcd.points = o3d.utility.Vector3dVector(bbox_corners)
            bbox_pcd.paint_uniform_color([1.0, 0.0, 0.0]) 
            bbox_path = os.path.join(visualize_dir, f"wall_{wall_id}_bbox.ply")
            o3d.io.write_point_cloud(bbox_path, bbox_pcd, write_ascii=True)
            

        except Exception as e:
            print(f"Warning: Could not save visualization for wall {wall_id}: {e}")
            
    return bbox_data

def load_plane_params(csv_path):
    """Load RANSAC plane parameters"""
    planes = {}
    
    if not os.path.exists(csv_path):
        print(f"Warning: Plane parameter file not found: {csv_path}")
        return planes
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:

                plane_id = int(row['plane_id'])
                planes[plane_id] = {
                    'normal': [float(row['a']), float(row['b']), float(row['c'])],
                    'd': float(row['d']),
                    'num_points': int(row['num_points'])
                }
            except (ValueError, KeyError):
                print(f"Warning: Skipping malformed row in CSV: {row}")
    
    return planes

def find_plane_csv(search_dirs):
    """Search for the plane CSV file in common directories"""
    csv_names = ["walls_ransac_planes.csv", "ransac_planes.csv", "planes.csv"]
    
    for dir_path in search_dirs:
        for csv_name in csv_names:
            csv_path = os.path.join(dir_path, csv_name)
            if os.path.exists(csv_path):
                return csv_path
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Calculate bounding boxes for wall segments")
    parser.add_argument(
        "--input", "-i",
        default="per_wall",
        help="Input directory containing wall_*.ply files (default: per_wall)"
    )
    parser.add_argument(
        "--planes-csv", "-p",
        help="Path to RANSAC plane parameters CSV file (optional)"
    )
    parser.add_argument(
        "--output", "-o",
        default=".",
        help="Output directory (default: current directory)"
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix to add to output filenames (e.g., 'batch1')"
    )
    # --- (REFINED) Flag is still here, but cleaning method is now DBSCAN ---
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Disable the DBSCAN clustering step"
    )
    # --- NEW: Optional directory for visualization ---
    parser.add_argument(
        "--visualize-dir",
        default=None,
        help="Directory to save debug PLY files (point cloud + bbox)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input directory not found: {args.input}")
        sys.exit(1)
    
    if args.planes_csv:
        plane_csv = args.planes_csv
    else:

        search_dirs = [args.input, ".", "output_clean", "structure", "walls", ".."]

        search_dirs.append(os.path.join(args.input, "..", "output_clean")) 
        plane_csv = find_plane_csv(search_dirs)
        if plane_csv:
            print(f"Found plane parameter file: {plane_csv}")
    
    planes = {}
    if plane_csv and os.path.exists(plane_csv):
        planes = load_plane_params(plane_csv)
        print(f"Loaded {len(planes)} plane parameters")
    
    wall_pattern = os.path.join(args.input, "wall_*.ply")
    wall_files = sorted(glob.glob(wall_pattern))
    
    # Filter out any files we might have created
    wall_files = [f for f in wall_files if "_scalar" not in f and "_bbox_debug" not in f]
    
    if not wall_files:
        print(f"Error: No wall_*.ply files found in {args.input}")
        sys.exit(1)
    
    print(f"Found {len(wall_files)} wall files\n")
    
    all_bboxes = {}
    wall_summary = []
    
    if args.visualize_dir:
        print(f"Saving BBox visualizations to: {args.visualize_dir}")
    
    for wf in wall_files:
        basename = os.path.basename(wf)
        try:
            # Try to parse ID like 'wall_1.ply'
            wall_id_str = basename.split('_')[1].split('.')[0]
            wall_id = int(wall_id_str)
        except:
            print(f"Warning: Could not parse wall ID from filename: {basename}. Skipping.")
            continue
        
        # (REFINED) 根據 wall_id 獲取平面
        plane_params = planes.get(wall_id)
        if plane_params is None:
            # (REFINED) 應對 04_split_walls_corrected.py 中 i 和 plane_id 不匹配的情況
            # 嘗試使用索引
            plane_params = planes.get(wall_id) 
            
        
        bbox = compute_wall_bbox(
            wf, 
            plane_params, 
            clean_pcd=(not args.no_clean), 
            visualize_dir=args.visualize_dir, 
            wall_id=wall_id
        )
        
        if bbox is None:
            print(f"Wall {wall_id:2d}: Skipped (empty after cleaning)")
            continue
        
        all_bboxes[wall_id] = bbox
        
        dims = bbox['extent']
        sorted_dims = sorted(dims)
        
        print(f"Wall {wall_id:2d}: "
              f"{sorted_dims[1]:.2f}m (W) × "
              f"{sorted_dims[0]:.2f}m (T) × "
              f"{sorted_dims[2]:.2f}m (H) | "
              f"{bbox['num_points']} points")
        
        wall_summary.append({
            'wall_id': wall_id,
            'center_x': bbox['center'][0],
            'center_y': bbox['center'][1],
            'center_z': bbox['center'][2],
            'width': sorted_dims[1],
            'thickness': sorted_dims[0],
            'height': sorted_dims[2],
            'num_points': bbox['num_points']
        })
    
    os.makedirs(args.output, exist_ok=True)
    
    json_filename = f"wall_bboxes_{args.suffix}.json" if args.suffix else "wall_bboxes.json"
    output_json = os.path.join(args.output, json_filename)
    with open(output_json, 'w') as f:
        json.dump(all_bboxes, f, indent=2)
    
    print(f"\nBounding boxes (math) saved to: {output_json}")
    
    csv_filename = f"wall_bboxes_summary_{args.suffix}.csv" if args.suffix else "wall_bboxes_summary.csv"
    summary_csv = os.path.join(args.output, csv_filename)
    with open(summary_csv, 'w', newline='') as f:
        if wall_summary:
            fieldnames = wall_summary[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(wall_summary)
    
    print(f"Summary saved to: {summary_csv}")
    
    if wall_summary:
        total_points = sum(w['num_points'] for w in wall_summary)
        avg_thickness = np.mean([w['thickness'] for w in wall_summary])
        
        print(f"\nStatistics (after cleaning):")
        print(f"  Total walls: {len(wall_summary)}")
        print(f"  Total points: {total_points:,}")
        print(f"  Avg. thickness: {avg_thickness:.3f}m")

if __name__ == "__main__":
    main()
