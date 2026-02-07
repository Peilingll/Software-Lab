# 03_wall_ransac.py - REFINED
import os, sys, csv, math, numpy as np, open3d as o3d
np.random.seed(42)


def angle_deg(v1, v2):
    """Calculate angle between two vectors in degrees."""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    c = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return math.degrees(math.acos(c))


def main(in_ply, dist_thr=0.03, min_points=800, max_planes=50, vertical_tol_deg=15.0,
         min_wall_area=2.0, min_wall_length=1.0, min_wall_height=1.5, max_wall_thickness=0.5):
    """
    Detect vertical wall planes using RANSAC with size constraints.
    """
    # Load point cloud
    pcd = o3d.io.read_point_cloud(in_ply)
    if len(pcd.points) == 0:
        raise SystemExit("Empty input file")
    
    # Estimate normals for plane detection
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.10, max_nn=50))

    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if len(pcd.colors) else None
    remain = pcd
    planes = []
    colored_pts = []
    colored_cols = []

    # Z-axis for verticality check (walls should be perpendicular to Z)
    z_axis = np.array([0, 0, 1])

    print(f"\nRANSAC Parameters:")
    print(f"  Distance threshold: {dist_thr}m")
    print(f"  Min points per plane: {min_points}")
    print(f"  Verticality tolerance: ±{vertical_tol_deg}°")
    print(f"  Min wall area: {min_wall_area}m²")
    print(f"  Min wall length: {min_wall_length}m")
    print(f"  Min wall height: {min_wall_height}m")
    print(f"  Max wall thickness: {max_wall_thickness}m\n")

    # RANSAC loop
    for k in range(max_planes):
        if len(remain.points) < min_points:
            break
        
        # Detect plane
        plane_model, inliers = remain.segment_plane(
            distance_threshold=dist_thr,
            ransac_n=3,
            num_iterations=2000
        )
        
        [a, b, c, d] = plane_model
        n = np.array([a, b, c])
        
        if np.linalg.norm(n) < 1e-6 or len(inliers) < min_points:
            break

        # Check 1: Verticality (wall normals should be ~horizontal)
        angle_from_vertical = abs(angle_deg(n, z_axis) - 90.0)
        if angle_from_vertical > vertical_tol_deg:
            # print(f"  Plane {k+1}: Rejected (not vertical, angle={angle_from_vertical:.1f}°)")
            remain = remain.select_by_index(inliers, invert=True)
            continue

        # Extract plane points
        in_cloud = remain.select_by_index(inliers)
        
        # --- (REFINED) START: Cluster to find main wall segment and remove noise ---
        labels = np.array(in_cloud.cluster_dbscan(eps=0.15, min_points=50, print_progress=False))
        
        if len(labels) == 0:
            remain = remain.select_by_index(inliers, invert=True)
            continue


        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        if len(counts) == 0:
            # print(f"  Plane {k+1}: Rejected (all points classified as noise)")
            remain = remain.select_by_index(inliers, invert=True)
            continue
            
        main_cluster_label = unique_labels[np.argmax(counts)]
        

        main_cluster_indices = np.where(labels == main_cluster_label)[0]
        

        if len(main_cluster_indices) < min_points:
            # print(f"  Plane {k+1}: Rejected (main cluster too small: {len(main_cluster_indices)} pts)")
            remain = remain.select_by_index(inliers, invert=True)
            continue
            
        in_cloud = in_cloud.select_by_index(main_cluster_indices)
        wall_points = np.asarray(in_cloud.points)
        # --- (REFINED) END: Cluster step ---

        # Calculate bounding box dimensions (on CLEANED points)
        min_bound = wall_points.min(axis=0)
        max_bound = wall_points.max(axis=0)
        dims = max_bound - min_bound
        
        # Sort dimensions: smallest = thickness, medium = width, largest = height
        sorted_dims = sorted(dims)
        thickness = sorted_dims[0]
        width = sorted_dims[1]
        height = sorted_dims[2]
        
        # Check 2: Minimum wall area
        wall_area = width * height
        if wall_area < min_wall_area:
            print(f"  Plane {k+1}: Rejected (area {wall_area:.2f}m² < {min_wall_area}m²)")
            remain = remain.select_by_index(inliers, invert=True)
            continue
        
        # Check 3: Minimum wall length
        if width < min_wall_length:
            print(f"  Plane {k+1}: Rejected (length {width:.2f}m < {min_wall_length}m)")
            remain = remain.select_by_index(inliers, invert=True)
            continue
        
        # Check 4: Minimum wall height
        if height < min_wall_height:
            print(f"  Plane {k+1}: Rejected (height {height:.2f}m < {min_wall_height}m)")
            remain = remain.select_by_index(inliers, invert=True)
            continue
        
        # Check 5: Maximum wall thickness
        if thickness > max_wall_thickness:
            print(f"  Plane {k+1}: Rejected (thickness {thickness:.2f}m > {max_wall_thickness}m)")
            remain = remain.select_by_index(inliers, invert=True)
            continue

        # Only remove main cluster points from remain (not all inliers).
        # Non-main-cluster points stay in remain for future RANSAC iterations.
        inliers_arr = np.array(inliers)
        main_inlier_indices = inliers_arr[main_cluster_indices].tolist()
        out_cloud = remain.select_by_index(main_inlier_indices, invert=True)

        # Assign unique color to this plane
        color = np.random.default_rng(k).random(3)
        planes.append((plane_model, len(wall_points), color))

        in_cols = np.tile(color, (len(wall_points), 1))
        colored_pts.append(wall_points)
        colored_cols.append(in_cols)

        print(f"  Plane {k+1}: Accepted ({len(wall_points)} pts, "
              f"{width:.2f}m × {thickness:.2f}m × {height:.2f}m, area={wall_area:.2f}m²)")

        remain = out_cloud

    # Check if any planes were found
    if not planes:
        raise SystemExit("\nNo vertical wall planes found meeting the criteria. "
                        "Try relaxing the thresholds or check input data.")

    print(f"\nTotal walls detected: {len(planes)}")

    # Merge all detected planes
    all_pts = np.vstack(colored_pts)
    all_cols = np.vstack(colored_cols)
    out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_pts))
    out.colors = o3d.utility.Vector3dVector(all_cols)

    # Save colored point cloud
    out_dir = os.path.dirname(in_ply) or "."
    out_path = os.path.join(out_dir, "walls_ransac_colored.ply")
    o3d.io.write_point_cloud(out_path, out)

    # Save plane parameters to CSV
    csv_path = os.path.join(out_dir, "walls_ransac_planes.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plane_id", "a", "b", "c", "d", "num_points", "r", "g", "b_color"])
        for i, (pm, npts, color) in enumerate(planes, start=1):
            a, b, c, d = pm
            r_int = int(round(color[0] * 255))
            g_int = int(round(color[1] * 255))
            b_int = int(round(color[2] * 255))
            w.writerow([i, a, b, c, d, npts, r_int, g_int, b_int])

    print(f"\nOutput files:")
    print(f"  Colored PLY: {out_path}")
    print(f"  Plane CSV: {csv_path}")


if __name__ == "__main__":
    """
    RANSAC Wall Detection with Size Constraints
    
    Usage:
        python 03_wall_ransac.py <input.ply> [dist_thr] [min_pts] [max_planes] [vtol] [min_area] [min_len] [min_height]
    
    Example:
        python 03_wall_ransac.py walls/walls_prep.ply 0.03 800 50 15 2.0 1.0 1.5
    
    Parameters:
        input.ply    - Input wall point cloud (preprocessed)
        dist_thr     - RANSAC distance threshold in meters (default: 0.03)
        min_pts      - Minimum points per plane (default: 800)
        max_planes   - Maximum planes to detect (default: 50)
        vtol         - Verticality tolerance in degrees (default: 15)
        min_area     - Minimum wall area in m² (default: 2.0)
        min_len      - Minimum wall length in meters (default: 1.0)
        min_height   - Minimum wall height in meters (default: 1.5)
    """
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    in_ply = sys.argv[1]
    dist_thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
    min_pts = int(sys.argv[3]) if len(sys.argv) > 3 else 800
    max_p = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    vtol = float(sys.argv[5]) if len(sys.argv) > 5 else 15.0
    
    # Size constraint parameters
    min_area = float(sys.argv[6]) if len(sys.argv) > 6 else 2.0      # Minimum area: 2m²
    min_length = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0    # Minimum length: 1m
    min_height = float(sys.argv[8]) if len(sys.argv) > 8 else 1.5    # Minimum height: 1.5m
    max_thick = float(sys.argv[9]) if len(sys.argv) > 9 else 0.5     # Maximum thickness: 0.5m
    
    main(in_ply, dist_thr, min_pts, max_p, vtol, min_area, min_length, min_height, max_thick)
