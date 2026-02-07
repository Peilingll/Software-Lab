# Save as: script/04_split_walls_corrected.py
import numpy as np
import open3d as o3d
import os
import sys
import csv
import math
import argparse
import json


def merge_coplanar_walls(wall_groups, planes_csv, angle_thr=10.0, dist_thr=0.3):
    """
    Group walls whose RANSAC plane params are nearly identical, then
    concatenate their points.

    Args:
        wall_groups: list of (wall_xyz, wall_rgb, plane_id) from color-based splitting
        planes_csv: dict {plane_id: {'normal': [a,b,c], 'd': float}}
        angle_thr: max angle between normals (degrees) to consider same plane
        dist_thr: max |d1-d2| to consider same plane (meters)

    Returns:
        merged_groups: list of (merged_xyz, merged_rgb)
    """
    n = len(wall_groups)
    if n == 0:
        return []

    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Build adjacency by comparing plane parameters
    for i in range(n):
        pid_i = wall_groups[i][2]
        info_i = planes_csv.get(pid_i)
        if info_i is None:
            continue
        ni = np.array(info_i['normal'])
        di = info_i['d']

        for j in range(i + 1, n):
            pid_j = wall_groups[j][2]
            info_j = planes_csv.get(pid_j)
            if info_j is None:
                continue
            nj = np.array(info_j['normal'])
            dj = info_j['d']

            # Angle between normals (handle antiparallel normals)
            ni_norm = ni / (np.linalg.norm(ni) + 1e-12)
            nj_norm = nj / (np.linalg.norm(nj) + 1e-12)
            cos_angle = np.clip(np.dot(ni_norm, nj_norm), -1.0, 1.0)
            angle = math.degrees(math.acos(abs(cos_angle)))

            # For antiparallel normals, flip d sign for comparison
            if cos_angle < 0:
                dj_cmp = -dj
            else:
                dj_cmp = dj

            if angle < angle_thr and abs(di - dj_cmp) < dist_thr:
                union(i, j)

    # Group by root
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    # Merge point clouds
    merged = []
    for indices in groups.values():
        xyz_list = [wall_groups[i][0] for i in indices]
        rgb_list = [wall_groups[i][1] for i in indices]
        merged_xyz = np.vstack(xyz_list)
        merged_rgb = np.vstack(rgb_list)
        merged.append((merged_xyz, merged_rgb))

    return merged


def split_by_spatial_clustering(xyz, rgb, eps=0.5, min_points=100):
    """
    Use DBSCAN to split a point cloud into spatially connected components.

    Returns:
        list of (segment_xyz, segment_rgb)
    """
    if len(xyz) < min_points:
        return [(xyz, rgb)]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

    segments = []
    unique_labels = set(labels)
    for lbl in sorted(unique_labels):
        if lbl < 0:
            continue  # skip noise
        mask = labels == lbl
        seg_xyz = xyz[mask]
        seg_rgb = rgb[mask]
        if len(seg_xyz) >= min_points:
            segments.append((seg_xyz, seg_rgb))

    # If no clusters found, return everything as one segment
    if not segments:
        return [(xyz, rgb)]

    return segments


def load_planes_csv(csv_path):
    """
    Load RANSAC plane parameters from CSV.

    Returns:
        planes: dict {plane_id (int): {'normal': [a,b,c], 'd': float}}
        color_to_plane_id: dict {(r,g,b) int tuple: plane_id}
    """
    planes = {}
    color_to_pid = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row['plane_id'])
            a = float(row['a'])
            b = float(row['b'])
            c = float(row['c'])
            d = float(row['d'])
            planes[pid] = {'normal': [a, b, c], 'd': d}
            # Color columns may or may not exist (backward compat)
            if 'r' in row and 'g' in row and 'b_color' in row:
                r = int(row['r'])
                g = int(row['g'])
                b_val = int(row['b_color'])
                color_to_pid[(r, g, b_val)] = pid
    return planes, color_to_pid


def split_walls_by_color(colored_ply, output_dir, planes_csv_path=None,
                         merge_angle=10.0, merge_dist=0.3, cluster_eps=0.5):
    """
    Split walls_ransac_colored.ply into individual wall_*.ply files.

    1. Split by unique RGB color (each color = one RANSAC plane)
    2. If planes_csv provided: merge coplanar walls (duplicate detections)
    3. Spatial clustering on each merged group to separate disconnected segments
    4. Each cluster becomes a separate wall file

    Args:
        colored_ply: Path to colored PLY file from RANSAC
        output_dir: Output directory for individual wall files
        planes_csv_path: Optional path to RANSAC plane parameters CSV
        merge_angle: Max angle (degrees) between normals to merge planes
        merge_dist: Max plane_d difference to merge planes (meters)
        cluster_eps: DBSCAN eps for spatial splitting after merge (meters)
    """
    os.makedirs(output_dir, exist_ok=True)

    pcd = o3d.io.read_point_cloud(colored_ply)
    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors)

    if len(xyz) == 0:
        print("Warning: Input file is empty")
        return

    # --- Step 1: Split by color (each color = one RANSAC plane) ---
    rgb_int = (rgb * 255).round().astype(int)
    unique_colors = np.unique(rgb_int, axis=0)

    print(f"Found {len(unique_colors)} unique RANSAC planes (by color)")

    # Build wall groups: (xyz, rgb, color_tuple)
    # color_tuple is used later to map to plane_id via CSV
    wall_groups_raw = []  # list of (wall_xyz, wall_rgb, color_tuple)

    for i, color in enumerate(unique_colors):
        mask = np.all(rgb_int == color, axis=1)
        wall_xyz = xyz[mask]
        wall_rgb = rgb[mask]

        if len(wall_xyz) < 100:
            print(f"  Skipping color group {i + 1}: Too few points ({len(wall_xyz)} < 100)")
            continue

        wall_groups_raw.append((wall_xyz, wall_rgb, tuple(color.tolist())))

    # --- Step 2: Merge coplanar walls if plane CSV is available ---
    if planes_csv_path and os.path.exists(planes_csv_path):
        planes_data, color_to_pid = load_planes_csv(planes_csv_path)

        # Map each wall group to its plane_id using color matching
        wall_groups = []
        for wall_xyz, wall_rgb, color_tuple in wall_groups_raw:
            pid = color_to_pid.get(color_tuple, -1)
            if pid == -1:
                print(f"  Warning: No plane_id found for color {color_tuple}, skipping merge for this group")
            wall_groups.append((wall_xyz, wall_rgb, pid))

        print(f"\nMerging coplanar walls (angle_thr={merge_angle}°, dist_thr={merge_dist}m)...")
        merged_groups = merge_coplanar_walls(wall_groups, planes_data,
                                             angle_thr=merge_angle, dist_thr=merge_dist)
        print(f"  {len(wall_groups)} RANSAC planes → {len(merged_groups)} merged groups")
    else:
        if planes_csv_path:
            print(f"Warning: planes CSV not found at {planes_csv_path}, skipping merge")
        merged_groups = [(wg[0], wg[1]) for wg in wall_groups_raw]

    # --- Step 3: Spatial clustering on each merged group ---
    print(f"\nSpatial splitting (DBSCAN eps={cluster_eps}m)...")
    wall_info = []
    wall_idx = 1

    for group_i, (group_xyz, group_rgb) in enumerate(merged_groups):
        segments = split_by_spatial_clustering(group_xyz, group_rgb,
                                               eps=cluster_eps, min_points=100)

        for seg_xyz, seg_rgb in segments:
            wall_pcd = o3d.geometry.PointCloud()
            wall_pcd.points = o3d.utility.Vector3dVector(seg_xyz)
            wall_pcd.colors = o3d.utility.Vector3dVector(seg_rgb)

            out_path = os.path.join(output_dir, f"wall_{wall_idx}.ply")
            o3d.io.write_point_cloud(out_path, wall_pcd)

            print(f"  wall_{wall_idx}.ply: {len(seg_xyz)} points (from merged group {group_i + 1})")

            wall_info.append({
                'id': wall_idx,
                'num_points': len(seg_xyz),
                'merged_group': group_i + 1
            })
            wall_idx += 1

    info_file = os.path.join(output_dir, "wall_info.json")
    with open(info_file, 'w') as f:
        json.dump(wall_info, f, indent=2)

    print(f"\nTotal output walls: {wall_idx - 1}")
    print(f"Wall info saved to: {info_file}")
    return wall_info


def main():
    parser = argparse.ArgumentParser(description="Split RANSAC wall planes based on color")
    parser.add_argument(
        "input",
        help="Input colored PLY file (usually walls_ransac_colored.ply)"
    )
    parser.add_argument(
        "--output", "-o",
        default="per_wall",
        help="Output directory (default: per_wall)"
    )
    parser.add_argument("--planes-csv", default=None,
        help="Path to RANSAC plane parameters CSV (for merging duplicate planes)")
    parser.add_argument("--merge-angle", type=float, default=10.0,
        help="Max angle (degrees) between normals to merge planes (default: 10)")
    parser.add_argument("--merge-dist", type=float, default=0.3,
        help="Max plane_d difference to merge planes (default: 0.3m)")
    parser.add_argument("--cluster-eps", type=float, default=0.5,
        help="DBSCAN eps for spatial splitting after merge (default: 0.5m)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        possible_paths = [
            args.input,
            os.path.join("output_clean", "walls_ransac_colored.ply"),
            "walls_ransac_colored.ply"
        ]
        found = False
        for path in possible_paths:
            if os.path.exists(path):
                args.input = path
                found = True
                print(f"Found input file: {path}")
                break

        if not found:
            print(f"Error: Input file not found: {args.input}")
            sys.exit(1)

    split_walls_by_color(args.input, args.output,
                         planes_csv_path=args.planes_csv,
                         merge_angle=args.merge_angle,
                         merge_dist=args.merge_dist,
                         cluster_eps=args.cluster_eps)


if __name__ == "__main__":
    main()
