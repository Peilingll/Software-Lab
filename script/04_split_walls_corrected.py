# Save as: script/04_split_walls_corrected.py
import numpy as np
import open3d as o3d
import os
import sys
import argparse
import json

def split_walls_by_color(colored_ply, output_dir):
    """
    Split walls_ransac_colored.ply into individual wall_*.ply files
    based on unique RGB colors.
    
    Args:
        colored_ply: Path to colored PLY file from RANSAC
        output_dir: Output directory for individual wall files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    pcd = o3d.io.read_point_cloud(colored_ply)
    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors)
    
    if len(xyz) == 0:
        print("Warning: Input file is empty")
        return
    
    # Find unique colors (each color = one wall)
    rgb_int = (rgb * 255).round().astype(int)
    unique_colors = np.unique(rgb_int, axis=0)
    
    print(f"Found {len(unique_colors)} unique wall planes")
    
    wall_info = []
    
    for i, color in enumerate(unique_colors, start=1):
        mask = np.all(rgb_int == color, axis=1)
        wall_xyz = xyz[mask]
        wall_rgb = rgb[mask]
        
        if len(wall_xyz) < 100:  # Filter out very small planes
            print(f"  Skipping wall {i}: Too few points ({len(wall_xyz)} < 100)")
            continue
        
        wall_pcd = o3d.geometry.PointCloud()
        wall_pcd.points = o3d.utility.Vector3dVector(wall_xyz)
        wall_pcd.colors = o3d.utility.Vector3dVector(wall_rgb)
        
        out_path = os.path.join(output_dir, f"wall_{i}.ply")
        o3d.io.write_point_cloud(out_path, wall_pcd)
        
        print(f"  wall_{i}.ply: {len(wall_xyz)} points")
        
        wall_info.append({
            'id': i,
            'num_points': len(wall_xyz),
            'color': color.tolist()
        })
    
    info_file = os.path.join(output_dir, "wall_info.json")
    with open(info_file, 'w') as f:
        json.dump(wall_info, f, indent=2)
    
    print(f"\nWall info saved to: {info_file}")
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
    
    split_walls_by_color(args.input, args.output)


if __name__ == "__main__":
    main()