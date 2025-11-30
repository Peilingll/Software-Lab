#!/usr/bin/env python3
import os, sys, numpy as np
import open3d as o3d
import shutil  
try:
    import plyfile
except ImportError:
    plyfile = None


def load_from_npz(npz_path):
    """Load point cloud data from Sonata NPZ format."""
    dat = np.load(npz_path, allow_pickle=True)
    
    # Coordinates
    if "coord" in dat:
        xyz = dat["coord"]
    elif "xyz" in dat:
        xyz = dat["xyz"]
    elif "points" in dat:
        xyz = dat["points"][:, :3]
    else:
        raise KeyError(f"No coordinate data found. Available keys: {list(dat.keys())}")
    
    # Colors
    if "color" in dat:
        rgb = dat["color"]
    elif "rgb" in dat:
        rgb = dat["rgb"]
    elif "colors" in dat:
        rgb = dat["colors"]
    else:
        rgb = None

    # Labels
    if "predictions" in dat:
        labels = dat["predictions"]
    elif "labels" in dat:
        labels = dat["labels"]
    elif "pred_labels" in dat:
        labels = dat["pred_labels"]
    else:
        raise KeyError(f"No label data found. Available keys: {list(dat.keys())}")
    
    # Class names
    label_names = dat.get("class_names", dat.get("label_names", None))
    
    return xyz, rgb, labels, label_names


def load_from_ply(ply_path):
    """Load point cloud from PLY format with optional label field."""
    pcd = o3d.io.read_point_cloud(ply_path, remove_nan_points=True, remove_infinite_points=True)
    xyz = np.asarray(pcd.points)
    rgb = (np.asarray(pcd.colors)*255).astype(np.uint8) if len(pcd.colors) else None
    
    # Try to read per-point integer label if present
    labels = None
    if plyfile:
        try:
            ply = plyfile.PlyData.read(ply_path)
            el = ply['vertex']
            for key in ['label','class','classification','pred','semantic']:
                if key in el.data.dtype.names:
                    labels = el[key]
                    break
        except Exception:
            pass
    
    return xyz, rgb, labels, None


def save_ply(path, xyz, rgb=None):
    """Save point cloud to PLY format."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb/255.0, 0, 1))
    o3d.io.write_point_cloud(path, pcd, print_progress=False)


def find_structure_ids(label_names):
    """
    Find wall, floor, and ceiling IDs from label names.
    Returns dict with keys: 'wall', 'floor', 'ceiling' (values are indices).
    """
    idmap = {}
    
    if label_names is None:
        return idmap
    
    # Normalize names and find matches
    for i, name in enumerate(label_names):
        name_lower = str(name).lower()
        
        if "wall" in name_lower and "wall" not in idmap:
            idmap["wall"] = i
        if "floor" in name_lower and "floor" not in idmap:
            idmap["floor"] = i
        if "ceiling" in name_lower and "ceiling" not in idmap:
            idmap["ceiling"] = i
    
    return idmap


def main(in_path, out_dir):
    """Main pipeline: load, filter structure, save separated files."""
    os.makedirs(out_dir, exist_ok=True)
    
    # Load data
    if in_path.endswith(".npz"):
        xyz, rgb, labels, label_names = load_from_npz(in_path)
    else:
        xyz, rgb, labels, label_names = load_from_ply(in_path)

    if labels is None:
        raise SystemExit("No per-point labels found. Run Sonata segmentation first.")

    # Find structure IDs
    print(f"Available classes: {label_names}")
    print(f"Unique label IDs in data: {np.unique(labels).tolist()}")
    
    idmap = find_structure_ids(label_names)
    
    WALL_ID = idmap.get("wall")
    FLOOR_ID = idmap.get("floor")
    CEIL_ID = idmap.get("ceiling")
    
    # Validate we found wall (floor/ceiling optional for this part)
    if WALL_ID is None:
        print(f"\nDetected IDs: wall={WALL_ID}, floor={FLOOR_ID}, ceiling={CEIL_ID}")
        raise SystemExit("Could not detect wall. Check class_names.")
    
    print(f"Using: wall={WALL_ID}, floor={FLOOR_ID}, ceiling={CEIL_ID}")

    # --- (REFINED) Part 1: Copy the correct floor.ply from Sonata output ---
    
    sonata_output_dir = os.path.dirname(in_path)
    source_floor_ply = os.path.join(sonata_output_dir, "floor_ceiling.ply")
    dest_floor_ply = os.path.join(out_dir, "floor_ceiling.ply") 

    if os.path.exists(source_floor_ply):
        shutil.copy2(source_floor_ply, dest_floor_ply)
        print(f"  floor_ceiling.ply: Copied from {source_floor_ply} (Corrected)")
    else:
        print(f"  Warning: {source_floor_ply} not found. 'floor_ceiling.ply' was NOT created.")


    # --- Part 2: Filter and save remaining structure (walls, ceiling, combined) ---

    structure_ids = [WALL_ID]
    if CEIL_ID is not None:
        structure_ids.append(CEIL_ID)
    
    keep_mask = np.isin(labels, structure_ids)
    struct_xyz = xyz[keep_mask]
    struct_rgb = rgb[keep_mask] if rgb is not None else None
    struct_lbl = labels[keep_mask]

    try:
        floor_pcd = o3d.io.read_point_cloud(dest_floor_ply)
        floor_xyz = np.asarray(floor_pcd.points)
        floor_rgb = np.asarray(floor_pcd.colors) if floor_pcd.has_colors() else None
        
        combined_xyz = np.vstack((struct_xyz, floor_xyz))
        
        combined_rgb = None
        if struct_rgb is not None and floor_rgb is not None and struct_rgb.shape[1] == floor_rgb.shape[1]:
            combined_rgb = np.vstack((struct_rgb, floor_rgb))
        elif struct_rgb is not None:
    
            floor_rgb_dummy = np.tile([0.5, 0.5, 0.5], (len(floor_xyz), 1))
            combined_rgb = np.vstack((struct_rgb, floor_rgb_dummy))
        
        save_ply(os.path.join(out_dir, "structure_only.ply"), combined_xyz, combined_rgb)
        print(f"  structure_only.ply: {len(combined_xyz)} points (Combined)")
    
    except Exception as e:
        print(f"  Could not combine floor into structure_only.ply: {e}")
        save_ply(os.path.join(out_dir, "structure_only.ply"), struct_xyz, struct_rgb)
        print(f"  structure_only.ply: {len(struct_xyz)} points (Walls/Ceiling only)")


    # Save individual components (walls and ceiling)
    components = [("walls", WALL_ID)]
    if CEIL_ID is not None:
        components.append(("ceiling", CEIL_ID))
    
    for name, lid in components:
        mask = (struct_lbl == lid)
        pts = struct_xyz[mask]
        cols = struct_rgb[mask] if struct_rgb is not None else None
        save_ply(os.path.join(out_dir, f"{name}.ply"), pts, cols)
        print(f"  {name}.ply: {len(pts)} points (Filtered)")

    print(f"\nAll files saved to: {out_dir}/")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Extract wall, floor, ceiling from Sonata segmentation")
    ap.add_argument("input_path", help="Path to merged_classification.npz or labeled .ply")
    ap.add_argument("--out", default="output_clean", help="Output directory")
    args = ap.parse_args()
    main(args.input_path, args.out)