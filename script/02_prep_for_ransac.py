import os, sys, numpy as np, open3d as o3d

in_ply = sys.argv[1]          # output_clean/walls.ply
voxel = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02  # 2 cm
pcd = o3d.io.read_point_cloud(in_ply)
pcd = pcd.voxel_down_sample(voxel)
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
o3d.io.write_point_cloud(in_ply.replace(".ply","_prep.ply"), pcd)
print("Saved:", in_ply.replace(".ply","_prep.ply"))
