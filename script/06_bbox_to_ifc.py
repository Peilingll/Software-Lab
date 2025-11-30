#!/usr/bin/env python3
"""
06_bbox_to_ifc_from_json.py
===========================
(JSON Input + Manhattan Alignment)

01_filter_structure.py (提供 floor/ceiling)
05_compute_bbox_corrected.py (提供 wall_bboxes.json)
"""

import ifcopenshell
import ifcopenshell.guid
import numpy as np
import json
import time
import argparse
import sys
import os
import math
import open3d as o3d
import glob
from pathlib import Path



def ensure_manhattan_rotation(R, tolerance_deg=10.0):

    basis_vectors = R.T
    aligned_R = np.zeros_like(R)
    
    for i in range(3):
        direction = basis_vectors[i]
        abs_dir = np.abs(direction)
        dominant_axis = np.argmax(abs_dir)
        angle_from_axis_rad = np.arccos(abs_dir[dominant_axis])
        
        if np.degrees(angle_from_axis_rad) > tolerance_deg:
            return None 
        
        aligned_dir = np.zeros(3)
        aligned_dir[dominant_axis] = np.sign(direction[dominant_axis])
        aligned_R[:, i] = aligned_dir
    
    try:
        U, _, Vt = np.linalg.svd(aligned_R)
        aligned_R = U @ Vt
    except np.linalg.LinAlgError:
        return None 
    
    if np.linalg.det(aligned_R) < 0:
        aligned_R[:, -1] *= -1 
        
    return aligned_R

def create_ifc_project():
    """ IFC project structure"""
    ifc = ifcopenshell.file(schema="IFC4")
    
    person = ifc.create_entity("IfcPerson")
    org = ifc.create_entity("IfcOrganization", Name="PointCloud2IFC")
    user = ifc.create_entity("IfcPersonAndOrganization", 
                             ThePerson=person, 
                             TheOrganization=org)
    app = ifc.create_entity("IfcApplication", 
                            ApplicationDeveloper=org, 
                            Version="7.0-JSON-Manhattan",
                            ApplicationFullName="BBox2IFC-JSON", 
                            ApplicationIdentifier="B2I-S7")
    owner_history = ifc.create_entity("IfcOwnerHistory",
                                      OwningUser=user,
                                      OwningApplication=app,
                                      ChangeAction="ADDED",
                                      CreationDate=int(time.time()))
    
    length_unit = ifc.create_entity("IfcSIUnit", 
                                    UnitType="LENGTHUNIT", 
                                    Name="METRE")
    units = ifc.create_entity("IfcUnitAssignment", 
                              Units=[length_unit])

    origin = ifc.create_entity("IfcCartesianPoint", Coordinates=[0.0, 0.0, 0.0])
    x_axis = ifc.create_entity("IfcDirection", DirectionRatios=[1.0, 0.0, 0.0])
    z_axis = ifc.create_entity("IfcDirection", DirectionRatios=[0.0, 0.0, 1.0])
    placement = ifc.create_entity("IfcAxis2Placement3D", 
                                  Location=origin, 
                                  Axis=z_axis, 
                                  RefDirection=x_axis)
    context = ifc.create_entity("IfcGeometricRepresentationContext",
                                ContextIdentifier="Model", 
                                ContextType="Model",
                                CoordinateSpaceDimension=3, 
                                Precision=1e-5,
                                WorldCoordinateSystem=placement)

    body_context = ifc.create_entity("IfcGeometricRepresentationSubContext",
                                     ContextIdentifier="Body",
                                     ContextType="Model",
                                     ParentContext=context,
                                     TargetView="MODEL_VIEW")

    project = ifc.create_entity("IfcProject", 
                                GlobalId=ifcopenshell.guid.new(),
                                OwnerHistory=owner_history, 
                                Name="Point Cloud Building (JSON-Manhattan)",
                                RepresentationContexts=[context],
                                UnitsInContext=units)

    site = ifc.create_entity("IfcSite", 
                             GlobalId=ifcopenshell.guid.new(),
                             OwnerHistory=owner_history, 
                             Name="Site")
    building = ifc.create_entity("IfcBuilding", 
                                 GlobalId=ifcopenshell.guid.new(),
                                 OwnerHistory=owner_history, 
                                 Name="Building")
    storey = ifc.create_entity("IfcBuildingStorey", 
                               GlobalId=ifcopenshell.guid.new(),
                               OwnerHistory=owner_history, 
                               Name="Ground Floor",
                               Elevation=0.0)

    ifc.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), 
                      OwnerHistory=owner_history, RelatingObject=project, 
                      RelatedObjects=[site])
    ifc.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), 
                      OwnerHistory=owner_history, RelatingObject=site, 
                      RelatedObjects=[building])
    ifc.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), 
                      OwnerHistory=owner_history, RelatingObject=building, 
                      RelatedObjects=[storey])

    return ifc, storey, body_context, owner_history

def create_box_geometry_aabb(ifc, body_context, min_bound, max_bound):
    """(AABB )"""
    dx = abs(max_bound[0] - min_bound[0])
    dy = abs(max_bound[1] - min_bound[1])
    dz = abs(max_bound[2] - min_bound[2])
    
    if dx < 0.01 or dy < 0.01 or dz < 0.01:
        return None

    profile_origin = ifc.create_entity("IfcCartesianPoint", Coordinates=[0.0, 0.0])
    profile_axis = ifc.create_entity("IfcAxis2Placement2D", Location=profile_origin)
    
    profile = ifc.create_entity("IfcRectangleProfileDef",
                                ProfileType="AREA",
                                Position=profile_axis,
                                XDim=float(dx),
                                YDim=float(dy))

    center_offset_x = dx / 2.0
    center_offset_y = dy / 2.0

    base_point = ifc.create_entity("IfcCartesianPoint",
        Coordinates=[float(min_bound[0] + center_offset_x), 
                    float(min_bound[1] + center_offset_y), 
                    float(min_bound[2])])

    z_dir = ifc.create_entity("IfcDirection", DirectionRatios=[0.0, 0.0, 1.0])
    x_dir = ifc.create_entity("IfcDirection", DirectionRatios=[1.0, 0.0, 0.0])
    axis3d = ifc.create_entity("IfcAxis2Placement3D",
                            Location=base_point,
                            Axis=z_dir,
                            RefDirection=x_dir)

    extrude_dir = ifc.create_entity("IfcDirection", DirectionRatios=[0.0, 0.0, 1.0])
    solid = ifc.create_entity("IfcExtrudedAreaSolid",
                              SweptArea=profile,
                              Position=axis3d,
                              ExtrudedDirection=extrude_dir,
                              Depth=float(dz))

    body_rep = ifc.create_entity("IfcShapeRepresentation",
                                 ContextOfItems=body_context,
                                 RepresentationIdentifier="Body",
                                 RepresentationType="SweptSolid",
                                 Items=[solid])

    shape = ifc.create_entity("IfcProductDefinitionShape", 
                              Representations=[body_rep])
    
    return shape

def create_box_geometry_obb(ifc, body_context, center, extent, rotation_matrix, manhattan_tolerance=10.0):
    """
    OBB + Manhattan 
    
    """
    R = np.array(rotation_matrix)
    
    
    aligned_R = ensure_manhattan_rotation(R, tolerance_deg=manhattan_tolerance)
    
    if aligned_R is None:
        print("    Warning: non-Manhattan，use AABB ")
        
        corners = o3d.geometry.OrientedBoundingBox(center, R, extent).get_box_points()
        min_bound = np.min(corners, axis=0)
        max_bound = np.max(corners, axis=0)
        return create_box_geometry_aabb(ifc, body_context, min_bound, max_bound)

    local_x_axis = aligned_R[:, 0]
    local_y_axis = aligned_R[:, 1]
    local_z_axis = aligned_R[:, 2]

    z_dot_products = np.abs([np.dot(local_x_axis, [0,0,1]),
                              np.dot(local_y_axis, [0,0,1]),
                              np.dot(local_z_axis, [0,0,1])])
    
    height_idx = np.argmax(z_dot_products)
    other_indices = [i for i in range(3) if i != height_idx]
    
    if extent[other_indices[0]] < extent[other_indices[1]]:
        thickness_idx = other_indices[0]
        length_idx = other_indices[1]
    else:
        thickness_idx = other_indices[1]
        length_idx = other_indices[0]
        
    length = extent[length_idx]
    thickness = extent[thickness_idx]
    height = extent[height_idx]

    if length < 0.01 or height < 0.01 or thickness < 0.01:
        return None
    
    profile_origin = ifc.create_entity("IfcCartesianPoint", Coordinates=[0.0, 0.0])
    profile_axis = ifc.create_entity("IfcAxis2Placement2D", Location=profile_origin)
    
    profile = ifc.create_entity("IfcRectangleProfileDef",
                                ProfileType="AREA",
                                Position=profile_axis,
                                XDim=float(length),
                                YDim=float(thickness)) 
    
    location = ifc.create_entity("IfcCartesianPoint", 
                                 Coordinates=[float(c) for c in center])
    
    axis_dir = ifc.create_entity("IfcDirection", 
                                 DirectionRatios=[float(c) for c in aligned_R[:, height_idx]])
    
    ref_dir = ifc.create_entity("IfcDirection", 
                                DirectionRatios=[float(c) for c in aligned_R[:, length_idx]])
    
    placement = ifc.create_entity("IfcAxis2Placement3D",
                                 Location=location,
                                 Axis=axis_dir,
                                 RefDirection=ref_dir)
    
    extrude_dir = ifc.create_entity("IfcDirection", DirectionRatios=[0.0, 0.0, 1.0])
    
    adjusted_center = np.array(center) - (height / 2.0) * aligned_R[:, height_idx]
    adjusted_location_point = ifc.create_entity("IfcCartesianPoint", 
                                                Coordinates=[float(c) for c in adjusted_center])
    
    placement.Location = adjusted_location_point
    
    solid = ifc.create_entity("IfcExtrudedAreaSolid",
                              SweptArea=profile,
                              Position=placement,
                              ExtrudedDirection=extrude_dir,
                              Depth=float(height))
    
    body_rep = ifc.create_entity("IfcShapeRepresentation",
                                 ContextOfItems=body_context,
                                 RepresentationIdentifier="Body",
                                 RepresentationType="SweptSolid",
                                 Items=[solid])
    
    shape = ifc.create_entity("IfcProductDefinitionShape", 
                              Representations=[body_rep])
    
    return shape

def add_floor_ceiling_from_ply(ifc, storey, body_context, owner_history, 
                               floor_ply=None, ceiling_ply=None):
    """
    Add floor and ceiling slabs from point cloud .ply files.
    """
    
    if floor_ply and os.path.exists(floor_ply):
        print(f"\n process floor and ceiling: {floor_ply}")
        try:
            pcd = o3d.io.read_point_cloud(floor_ply)
            if not pcd.has_points():
                print("  empty file。")
                return

            z_min = pcd.get_min_bound()[2]
            z_max = pcd.get_max_bound()[2]
            z_mid = (z_min + z_max) / 2.0
            points = np.asarray(pcd.points)
            floor_points = points[points[:, 2] < z_mid]
            ceil_points = points[points[:, 2] >= z_mid]
            
            if len(floor_points) > 100:
                floor_pcd = o3d.geometry.PointCloud()
                floor_pcd.points = o3d.utility.Vector3dVector(floor_points)
                aabb = floor_pcd.get_axis_aligned_bounding_box()
                min_b = aabb.min_bound
                max_b = aabb.max_bound
                avg_z = (min_b[2] + max_b[2]) / 2.0
                min_b[2] = avg_z - 0.05
                max_b[2] = avg_z + 0.05
                
                shape = create_box_geometry_aabb(ifc, body_context, min_b, max_b)
                if shape:
                    slab = ifc.create_entity("IfcSlab",
                                           GlobalId=ifcopenshell.guid.new(),
                                           OwnerHistory=owner_history,
                                           Name="Floor_Slab",
                                           PredefinedType="FLOOR")
                    slab.Representation = shape
                    ifc.create_entity("IfcRelContainedInSpatialStructure",
                                     GlobalId=ifcopenshell.guid.new(),
                                     OwnerHistory=owner_history,
                                     RelatingStructure=storey,
                                     RelatedElements=[slab])
                    print(f"  floor add: {len(floor_points):,} points")

            if len(ceil_points) > 100:
                ceil_pcd = o3d.geometry.PointCloud()
                ceil_pcd.points = o3d.utility.Vector3dVector(ceil_points)
                aabb = ceil_pcd.get_axis_aligned_bounding_box()
                min_b = aabb.min_bound
                max_b = aabb.max_bound
                avg_z = (min_b[2] + max_b[2]) / 2.0
                min_b[2] = avg_z - 0.05
                max_b[2] = avg_z + 0.05
                
                shape = create_box_geometry_aabb(ifc, body_context, min_b, max_b)
                if shape:
                    covering = ifc.create_entity("IfcCovering",
                                                GlobalId=ifcopenshell.guid.new(),
                                                OwnerHistory=owner_history,
                                                Name="Ceiling",
                                                PredefinedType="CEILING")
                    covering.Representation = shape
                    ifc.create_entity("IfcRelContainedInSpatialStructure",
                                     GlobalId=ifcopenshell.guid.new(),
                                     OwnerHistory=owner_history,
                                     RelatingStructure=storey,
                                     RelatedElements=[covering])
                    print(f"  ceiling add: {len(ceil_points):,} points")

        except Exception as e:
            print(f"  floor/ceiling processing failed: {e}")

    if ceiling_ply and os.path.exists(ceiling_ply) and (floor_ply != ceiling_ply):
        print(f"\n process ceiling: {ceiling_ply}")
        pass

def add_walls_from_json(ifc, storey, body_context, owner_history, json_path, manhattan_tolerance=10.0):
    """
    05 script output JSON -> IFC walls with OBB + Manhattan alignment
    """
    print(f"\n process walls: {json_path}")

    with open(json_path, 'r') as f:
        all_bboxes = json.load(f)
        
    if not all_bboxes:
        print("Warning: JSON file empty no walls")
        return 0

    print(f"Found {len(all_bboxes)} walls. Applying Manhattan alignment (tolerance: {manhattan_tolerance}°)\n")

    wall_count = 0
    for wall_id, bbox_data in all_bboxes.items():
        try:
            
            center = bbox_data['center']
            extent = bbox_data['extent']
            rotation_matrix = bbox_data['rotation_matrix']
            
            
            shape = create_box_geometry_obb(ifc, body_context, 
                                           center,
                                           extent,
                                           rotation_matrix,
                                           manhattan_tolerance)
            
            dims = extent
            sorted_dims = sorted(dims)
            print(f"  Wall {wall_id: >2}: {sorted_dims[1]:.2f}m(L) × {sorted_dims[0]:.2f}m(T) × {sorted_dims[2]:.2f}m(H) "
                  f"({bbox_data.get('num_points', 'N/A'):,} points)")
            
            # 3. 創建 IFC 實體
            if shape:
                wall_entity = ifc.create_entity("IfcWall",
                                               GlobalId=ifcopenshell.guid.new(),
                                               OwnerHistory=owner_history,
                                               Name=f"Wall_{wall_id}",
                                               PredefinedType="SOLIDWALL")
                wall_entity.Representation = shape
                
                ifc.create_entity("IfcRelContainedInSpatialStructure",
                                 GlobalId=ifcopenshell.guid.new(),
                                 OwnerHistory=owner_history,
                                 RelatingStructure=storey,
                                 RelatedElements=[wall_entity])
                wall_count += 1
                
        except Exception as e:
            print(f"  Wall {wall_id}: creation failed - {e}")

    return wall_count


def main():
    parser = argparse.ArgumentParser(
        description="from 05 JSON output IFC (OBB + Manhattan alignment)")

    # *** Changed parameters ***
    parser.add_argument("--input-json", "-i", required=True,
                       help=" 05_compute_bbox_corrected.py create 'wall_bboxes_....json' file")
    parser.add_argument("--floor-ply", "-f",
                       help="floor/ceiling .ply file (e.g. 'output_clean/floor_ceiling.ply')")
    parser.add_argument("--ceiling-ply", "-c",
                       help="(optional) separate ceiling .ply file (if not included in floor-ply)")
    parser.add_argument("--output", "-o", default="building_obb_manhattan.ifc",
                       help="output .ifc file")
    parser.add_argument("--tolerance", type=float, default=10.0,
                       help="Manhattan alignment tolerance angle (degrees)")

    args = parser.parse_args()
    
    if not os.path.exists(args.input_json):
        print(f"Error: input JSON file not found: {args.input_json}")
        sys.exit(1)
    
    print("="*60)
    print("JSON -> IFC (OBB + Manhattan alignment)")
    print(f"  JSON input: {args.input_json}")
    print(f"  Floor: {args.floor_ply}")
    print(f"  Output: {args.output}")
    print("="*60)

    # 1. Create IFC structure
    ifc, storey, body_context, owner_history = create_ifc_project()

    # 2. Add floor and ceiling (unchanged)
    add_floor_ceiling_from_ply(ifc, storey, body_context, owner_history,
                              args.floor_ply, args.ceiling_ply)

    # 3. Add walls (new function)
    wall_count = add_walls_from_json(ifc, storey, body_context, owner_history,
                                     args.input_json, 
                                     manhattan_tolerance=args.tolerance)

    # 4. Save IFC
    ifc.write(args.output)
    
    print(f"\n{'='*60}")
    print(f"✓ IFC save {args.output}")
    print(f"  Total walls: {wall_count}")
    print(f"  JSON-Input + OBB+Manhattan")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()