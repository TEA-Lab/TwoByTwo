# This script takes a raw .glb file, adds vertex colors or throw away if without color
# It then translates by its center of mass and normalize the bounding box
# Finally, it saves the mesh or point cloud to a specified folder 
import open3d as o3d
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import json
import trimesh
import sys
# import bpy
import subprocess
from tqdm import tqdm

global_scale = 3

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

"""
partA.obj -> partA_tri.obj (triangulate
partA_tri.obj -> partA_new.obj (recenter and rescale)
normal_vector.json
"""

# scale_factor = 2
def file_exists(filepath):
    return os.path.exists(filepath)

# activate when need to triangulate the meshes

def triangulate_mesh_with_pyblender(instance_dir):
    blender_script_path = os.path.join(current_dir, 'blender_triangulate.py')
    subprocess.run(['blender', '--factory-startup', '--background', '--python', blender_script_path, '--', os.path.join(current_dir, instance_dir, 'partA.obj'), os.path.join(current_dir, instance_dir, 'partA_tri.obj')])
    subprocess.run(['blender','--factory-startup', '--background', '--python', blender_script_path, '--', os.path.join(current_dir, instance_dir, 'partB.obj'), os.path.join(current_dir, instance_dir, 'partB_tri.obj')])

    

# use pyblender change the non-triangle mesh to triangle mesh
def combine_meshes(instance_dir):
    triangulate_mesh_with_pyblender(instance_dir)

    # Load the two meshes
    mesh1_path = os.path.join(current_dir, instance_dir, "partA_tri.obj")
    mesh2_path = os.path.join(current_dir, instance_dir, "partB_tri.obj")
    combined_mesh_path = os.path.join(current_dir, instance_dir, "combined_mesh.obj")

    if not file_exists(mesh1_path) or not file_exists(mesh2_path):
        sys.exit(1)


    mesh1 = o3d.io.read_triangle_mesh(mesh1_path)
    # mesh1.triangulate()
    mesh2 = o3d.io.read_triangle_mesh(mesh2_path)
    # mesh2.triangulate()

    if not mesh1.has_vertices():
        raise ValueError("mesh1 does not contain any vertices.", mesh1_path)
    if not mesh2.has_vertices():
       raise ValueError("mesh2 does not contain any vertices.", mesh2_path)

    # Combine vertices and triangles
    combined_vertices = np.vstack((np.asarray(mesh1.vertices), np.asarray(mesh2.vertices)))
    combined_triangles = np.vstack((np.asarray(mesh1.triangles), np.asarray(mesh2.triangles) + len(mesh1.vertices)))

    # Create a new mesh with the combined vertices and triangles
    combined_mesh = o3d.geometry.TriangleMesh()
    combined_mesh.vertices = o3d.utility.Vector3dVector(combined_vertices)
    combined_mesh.triangles = o3d.utility.Vector3iVector(combined_triangles)

    # Optionally combine vertex colors if they exist
    if mesh1.has_vertex_colors() and mesh2.has_vertex_colors():
        combined_colors = np.vstack((np.asarray(mesh1.vertex_colors), np.asarray(mesh2.vertex_colors)))
        combined_mesh.vertex_colors = o3d.utility.Vector3dVector(combined_colors)

    # print("Combined mesh saved to {}".format(combined_mesh_path))
    return (mesh1, mesh2, combined_mesh)

def calculate_center_and_scale_two_seperate_part(instance_dir):

    # calculate the center after combined two meshes and remove them to the center
    (mesh1, mesh2, combined_mesh) = combine_meshes(instance_dir)
    global_center = combined_mesh.get_center()
   
    mesh1.translate(-global_center)
    mesh2.translate(-global_center)
    combined_mesh.translate(-global_center)

    bounding_box = combined_mesh.get_axis_aligned_bounding_box()
    max_extent = np.max(bounding_box.get_extent())
    max_extent_index = np.argmax(bounding_box.get_extent())
    normal_vector = np.zeros(3)
    normal_vector[max_extent_index] = 1

    # Write stats to JSON file
    stats = {
        "Normal Vector":{
            "x": normal_vector[0],
            "y": normal_vector[1],
            "z": normal_vector[2]
        }
    }

    with open(os.path.join(current_dir, instance_dir, "normal_vector.json"), "w+") as f:
        json.dump(stats, f, indent=4)


    scale_factor = global_scale / max_extent
  
    mesh1.scale(scale_factor, (0,0,0))
    mesh2.scale(scale_factor, (0,0,0))
    combined_mesh.scale(scale_factor, (0,0,0))
    
    """
    save the final files, after re-centered and re-scale, including writing normal_vector.json
    """

    o3d.io.write_triangle_mesh(os.path.join(current_dir, instance_dir, "partA_new.obj"), mesh1)
    o3d.io.write_triangle_mesh(os.path.join(current_dir, instance_dir, "partB_new.obj"), mesh2)
    o3d.io.write_triangle_mesh(os.path.join(current_dir, instance_dir, "combined_mesh.obj"), combined_mesh)

# Example usage
# calculate_center_and_scale_two_seperate_part("original-2.obj", "original-3.obj", "combined_mesh.obj")


if __name__ == '__main__':

    instance_dir = sys.argv[1]
    print("In the processing of {}".format(instance_dir))
    calculate_center_and_scale_two_seperate_part(instance_dir)



