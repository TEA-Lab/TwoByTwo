import os
import trimesh

input_obj = "insert_flower/1/partA_new.obj"
vhacd_obj = "insert_flower/1/vhacd_partA_new.obj"
urdf_file = "insert_flower/1/partA.urdf"

# Step 1: Run VHACD
mesh = trimesh.load(input_obj)
print(f"loaded mesh with {len(mesh.faces)} faces and {len(mesh.vertices)} vertices")

# simple = mesh.simplify_quadric_decimation(500)

simple = mesh.simplify_quadric_decimation(0.8)

simple.export(vhacd_obj)

print(f"VHACD complete. Output saved to: {vhacd_obj}")

# Step 2: Write URDF
print(f"Generating URDF: {urdf_file}")

urdf_content = f"""<?xml version="1.0"?>
<robot name="partA">
  <link name="part_link">
    <visual>
      <geometry>
        <mesh filename="{input_obj}" scale="1 1 1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="{vhacd_obj}" scale="1 1 1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
</robot>
"""

with open(urdf_file, "w") as f:
    f.write(urdf_content)

print(f"URDF file written: {urdf_file}")