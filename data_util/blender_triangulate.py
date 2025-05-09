import bpy
import sys

def triangulate_object(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    # Select the object
    bpy.context.view_layer.objects.active = obj
    # Switch to object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    # Switch to edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    # Select all faces
    bpy.ops.mesh.select_all(action='SELECT')
    # Triangulate the mesh
    bpy.ops.mesh.quads_convert_to_tris()
    # Switch back to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

def main():
    input_path = sys.argv[-2]
    output_path = sys.argv[-1]

    bpy.context.preferences.system.audio_device = 'Null'

    # Clear existing scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import the OBJ file
    bpy.ops.import_scene.obj(filepath=input_path)

    # Triangulate all imported objects
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            triangulate_object(obj)

    # Export the triangulated mesh to OBJ
    bpy.ops.export_scene.obj(filepath=output_path, use_selection=False)

if __name__ == "__main__":
    main()
