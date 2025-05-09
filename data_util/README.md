# Dataset Utility Support

## Point Cloud Generation

You can directly use our sampled point cloud in 2BY2 dataset, which contains 1024 points per object. But you can also generate your own point cloud. Same to [Breaking Bad](https://breaking-bad-dataset.github.io/), we use blue noise sampling to generate point cloud.

```shell
sudo apt update

sudo apt install build-essential cmake libglfw3-dev libgl1-mesa-dev libx11-dev libxi-dev libxrandr-dev libxinerama-dev libxcursor-dev libxxf86vm-dev libxext-dev libpthread-stubs0-dev libdl-dev

g++ -I ./eigen-3.4.0 -I ./libigl/include -I ./glad/include -I /usr/include \
    -L /usr/lib \
    -o generate_pc \
    generate_pc.cpp ./glad/src/glad.c \
    -ldl -lGL -lglfw -pthread

bash generate_pc.sh
```

After this step, you will generate `.csv` files containing point clouds.

If you would like different number of points, change `L150` of `generate_pc.cpp` to your target number.

## Mesh Preprocessing

We triangulate each mesh by blender and uniformly scale them before generate point cloud. This step takes `partA.obj` and `partB.obj` as input and outputs the uniform mesh `partA_new.obj` and `partB_new.obj`

```shell
bash preprocess_obj_blender.sh
```

## URDF(Unified Robot Description Format) Generation

If you would like to use our mesh in simulator, you can run the following script.

```python
python generate_urdf.py
```

When creating VHACD file for collision, you might need to modify the `simplify_quadric_decimation(0.8)` parameter for better collision effect.