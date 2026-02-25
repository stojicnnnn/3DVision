# 3DVision — GPU-Accelerated Industrial Picking Pipeline

Real-time 3D vision pipeline for robotic bin-picking. Captures depth data, segments objects, aligns them against a reference model using RANSAC + ICP, and sends pick poses to a robotic arm.

Built with C++17, CUDA, OpenGL, and Eigen. Designed for sub-second cycle times on commodity NVIDIA GPUs.

## Architecture

```
RealSense Camera ──► Depth Preprocessing (CUDA) ──► Point Cloud Generation (CUDA)
                                                            │
SAM2 Segmentation ◄── RGB Frame                            │
       │                                                    │
       ▼                                                    ▼
  Instance Masks ──► Masked Depth ──► Per-Object Point Cloud
                                            │
                                            ▼
                                  FPFH Feature Extraction
                                            │
                                            ▼
                              RANSAC Global Registration
                                            │
                                            ▼
                                   ICP Refinement (CUDA)
                                            │
                                            ▼
                                 6-DoF Pick Pose ──► xArm Robot
```

Processing is parallelized across detected instances using a thread pool. The OpenGL viewer renders point clouds, coordinate frames, and planned pick paths in real time.

## Quick Start (Demo Mode)

No camera or robot required. The pipeline generates a procedural test scene and runs the full registration stack on it.

```bash
# install dependencies (Ubuntu/Debian)
sudo apt install -y build-essential cmake libopencv-dev libeigen3-dev \
                    libglfw3-dev libglew-dev libglm-dev libyaml-cpp-dev \
                    libfmt-dev libspdlog-dev librealsense2-dev

# build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# run demo (no hardware needed)
./industry_picking ../config/pipeline_config.yaml
```

The default config ships with `use_camera: false` and `use_robot: false`, so this will:

1. Generate a synthetic RGB + depth scene with a box-shaped object
2. Create a dummy segmentation mask
3. Build point clouds and run FPFH + RANSAC + ICP registration
4. Output computed pick poses to stdout
5. Open the OpenGL viewer (close the window to exit)

To run headless, set `visualization: "none"` in the config.

## Production Setup

1. Connect an Intel RealSense depth camera and (optionally) an xArm robot.
2. Edit `config/pipeline_config.yaml`:
   ```yaml
   use_camera: true
   use_robot: true
   robot:
     ip: "192.168.1.184"
   segmentation:
     sam_server_url: "http://<host>:8090/sam2"
   reference_model_path: "/path/to/reference.ply"
   ```
3. Run:
   ```bash
   ./industry_picking ../config/pipeline_config.yaml
   ```

## Project Structure

```
├── cuda/                  CUDA kernels (depth preprocessing, point cloud, ICP)
├── include/               Headers
├── src/
│   ├── main.cpp           Entry point + config loader
│   ├── pipeline.cpp       Pipeline orchestrator
│   ├── registration.cpp   Voxel downsampling, normals, FPFH, RANSAC, ICP (CPU)
│   ├── camera.cpp         RealSense capture + intrinsics
│   ├── segmentation.cpp   SAM2 integration / mask loading
│   ├── robot.cpp          xArm motion planning + pick sequence
│   ├── gl_viewer.cpp      OpenGL point cloud / pose / path renderer
│   └── gpu_impl.cpp       CUDA dispatch layer
├── shaders/               GLSL shaders for the viewer
├── config/                YAML pipeline configuration
└── scripts/               Build and test helpers
```

## Dependencies

- **NVIDIA CUDA Toolkit** (optional — falls back to CPU)
- OpenCV, Eigen3, GLFW3, GLEW, GLM, yaml-cpp, fmt, spdlog
- Intel RealSense SDK (`librealsense2`) — only needed with `use_camera: true`

## Controls (OpenGL Viewer)

| Input         | Action |
|---------------|--------|
| Left Click    | Rotate |
| Middle Click  | Pan    |
| Scroll        | Zoom   |
