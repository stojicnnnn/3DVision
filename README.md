# Industry Picking (C++ GPU)

High-performance industrial picking pipeline using CUDA, OpenGL, and xArm.

## Dependencies

```bash
sudo apt install -y build-essential cmake libopencv-dev libeigen3-dev \
                    libglfw3-dev libglew-dev libglm-dev libyaml-cpp-dev \
                    libfmt-dev libspdlog-dev librealsense2-dev
# Requires NVIDIA CUDA Toolkit (nvcc)
```

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Run

1. Connect RealSense camera and xArm robot.
2. Edit `config/pipeline_config.yaml`.
3. Run:
   ```bash
   ./industry_picking ../config/pipeline_config.yaml
   ```

## Controls
- Left Click: Rotate
- Middle Click: Pan
- Scroll: Zoom

// Edit
// Edit
// Edit
// Edit
// Edit
// Edit
