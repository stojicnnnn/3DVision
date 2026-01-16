#!/bin/bash
# Test the pipeline in dummy mode
set -e

if [ ! -d "build" ]; then mkdir build; fi
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

echo "Running Pipeline Test..."
./industry_picking ../config/pipeline_config.yaml
echo "Test Passed."

// Edit
