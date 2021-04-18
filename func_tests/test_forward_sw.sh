#!/bin/bash

set -e

prj_dir="/home/rubin3737/ml/hw5/project-greywolf37"
pytorch_dir="/home/rubin3737/libtorch"

pushd ${prj_dir}/src/src_build

cmake -DCMAKE_PREFIX_PATH=${pytorch_dir} ..
cmake --build . --config Release
./host

popd