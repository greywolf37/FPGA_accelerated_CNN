#!/bin/bash

set -e
func=$1
prj_dir="/home/rubin3737/ml/hw5/project-greywolf37"
pytorch_dir="/home/rubin3737/libtorch/lib/"

torch_flags="-I/home/rubin3737/libtorch/include/torch/csrc/api/include/ -I/home/rubin3737/libtorch/include/ -L/home/rubin3737/libtorch/lib/ -lc10 -ltorch -D_GLIBCXX_USE_CXX11_ABI=0"
torch_flags="-I/home/rubin3737/libtorch/include -I/home/rubin3737/libtorch/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1 -std=gnu++11 -L/home/rubin3737/libtorch/lib/ -ltorch -lc10"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pytorch_dir}

pushd ${prj_dir}/func_tests/

g++ ${prj_dir}/func_tests/${func}_test.cpp -o ${func}_test
./${func}_test

popd


