#!/bin/bash

set -e

prj_dir="/home/rubin3737/ml/hw5/project-greywolf37"

pushd ${prj_dir}/func_tests/

g++ ${prj_dir}/func_tests/img2col_test.cpp -o img2col_test
./img2col_test

popd
