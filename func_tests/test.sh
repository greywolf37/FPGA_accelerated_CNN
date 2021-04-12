#!/bin/bash

set -e
func=$1
prj_dir="/home/rubin3737/ml/hw5/project-greywolf37"

pushd ${prj_dir}/func_tests/

g++ ${prj_dir}/func_tests/${func}_test.cpp -o ${func}_test
./${func}_test

popd
