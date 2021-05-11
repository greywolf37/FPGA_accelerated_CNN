#!/bin/bash
TARGET=hw
NAME=vdot
PROJECT_DIR=~/FPGA_accelerated_CNN

BUILD_DIR=${PROJECT_DIR}/build
TEMP_BUILD_DIR=${PROJECT_DIR}/temp_build

AWSXCLBIN=${TEMP_BUILD_DIR}/${NAME}.${TARGET}.awsxclbin

unset XCL_EMULATION_MODE

systemctl status mpd

source ~/aws-fpga/vitis_setup.sh

# source ~/aws-fpga/vitis_runtime_setup.sh
systemctl is-active --quiet mpd || sudo systemctl start mpd

STATUS=$(systemctl status mpd | grep Active | cut -f2 -d:|cut -f1 -d\()

while [ ${STATUS} == "inactive" ]; do
    sleep 5
    STATUS=$(systemctl status mpd | grep Active | cut -f2 -d:|cut -f1 -d\()
    echo The status is :${STATUS}
done

python3 ${PROJECT_DIR}/src/test.py ${AWSXCLBIN}