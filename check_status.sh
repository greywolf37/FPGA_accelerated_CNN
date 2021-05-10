#!/bin/bash

TARGET=hw
NAME=vdot
PROJECT_DIR=./FPGA_accelerated_CNN

BUILD_DIR=${PROJECT_DIR}/build
TEMP_BUILD_DIR=${PROJECT_DIR}/temp_build

cd ${TEMP_BUILD_DIR}

AFI_ID=$(grep FpgaImageId *_afi_id.txt | cut -f4 -d\")
STATUS=$(aws ec2 describe-fpga-images --fpga-image-ids $AFI_ID --query 'FpgaImages[*].[State.Code]' --output text)



echo The Status is:
echo ${STATUS}