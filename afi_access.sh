#!/bin/bash

TARGET=hw
NAME=vdot
PROJECT_DIR=FPGA_accelerated_CNN

BUILD_DIR=${PROJECT_DIR}/build
TEMP_BUILD_DIR=${PROJECT_DIR}/temp_build

cd ${TEMP_BUILD_DIR}

AFI_ID=$(grep FpgaImageId *_afi_id.txt | cut -f4 -d\")
USER_ID="835065147291"  #Janvi

aws ec2 --region us-east-1 modify-fpga-image-attribute --fpga-image-id ${AFI_ID} --operation-type add --user-ids ${USER_ID}