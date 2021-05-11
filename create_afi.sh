#!/bin/bash
TARGET=hw
NAME=vdot
PROJECT_DIR=~/FPGA_accelerated_CNN

BUILD_DIR=${PROJECT_DIR}/build
TEMP_BUILD_DIR=${PROJECT_DIR}/temp_build

PENNID='73006462'
BUCKET_NAME=ese539.${PENNID}

# S3 AFI
XCLBIN_S3=s3://${BUCKET_NAME}/afi/${NAME}.${TARGET}.xclbin
TEMP_XCLBIN=${TEMP_BUILD_DIR}/${NAME}.${TARGET}.xclbin
AWSXCLBIN=${NAME}.${TARGET}

aws s3 cp ${XCLBIN_S3} ${TEMP_BUILD_DIR}

# cd ${TEMP_BUILD_DIR}
cd ${AWS_FPGA_REPO_DIR}  #/home/centos/aws-fpga

${VITIS_DIR}/tools/create_vitis_afi.sh -xclbin=${TEMP_XCLBIN} -o=${AWSXCLBIN} -s3_bucket=${BUCKET_NAME} -s3_dcp_key=dcp -s3_logs_key=logs


AFI_ID=$(grep FpgaImageId *_afi_id.txt | cut -f4 -d\")
STATUS=$(aws ec2 describe-fpga-images --fpga-image-ids $AFI_ID --query 'FpgaImages[*].[State.Code]' --output text)

# while [ ${STATUS} == "pending" ]; do
#     STATUS=$(aws ec2 describe-fpga-images --fpga-image-ids $AFI_ID --query 'FpgaImages[*].[State.Code]' --output text)
#     echo The status is ${STATUS}
#     sleep 60
# done

echo The status is ${STATUS}
echo Moving awsxclbin files

cp *awsxclbin ${TEMP_BUILD_DIR}
cp *_afi_id.txt ${TEMP_BUILD_DIR}

rm ${TEMP_XCLBIN}

# export TARGET=hw;export NAME=vdot;export PROJECT_DIR=~/FPGA_accelerated_CNN;export BUILD_DIR=${PROJECT_DIR}/build;export PENNID='73006462';export BUCKET_NAME=ese539.${PENNID};export XCLBIN_S3=s3://${BUCKET_NAME}/afi/${NAME}.${TARGET}.xclbin;export AWSXCLBIN=${BUILD_DIR}/${NAME}.${TARGET}.awsxclbin;

