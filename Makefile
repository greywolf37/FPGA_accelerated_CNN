#!/bin/bash
NAME := vdot
PROJECT_DIR := .
BUILD_DIR := ${PROJECT_DIR}/build
SOURCE_DIR := ${PROJECT_DIR}/src
CONFIG_DIR := ${PROJECT_DIR}/configs
LOG_DIR := ${PROJECT_DIR}/logs
REPORT_DIR := ${PROJECT_DIR}/reports
PENNID='73006462'
BUCKET_NAME=ese539.${PENNID}

VPP := v++
EMCONFIGUTIL := emconfigutil

# host sources
HOST_SRC := ${SOURCE_DIR}/host.cpp

# host bin
HOST := ${BUILD_DIR}/host

# kernel sources
KERNEL_SRC := ${SOURCE_DIR}/${NAME}.cpp

# Xilinx kernel objects
XOS := ${BUILD_DIR}/${NAME}.${TARGET}.xo

# Kernel bin
XCLBIN := ${BUILD_DIR}/${NAME}.${TARGET}.xclbin
EMCONFIG_FILE := ${BUILD_DIR}/emconfig.json

# S3 AFI
XCLBIN_S3 := s3://${BUCKET_NAME}/afi/${NAME}.${TARGET}.xclbin
AWSXCLBIN := ${BUILD_DIR}/${NAME}.${TARGET}.awsxclbin

# Host options
GCC_OPTS := -I${XILINX_XRT}/include/ -I${XILINX_VIVADO}/include/ -Wall -O0 -g -std=c++11 -L${XILINX_XRT}/lib/ -lOpenCL -lpthread -lrt -lstdc++

# VPP Linker options
VPP_INCLUDE_OPTS := -I ${PROJECT_DIR}/src 

# VPP common options
VPP_COMMON_OPTS := --config ${CONFIG_DIR}/design.cfg \
				   --log_dir ${LOG_DIR} \
				   --report_dir ${REPORT_DIR} \
				   --platform ${AWS_PLATFORM} \
				   --compile --kernel ${NAME}	

.PHONY: all
all: emulate

# error check target variable
ifndef TARGET
	$(error TARGET is not defined. It should be 'sw_emu' or 'hw_emu')
endif

${HOST}: ${HOST_SRC}

	g++ ${GCC_OPTS} -o $@ $+
	@echo 'Compiled Host Executable: ${HOST_EXE}'

${XOS}: ${KERNEL_SRC}
	@${RM} $@
	${VPP} -t ${TARGET} --config ${CONFIG_DIR}/design.cfg \
		--log_dir ${LOG_DIR} \
		--report_dir ${REPORT_DIR} \
		--platform ${AWS_PLATFORM} \
		--compile --kernel ${NAME} \
		-I ${SOURCE_DIR} -o $@ $+
	mv ${BUILD_DIR}/*compile_summary ${REPORT_DIR}/vdot.${TARGET}/

${XCLBIN}: ${XOS}
	${VPP} -t ${TARGET} --config ${CONFIG_DIR}/design.cfg \
		--log_dir ${LOG_DIR} \
		--report_dir ${REPORT_DIR} \
		--platform ${AWS_PLATFORM} \
		--link -o $@ $+
	mv ${BUILD_DIR}/*link_summary ${REPORT_DIR}/vdot.${TARGET}/

${EMCONFIG_FILE}:
	${EMCONFIGUTIL} --platform ${AWS_PLATFORM} --od ${BUILD_DIR}

emulate: ${HOST} ${XCLBIN} ${EMCONFIG_FILE}
	echo Running host code with kernel...
	XCL_EMULATION_MODE=${TARGET} ./${HOST} ${XCLBIN}
	echo Finished run
	mv profile_summary.csv ${PROJECT_DIR}/reports/vdot.${TARGET}/
	mv timeline_trace.csv ${PROJECT_DIR}/reports/vdot.${TARGET}/
	mv *.run_summary ${PROJECT_DIR}/reports/vdot.${TARGET}/

test: ${XCLBIN} ${EMCONFIG_FILE}
	echo Running host code with kernel...

	XCL_EMULATION_MODE=${TARGET} python3 ${PROJECT_DIR}/src/test.py ${XCLBIN}

	mv profile_summary.csv ${PROJECT_DIR}/reports/vdot.${TARGET}/
	mv timeline_trace.csv ${PROJECT_DIR}/reports/vdot.${TARGET}/
	mv *.run_summary ${PROJECT_DIR}/reports/vdot.${TARGET}/

hw_setup: ${XCLBIN} ${EMCONFIG_FILE}

	mv *.run_summary ${PROJECT_DIR}/reports/vdot.${TARGET}/

	aws s3 cp ${XCLBIN} s3://${BUCKET_NAME}/afi/

hw_afi:
	
	cd ${AWS_FPGA_REPO_DIR}                                         
	source vitis_setup.sh
	${VITIS_DIR}/tools/create_vitis_afi.sh -xclbin=${XCLBIN_S3} \
			-o=${AWSXCLBIN} -s3_bucket=${BUCKET_NAME} \
			-s3_dcp_key=dcp -s3_logs_key=logs

	

# {XILINX_XRT} = /opt/xilinx/xrt
# {XILINX_VIVADO} = /opt/Xilinx/Vivado/2020.1
# launch fpga_build
# start_job fpga_build -d FPGA_accelerated_CNN -c 'export TARGET=sw_emu; export LC_ALL="C"; source aws-fpga/vitis_setup.sh; cd FPGA_accelerated_CNN; make emulate' -s

# BRAM
# grep -A 14 "Utilization Estimates" vdot_csynth.rpt | grep "Total" | cut -d "|" -f3
# FF
# grep -A 14 "Utilization Estimates" vdot_csynth.rpt | grep "Total" | cut -d "|" -f5
# LUT
# grep -A 14 "Utilization Estimates" vdot_csynth.rpt | grep "Total" | cut -d "|" -f6

