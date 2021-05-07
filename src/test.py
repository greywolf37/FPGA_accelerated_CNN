import math
import torch
from torch import nn
from torch.utils.cpp_extension import load

XILINX_XRT="/opt/xilinx/xrt"
XILINX_VIVADO="/opt/Xilinx/Vivado/2020.1"

cnn = load(name="host", sources=["src/host.cpp"],
            extra_cflags=["-Wall", "-O0", "-g", "-std=c++14"],
            extra_ldflags=["-lOpenCL", "-lpthread", "-lrt", "-lstdc++"],
            extra_include_paths=["-I${XILINX_XRT}/include/", "-I${XILINX_VIVADO}/include/"
                                "-L${XILINX_XRT}/lib/"]            
            )

# cnn = load(name="host", sources=["host.cpp"],
#             extra_cflags="-I${XILINX_XRT}/include/ -I${XILINX_VIVADO}/include/\
#             -Wall -O0 -g -std=c++11\
#             -L${XILINX_XRT}/lib/ -lOpenCL -lpthread -lrt -lstdc++")
import host

print("begin")

input_tensor = torch.rand((1,1,3,3))
weights_tensor = torch.rand((1,1, 2,2))
output = cnn.forward_hw(input_tensor, weights_tensor, "build/vdot.sw_emu.xclbin")
# output = cnn.forward_sw(input_tensor, weights_tensor)

print(output)

# start_job fpga_build -d FPGA_accelerated_CNN -c 'source aws-fpga/vitis_setup.sh; export LC_ALL="C"; export CXX=g++; cd FPGA_accelerated_CNN/src; python3 test.py' -s
# start_job fpga_build -d FPGA_accelerated_CNN -c 'source aws-fpga/vitis_setup.sh; export LC_ALL="C"; export CXX=g++; cd FPGA_accelerated_CNN; export TARGET=sw_emu; make test' -s