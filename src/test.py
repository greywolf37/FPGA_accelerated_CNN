import math
import torch
import time
from torch import nn
from torch.utils.cpp_extension import load
import sys

XILINX_XRT="/opt/xilinx/xrt"
XILINX_VIVADO="/opt/Xilinx/Vivado/2020.1"

# XCLBIN_LOC="build/vdot.sw_emu.xclbin"
XCLBIN_LOC=sys.argv[1]

cnn = load(name="host", sources=["src/host.cpp"],
            extra_cflags=["-Wall", "-O0", "-g", "-std=c++14"],
            extra_ldflags=["-lOpenCL", "-lpthread", "-lrt", "-lstdc++"],
            extra_include_paths=["-I${XILINX_XRT}/include/", "-I${XILINX_VIVADO}/include/"
                                "-L${XILINX_XRT}/lib/"]            
            )


import host

print("begin")

# output = cnn.forward_hw(input_tensor, weights_tensor, "build/vdot.sw_emu.xclbin")
# output = cnn.forward_sw(input_tensor, weights_tensor)

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size = 2, stride=1, bias = False)
    
    def get_kernel(self, layer):
        if layer == 1:
            return self.conv1.weight
    def forward(self, input_tensor):
        return self.conv1(input_tensor), self.conv1.weight

class CNN_cpp(torch.nn.Module):
    def __init__(self):
        super(CNN_cpp, self).__init__()
        self.inputs_size = 4

    def forward(self, input_tensor, kernel):
        return cnn.forward_sw(input_tensor, kernel)

class CNN_cpp_hw(torch.nn.Module):
    def __init__(self):
        super(CNN_cpp_hw, self).__init__()
        self.inputs_size = 4

    def forward(self, input_tensor, kernel, file):
        return cnn.forward_hw(input_tensor, kernel, file)



model = CNN()
input_tensor = torch.ones((50,3,25,25))
ouput_torch = model.forward(input_tensor)[0]
kernel = model.get_kernel(1)

cpp_start = time.time() 
model_cpp = CNN_cpp()
ouput_cpp = model_cpp.forward(input_tensor, kernel)
cpp_end = time.time() 

cpp_hw_start = time.time() 
model_cpp_hw = CNN_cpp_hw()
ouput_cpp_hw = model_cpp_hw.forward(input_tensor, kernel, XCLBIN_LOC)
cpp_hw_end = time.time() 

stars = '*' * 100
print(stars, "\n Weight Check ",  "\n", stars, "\n", torch.equal(model.forward(input_tensor)[1], kernel ))
print(stars, "\n Inputs",  "\n", stars, "\n", input_tensor )
print(stars, "\n Weights",  "\n", stars, "\n", kernel )
print(stars, "\n Ouput of Torch Forward", "\n", stars ,"\n", ouput_torch)
print(stars, "\n Output of cpp Forward", "\n", stars, "\n", ouput_cpp)
print(stars, "\n Output of cpp Forward Hardware", "\n", stars, "\n", ouput_cpp_hw)
print(stars, "\n Output Check ",  "\n",stars, "\n", torch.isclose(ouput_torch, ouput_cpp))
print(stars, "\n Output Check with hw", "\n", stars, "\n", torch.isclose(ouput_torch, ouput_cpp_hw))
print(stars, "\n Time cpp", cpp_end - cpp_start)
print(stars, "\n Time cpp_hw", cpp_hw_end - cpp_hw_start)

# start_job fpga_build -d FPGA_accelerated_CNN -c 'source aws-fpga/vitis_setup.sh; export LC_ALL="C"; export CXX=g++; cd FPGA_accelerated_CNN/src; python3 test.py' -s
# start_job fpga_build -d FPGA_accelerated_CNN -c 'source aws-fpga/vitis_setup.sh; export LC_ALL="C"; export CXX=g++; cd FPGA_accelerated_CNN; export TARGET=sw_emu; make test' -s