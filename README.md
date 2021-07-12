# Project

Accelerating a DCNN (VGG16) with an FPGA by implementing GEMM convolutional Kernel on AWS F1 FPGA and integrate into Pytorch using C++ extension

This project describes an FPGA design that performs convolution operation within VGG16 architecture by using matrix multiplication.

The goal of the design is to optimize the throughput and time taken. The design of convolution on FPGA has few main parts such
as matrix multiplication, Hardware/Software kernel design, AFI Setup, VGG16 custom architecture design for bench-marking and
optimization. Few parts is designed and optimized to find the optimal balance among the throughput/ time taken and the accuracy.

According to the test results, the design with the optimal result used a loop unrolling, pipe-lining, systolic array, Block matrix
multiplication, out-of-order queues, and Compute units.
