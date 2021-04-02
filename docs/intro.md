# Final Project: Accelerating VGG16 DCNN with an FPGA

The final project will lead you towards accelerating a DCNN called VGG16 with an FPGA. You will begin with implementing a General Matrix Multiply (GEMM) convolution kernel on an AWS F1 FPGA. Then, you will integrate the kernel into PyTorch using C++ extension. Finally, you will search for an efficient scheduling of your computation and benchmark your results.

The project will be split up by the natural partition of hardware and software. In software, you will complete the following:

* Create an FPGA-accelerated conv2d layer
* Create C++ OpenCL host code that can be binded with PyTorch
* Modify VGG code to use your FPGA-accelerated conv2d layer

In hardware, you will complete the following:

* Implement question 5 of lab 4 with IEEE 754 binary16 and provide support for matrices larger than your systolic array (start with $16 \times 16$)
* Implement an im2col $\rightarrow$ matrix multiply $\rightarrow$ col2im dataflow
* Verify your dataflow with PyTorch's conv2d results
* Use the V++ linker to create two compute units

By the end of the final project, you will have completed these overall goals:

* Created and verified a functional GEMM convolutional kernel on FPGA
* Implement two compute units implementing your kernel
* Optimize your GEMM kernel using techniques discussed in labs
* Use C++ Extension to communicate to your kernel with PyTorch API
* Implement software baseline and verify benchmarking with one compute unit
* Schedule computation and benchmark results with all two compute units
