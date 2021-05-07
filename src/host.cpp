// #include<time.h>
#include <chrono>
#include <stdio.h>
#include "host.hpp"
#define BLOCK_MATRIX_SIZE 16
#define DEBUG 1

using namespace std;

torch::Tensor forward_hw(torch::Tensor input, torch::Tensor weights, char* fileLoc);
torch::Tensor forward_sw(torch::Tensor input, torch::Tensor weights);

torch::Tensor forward_hw(torch::Tensor input, torch::Tensor weights, char* fileLoc){
    int stride =1;
    int pad =0;

    // Turning tensors to arrays
    int batches, in_channels, in_height, in_width;
    float * in_array = tensor2arr_4d(input, &batches, &in_channels, &in_height, &in_width);

    int out_channels, kernel_in_channels, kernel_height, kernel_width;
    float * kernel_array = tensor2arr_4d(weights, &out_channels, &kernel_in_channels, &kernel_height, &kernel_width);

    // Image to Col Functions
    int in_array_col_height, in_array_col_width;
    int out_height, out_width;
    float * in_array_col = img2col(in_array, batches, in_channels, in_height, in_width, 
            kernel_height, kernel_width, stride, pad, 
            &in_array_col_height, &in_array_col_width, &out_height, &out_width);

    int kernel_array_col_height, kernel_array_col_width;
    float * kernel_array_col = weight2col(kernel_array, out_channels, kernel_in_channels, kernel_height, kernel_width,
                    &kernel_array_col_height, &kernel_array_col_width);
    
    float * output_col = new float[kernel_array_col_height * in_array_col_width];

    // Binary files and Devices
    std::string binaryFile = fileLoc;
    unsigned fileBufSize;

    std::vector<cl::Device> devices = get_devices("Xilinx");
    devices.resize(1);
    cl::Device device = devices[0];

    char* fileBuf = read_binary_file(binaryFile, fileBufSize);
    cl::Program::Binaries bins{{fileBuf, fileBufSize}};

    cl_int err;  /*error variable*/
    std::cout<< "in_array_col" <<std::endl;
    print_tensor(in_array_col, 1, 1, in_array_col_height, in_array_col_width);
    std::cout<< "kernel_array_col" <<std::endl;
    print_tensor(kernel_array_col, 1, 1, kernel_array_col_height, kernel_array_col_width);
    // ------------------------------------------------------------------------------------------------------
    // START KERNEL CODE
    // ------------------------------------------------------------------------------------------------------

    // Matrix buffer size allocation
    size_t in_size_bytes = sizeof(float) * in_array_col_height * in_array_col_width;
    size_t kernel_size_bytes = sizeof(float) * kernel_array_col_height * kernel_array_col_width;
    size_t out_size_bytes = sizeof(float) * kernel_array_col_height * in_array_col_width;

    // Setting up queues
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE , &err));
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Setting up  buffers
    OCL_CHECK(err, cl::Buffer buffer_in1   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            kernel_size_bytes, kernel_array_col, &err));
    OCL_CHECK(err, cl::Buffer buffer_in2   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            in_size_bytes, in_array_col, &err));
    OCL_CHECK(err, cl::Buffer buffer_output(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
            out_size_bytes, output_col, &err));

    // Kernel setup
    OCL_CHECK(err, cl::Kernel krnl_matmul(program,"vdot", &err));
    OCL_CHECK(err, err = krnl_matmul.setArg(0, buffer_in1));
    OCL_CHECK(err, err = krnl_matmul.setArg(1, kernel_array_col_height));
    OCL_CHECK(err, err = krnl_matmul.setArg(2, kernel_array_col_width));
    OCL_CHECK(err, err = krnl_matmul.setArg(3, buffer_in2));
    OCL_CHECK(err, err = krnl_matmul.setArg(4, in_array_col_height));
    OCL_CHECK(err, err = krnl_matmul.setArg(5, in_array_col_width));
    OCL_CHECK(err, err = krnl_matmul.setArg(6, buffer_output));

    // Running
    std::vector<cl::Event> events;

    cl::Event write_event;
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1, buffer_in2},0/* 0 means from host*/, NULL, &write_event));	
    events.push_back(write_event);

    cl::Event kernel_event;
    OCL_CHECK(err, err = q.enqueueTask(krnl_matmul, NULL, &kernel_event));
    events.push_back(kernel_event);

    q.finish();

    cl::Event read_event;
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output},CL_MIGRATE_MEM_OBJECT_HOST, NULL, &read_event));

    q.finish();
    std::cout<< "output_col" <<std::endl;
    print_tensor(output_col, 1, 1, kernel_array_col_height, in_array_col_width);
    // -----------------------------------------------------------------------------------------------------------------------------
    // END KERNEL CODE
    // -----------------------------------------------------------------------------------------------------------------------------

    // Col to Image Functions
    float * out_array = col2img(output_col, batches, out_channels, out_height, out_width);

    // Converting array to tensor
    torch::Tensor output = arr2tensor_4d(out_array, batches, out_channels, out_height, out_width);

    // Delete intermediate arrays
    delete[] in_array_col, kernel_array_col, output_col;
    delete[] fileBuf;

    return output;
}

torch::Tensor forward_sw(torch::Tensor input, torch::Tensor weights){
    
    int stride =1;
    int pad =0;

    // Turning tensors to arrays
    int batches, in_channels, in_height, in_width;
    float * in_array = tensor2arr_4d(input, &batches, &in_channels, &in_height, &in_width);

    int out_channels, kernel_in_channels, kernel_height, kernel_width;
    float * kernel_array = tensor2arr_4d(weights, &out_channels, &kernel_in_channels, &kernel_height, &kernel_width);

    // Image to Col Functions
    int in_array_col_height, in_array_col_width;
    int out_height, out_width;
    float * in_array_col = img2col(in_array, batches, in_channels, in_height, in_width, 
            kernel_height, kernel_width, stride, pad, 
            &in_array_col_height, &in_array_col_width, &out_height, &out_width);

    int kernel_array_col_height, kernel_array_col_width;
    float * kernel_array_col = weight2col(kernel_array, out_channels, kernel_in_channels, kernel_height, kernel_width,
                    &kernel_array_col_height, &kernel_array_col_width);

    // Matrix Multiplication
    float * output_col = matmul_sw(kernel_array_col, kernel_array_col_height, kernel_array_col_width,
                                    in_array_col, in_array_col_height, in_array_col_width);

    // Col to Image Functions
    float * out_array = col2img(output_col, batches, out_channels, out_height, out_width);

    // Converting array to tensor
    torch::Tensor output = arr2tensor_4d(out_array, batches, out_channels, out_height, out_width);
    
    // Delete intermediate arrays
    delete[] in_array_col, kernel_array_col, output_col;

    return output;
}

std::vector<torch::Tensor> backward_sw(torch::Tensor output_grad,
            torch::Tensor input, torch::Tensor weights) {
    
    int stride = 1;
    int pad = 0;

    // Casting tensors into arrays
    // Weights array
    int out_channels, in_channels, kernel_height, kernel_width;
    float * weights_arr = tensor2arr_4d(weights, &out_channels, &in_channels, &kernel_height, &kernel_width);

    // Input array
    int in_batches, in_channels_input, in_height, in_width;
    float * input_arr = tensor2arr_4d(input, &in_batches, &in_channels_input, &in_height, &in_width);

    // Gradient of output
    int out_batches, out_channels_output, out_height, out_width;
    float * output_grad_arr = tensor2arr_4d(output_grad, &out_batches, &out_channels_output, &out_height, &out_width);

    // WEIGHT UPDATE
    // img2col for weight update (using input_arr)
    int weight_img2col_height, weight_img2col_width; /*Shape of output col*/
    int weight_grad_height, weight_grad_width; /*Shape of weight grad*/
    float * weight_update_img2col_arr = weight_update_img2col(input_arr, in_batches, in_channels_input, in_height, in_width, /*input array*/ 
            out_height, out_width, /*We treat the output_grad as kernel height and width*/
            stride, pad, /*pad and stride are identical to what is used in forward*/
            &weight_img2col_height, &weight_img2col_width, &weight_grad_height, &weight_grad_width);

    // weight2col for weight update (using output grad)
    int weight_weight2col_height, weight_weight2col_width; /*Shape of output col*/
    float * weight_update_weight2col_arr = weight_update_weight2col(output_grad_arr, out_batches, out_channels_output, out_height, out_width,
                    &weight_weight2col_height, &weight_weight2col_width);
    // Matrix multiplication
    std::cout<< weight_weight2col_width <<std::endl;
    std::cout<< weight_grad_height <<std::endl;
    float * weight_grad_col_arr = matmul_sw(weight_update_weight2col_arr, weight_weight2col_height, weight_weight2col_width,
                    weight_update_img2col_arr, weight_img2col_height, weight_img2col_width);
    int weight_grad_col_height = weight_weight2col_height;
    int weight_grad_col_width = weight_img2col_width;

    // col2img for weight update
    float * weight_grad_arr = weight_update_col2img(weight_grad_col_arr, out_channels_output, in_channels_input, weight_grad_height, weight_grad_width);

    // Converting to tensor
    torch::Tensor weight_grad = arr2tensor_4d(weight_grad_arr, out_channels_output, in_channels_input, weight_grad_height, weight_grad_width);

    // Printing
    std::cout<< "weight_update_img2col_arr" <<std::endl;
    print_tensor(weight_update_img2col_arr, 1, 1, weight_img2col_height, weight_img2col_width);
    std::cout<< "weight_update_weight2col_arr" <<std::endl;
    print_tensor(weight_update_weight2col_arr, 1, 1, weight_weight2col_height, weight_weight2col_width);
    std::cout<< "weight_grad_col" <<std::endl;
    print_tensor(weight_grad_col_arr, 1, 1, weight_grad_col_height, weight_grad_col_width);
    std::cout<< "weight_grad_col_arr" <<std::endl;
    print_tensor(weight_grad_arr, out_channels_output, in_channels_input, weight_grad_height, weight_grad_width);
    
    // Deleting intermediate arrays
    delete[] weight_update_img2col_arr, weight_update_weight2col_arr, weight_grad_col_arr;

    // INPUT GRAD
    // padding output grad
    int og_pad_batches, og_pad_out_channels, og_pad_height, og_pad_width;
    float * og_pad_arr = pad_array(output_grad_arr, out_batches, out_channels_output, out_height, out_width,
                                    in_height - out_height, in_width-out_width, /*padding required*/
                                    &og_pad_batches, &og_pad_out_channels, &og_pad_height, &og_pad_width);

    // Tranposing weights
    int w, x, y, z;
    float * weights_T_arr = transpose_weights(weights, &w, &x, &y, &z);

    // img2col for input grad (using output grad)
    int ig_i2c_height, ig_i2c_width;
    int input_grad_height, input_grad_width;
    float * ig_i2c_arr = input_grad_img2col(og_pad_arr, og_pad_batches, og_pad_out_channels, og_pad_height, og_pad_width, 
            kernel_height, kernel_width, stride, pad, 
            &ig_i2c_height, &ig_i2c_width, &input_grad_height, &input_grad_width);

    // weight2col for weight update (for input grad)
    int ig_w2c_height, ig_w2c_width; /*Shape of output col*/
    float * ig_w2c_arr = input_grad_weight2col(weights_T_arr, w, x, y, z,
                    &ig_w2c_height, &ig_w2c_width);

    // Matrix Multiplication
    float * input_grad_col_arr = matmul_sw(ig_w2c_arr, ig_w2c_height, ig_w2c_width, /*weighs*/
                                            ig_i2c_arr, ig_i2c_height, ig_i2c_width); /*output gradient*/
    int input_grad_col_height = ig_w2c_height;
    int input_grad_col_width = ig_i2c_width;

    // col2img for weight update
    float * input_grad_arr = input_grad_col2img(input_grad_col_arr, og_pad_batches, in_channels_input, input_grad_height, input_grad_width);

    // Converting to tensor
    torch::Tensor input_grad = arr2tensor_4d(input_grad_arr, og_pad_batches, in_channels_input, input_grad_height, input_grad_width);

    // Printing
    std::cout<< "weights_T_arr" <<std::endl;
    print_tensor(weights_T_arr, w, x, y, z);
    std::cout<< "og_pad_arr" <<std::endl;
    print_tensor(og_pad_arr, og_pad_batches, og_pad_out_channels, og_pad_height, og_pad_width);
    std::cout<< "ig_i2c_arr" <<std::endl;
    print_tensor(ig_i2c_arr, 1, 1, ig_i2c_height, ig_i2c_width);
    std::cout<< "ig_w2c_arr" <<std::endl;
    print_tensor(ig_w2c_arr, 1, 1, ig_w2c_height, ig_w2c_width);
    std::cout<< "input_grad_col_arr" <<std::endl;
    print_tensor(input_grad_col_arr, 1, 1, input_grad_col_height, input_grad_col_width);
    std::cout<< "input_grad_arr" <<std::endl;
    print_tensor(input_grad_arr, og_pad_batches, in_channels_input, input_grad_height, input_grad_width);

    // Deleting intermediate arrays
    delete[] og_pad_arr, weights_T_arr, ig_i2c_arr, ig_w2c_arr,
                input_grad_col_arr;


    // std::cout<< "Test1:" << weight_img2col_width<< std::endl;
    // std::cout<< "Test2:" << ig_i2c_width<< std::endl;
    //  std::cout<< "Test3:" << weight_img2col_width<< std::endl;
    //  std::cout<< "Test4:" << in_channels<< std::endl;
    std::cout << "------------*******************-----------------" << std::endl;

    return {input_grad, weight_grad};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_hw", &forward_hw, "forward_hw");
    m.def("forward_sw", &forward_sw, "forward_sw");

}