// #include<time.h>
#include <chrono>
#include <stdio.h>
#include "host.hpp"
#define BLOCK_MATRIX_SIZE 16
#define DEBUG 0

using namespace std;

torch::Tensor forward_hw(torch::Tensor input, torch::Tensor weights, char* fileLoc);
torch::Tensor forward_sw(torch::Tensor input, torch::Tensor weights);

torch::Tensor forward_hw(torch::Tensor input, torch::Tensor weights, char* fileLoc){
    int stride =1;
    int pad =0;
    
    if(1){
            std::cout << "Turning tensors to arrays..." << std::endl;
    }
    // Turning tensors to arrays
    int batches, in_channels, in_height, in_width;
    float * in_array = tensor2arr_4d(input, &batches, &in_channels, &in_height, &in_width);

    int out_channels, kernel_in_channels, kernel_height, kernel_width;
    float * kernel_array = tensor2arr_4d(weights, &out_channels, &kernel_in_channels, &kernel_height, &kernel_width);

    if(1){
            std::cout << "Converting Images to col..." << std::endl;
    }
    // Image to Col Functions
    int in_array_col_height, in_array_col_width;
    int out_height, out_width;
    float * in_array_col = img2col(in_array, batches, in_channels, in_height, in_width, 
            kernel_height, kernel_width, stride, pad, 
            &in_array_col_height, &in_array_col_width, &out_height, &out_width);

    if(1){
            std::cout << "Converting Weights to col..." << std::endl;
    }
    int kernel_array_col_height, kernel_array_col_width;
    float * kernel_array_col = weight2col(kernel_array, out_channels, kernel_in_channels, kernel_height, kernel_width,
                    &kernel_array_col_height, &kernel_array_col_width);
    
//     float output_col[kernel_array_col_height * in_array_col_width] = { 0 };
std::vector<float> output_col(kernel_array_col_height * in_array_col_width, 0);

    if(1){
            std::cout << "Setting up binary file..." << std::endl;
    }
    // Binary files and Devices
    std::string binaryFile = fileLoc;
    unsigned fileBufSize;

    std::vector<cl::Device> devices = get_devices("Xilinx");
    devices.resize(1);
    cl::Device device = devices[0];

    char* fileBuf = read_binary_file(binaryFile, fileBufSize);
    cl::Program::Binaries bins{{fileBuf, fileBufSize}};

    cl_int err;  /*error variable*/
    if(DEBUG){
        std::cout<< "in_array_col: " << in_array_col_height << ", "<< in_array_col_width <<std::endl;
        print_tensor(in_array_col, 1, 1, in_array_col_height, in_array_col_width);
        std::cout<< "kernel_array_col: " << kernel_array_col_height << ", "<< kernel_array_col_width <<std::endl;
        print_tensor(kernel_array_col, 1, 1, kernel_array_col_height, kernel_array_col_width);
    }
    // ------------------------------------------------------------------------------------------------------
    // START KERNEL CODE
    // ------------------------------------------------------------------------------------------------------

    // Byte size initialization
    size_t blk_1_bytes;
    size_t blk_2_bytes;
    size_t blk_o_bytes;

    // Setting up queues
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue q(context, device, cl::QueueProperties::Profiling | cl::QueueProperties::OutOfOrder, &err));
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    int mat_h_1 = kernel_array_col_height;
    int mat_w_1 = kernel_array_col_width;
    int mat_h_2 = in_array_col_height;
    int mat_w_2 = in_array_col_width;

    int i_int_max = int(mat_w_2/ BLOCK_MATRIX_SIZE) + 1;
    int j_int_max = int(mat_h_1/ BLOCK_MATRIX_SIZE) + 1;
        
    std::vector<float, aligned_allocator<float>> blk_1(BLOCK_MATRIX_SIZE * mat_w_1);
    std::vector<float, aligned_allocator<float>> blk_2(mat_h_2 * BLOCK_MATRIX_SIZE);
//     std::vector<float, aligned_allocator<float>> blk_o(BLOCK_MATRIX_SIZE * BLOCK_MATRIX_SIZE);
    std::vector<std::vector<std::vector<float, aligned_allocator<float>>>> blk_o_list(
            j_int_max, std::vector<std::vector<float, aligned_allocator<float>>>(
                    i_int_max, std::vector<float, aligned_allocator<float>>(
                            BLOCK_MATRIX_SIZE * BLOCK_MATRIX_SIZE
                    )
            )
    );

    //     Buffer List
    std::vector<cl::Buffer> buffer_in1_list(j_int_max);
    std::vector<cl::Buffer> buffer_in2_list(i_int_max);
    std::vector<cl::Buffer> buffer_output_list(i_int_max * j_int_max);


    std::vector<std::vector<std::vector<cl::Event>>> iteration_events(
            j_int_max, std::vector<std::vector<cl::Event>>(
                    i_int_max
            )
    );
    

        int j_int = -1;
        for(int b_j_1=0; b_j_1 < mat_h_1; b_j_1+=BLOCK_MATRIX_SIZE){
                j_int++;
                int i_int = -1;
                for(int b_i_2=0; b_i_2 < mat_w_2; b_i_2+=BLOCK_MATRIX_SIZE){
                        i_int++;
                        
                        int blk_h_1 = std::min(BLOCK_MATRIX_SIZE, mat_h_1-b_j_1);
                        int blk_w_1 = mat_w_1;
                        int blk_h_2 = blk_w_1;
                        int blk_w_2 = std::min(BLOCK_MATRIX_SIZE, mat_w_2-b_i_2);
                        int blk_h_o = blk_h_1;
                        int blk_w_o = blk_w_2;
                        
                        // Matrix buffer size allocation
                        blk_1_bytes = sizeof(float) * blk_h_1 * blk_w_1;
                        blk_2_bytes = sizeof(float) * blk_h_2 * blk_w_2;
                        blk_o_bytes = sizeof(float) * blk_h_o * blk_w_o;

                        for(int b_k_1=0; b_k_1 < mat_w_1; b_k_1+=BLOCK_MATRIX_SIZE){
                                for(int i=0; i<BLOCK_MATRIX_SIZE; i++){
                                        for(int j=0; j<BLOCK_MATRIX_SIZE; j++){

                                                // input block 1
                                                if(i<blk_w_1 && j<blk_h_1){
                                                        blk_1[(blk_w_1*j)+(i)] = kernel_array_col[(mat_w_1*(b_j_1+j))+(b_k_1+i)];
                                                }
                                                // input block 2
                                                if(i<blk_w_2 && j<blk_h_2){
                                                        blk_2[(blk_w_2*j)+(i)] = in_array_col[(mat_w_2*(b_k_1+j))+(b_i_2+i)];
                                                }
                                        }
                                }
                        }
                        
                        if(DEBUG){
                                std::cout<< "block 1 : " << blk_h_1 << ", "<< blk_w_1 << std::endl;
                                print_tensor(blk_1.data(), 1, 1, blk_h_1, blk_w_1);
                                std::cout<< "block 2 : " << blk_h_2 << ", "<< blk_w_2 <<std::endl;
                                print_tensor(blk_2.data(), 1, 1, blk_h_2, blk_w_2);
                        }

                        // Setting up  buffers
                        // OCL_CHECK(err, cl::Buffer buffer_in1(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                        //         blk_1_bytes, blk_1.data(), &err));
                        // OCL_CHECK(err, cl::Buffer buffer_in2(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                        //         blk_2_bytes, blk_2.data(), &err));
                        // OCL_CHECK(err, cl::Buffer buffer_output(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
                        //         blk_o_bytes, blk_o.data(), &err));
                        
                        buffer_in1_list[j_int] = cl::Buffer (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                blk_1_bytes, blk_1.data(), NULL);
                        buffer_in2_list[i_int] = cl::Buffer (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                blk_2_bytes, blk_2.data(), NULL);
                        buffer_output_list[(j_int*i_int_max)+(i_int)] = cl::Buffer(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
                                blk_o_bytes, blk_o_list[j_int][i_int].data(), NULL);

                        // Kernel setup
                        OCL_CHECK(err, cl::Kernel krnl_matmul(program,"vdot", &err));
                        OCL_CHECK(err, err = krnl_matmul.setArg(0, buffer_in1_list[j_int]));
                        OCL_CHECK(err, err = krnl_matmul.setArg(1, blk_h_1));
                        OCL_CHECK(err, err = krnl_matmul.setArg(2, blk_w_1));
                        OCL_CHECK(err, err = krnl_matmul.setArg(3, buffer_in2_list[i_int]));
                        OCL_CHECK(err, err = krnl_matmul.setArg(4, blk_h_2));
                        OCL_CHECK(err, err = krnl_matmul.setArg(5, blk_w_2));
                        OCL_CHECK(err, err = krnl_matmul.setArg(6, buffer_output_list[(j_int*i_int_max)+(i_int)]));

                        // Running
                        cl::Event write_event;
                        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1_list[j_int], buffer_in2_list[i_int]},0/* 0 means from host*/, NULL, &write_event));	
                        iteration_events[j_int][i_int].push_back(write_event);

                        cl::Event kernel_event;
                        OCL_CHECK(err, err = q.enqueueTask(krnl_matmul, &iteration_events[j_int][i_int], &kernel_event));
                        iteration_events[j_int][i_int].push_back(kernel_event);

                        iteration_events[j_int][i_int].push_back(kernel_event);
                        cl::Event read_event;
                        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output_list[(j_int*i_int_max)+(i_int)]},CL_MIGRATE_MEM_OBJECT_HOST, &iteration_events[j_int][i_int], &read_event));
                        iteration_events[j_int][i_int].push_back(read_event);

                }
        }
        
        j_int = -1;
        for(int b_j_1=0; b_j_1 < mat_h_1; b_j_1+=BLOCK_MATRIX_SIZE){
                j_int++;
                int i_int = -1;
                for(int b_i_2=0; b_i_2 < mat_w_2; b_i_2+=BLOCK_MATRIX_SIZE){
                        i_int++;

                        int blk_h_1 = std::min(BLOCK_MATRIX_SIZE, mat_h_1-b_j_1);
                        int blk_w_1 = mat_w_1;
                        int blk_h_2 = blk_w_1;
                        int blk_w_2 = std::min(BLOCK_MATRIX_SIZE, mat_w_2-b_i_2);
                        int blk_h_o = blk_h_1;
                        int blk_w_o = blk_w_2;

                        // waiting for queue to finish
                        iteration_events[j_int][i_int].back().wait();
    
                        for(int i=0; i<BLOCK_MATRIX_SIZE; i++){
                                for(int j=0; j<BLOCK_MATRIX_SIZE; j++){
                                        // output block
                                        if(i<blk_w_o && j<blk_h_o){
                                                output_col[(mat_w_2 * (b_j_1+j))+(b_i_2+i)] += blk_o_list[j_int][i_int][(blk_w_o * (j))+(i)];
                                        }
                                }
                        }

                        if(DEBUG){
                                std::cout<< "block out : " << blk_h_o << ", "<< blk_w_o <<std::endl;
                                print_tensor(blk_o_list[j_int][i_int].data(), 1, 1, blk_h_o, blk_w_o);
                        }
                }
        }

     q.finish();


    // -----------------------------------------------------------------------------------------------------------------------------
    // END KERNEL CODE
    // -----------------------------------------------------------------------------------------------------------------------------
    if(DEBUG){
        std::cout<< "output_col" <<std::endl;
        print_tensor(output_col.data(), 1, 1, kernel_array_col_height, in_array_col_width);
    }
    

    // Col to Image Functions
    float * out_array = col2img(output_col.data(), batches, out_channels, out_height, out_width);

    // Converting array to tensor
    torch::Tensor output = arr2tensor_4d(out_array, batches, out_channels, out_height, out_width);

    // Delete intermediate arrays
    delete[] in_array_col, kernel_array_col;
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