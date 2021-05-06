// #include<time.h>
#include <chrono>
#include <stdio.h>
#include "host.hpp"
#define BLOCK_MATRIX_SIZE 16
#define DEBUG 1

using namespace std;

torch::Tensor forward_sw(torch::Tensor input, torch::Tensor weights);

int main(int argc, char** argv)
{
    if (argc !=2)
    {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
		return EXIT_FAILURE;
	}

    int batches=1;
    int in_channels=1;
    int in_height=3;
    int in_width=3;

    int kernel_height=2;
    int kernel_width=2;

    int out_channels=1; 
    int fileLoc = argv[1]  
    char res = "Pass"

    float in_array[batches*in_channels*in_height*in_width];
    init_tensor(float *tensor, int batches, int channels, int height, int width);

    float kernel_array[out_channels*in_channels*kernel_height*kernel_width];
    init_tensor(kernel_array, out_channels, in_channels, kernel_height, kernel_width);

    torch::Tensor input = arr2tensor_4d(in_array, batches, in_channels, in_height, in_width);
    torch::Tensor weights = arr2tensor_4d(kernel_array, out_channels, in_channels, kernel_height, kernel_width);
    
    clock_t start_sw = clock();
    torch::Tensor output_sw = forward_sw(input, weights);
    clock_t end_sw = clock();
    
    clock_t start = clock();
    torch::Tensor output = forward(input, weights, fileLoc);
    clock_t end = clock();

    std::cout << "output_sw" << output_sw << std::endl;
    std::cout << "output" << output << std::endl;

    if (DEBUG){
        for(int b=0; b<output.size(0); b++){
            for(int c=0; c<output.size(1); c++){
                for(int h=0; h<output.size(2); h++){
                    for(int w=0; w<output.size(3); w++){
                        if output_sw[b][c][h][w] != output[b][c][h][w]{
                            std::cout << "Error: Result mismatch" << std::endl;
                            res = "fail"
                            break;

                        }
                    }
                }
            }
        }
    }

    double elapsed_time =(double)(end - start)
    double elapsed_time_sw =(double)(end_sw - start_sw)
    std::cout << "results" << res << std::endl;
    std::cout << "time_sw" << elapsed_time_sw << std::endl;
    std::cout << "time" << elapsed_time << std::endl;




}


// int main(int argc, char** argv)
// {
//     if (argc != 2) {
//         std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
// 		return EXIT_FAILURE;
// 	}

//     init_tensor(float *tensor, int batches, int channels, int height, int width)


//     // std::vector<int> MATRIX_SIZES{16,64,256,1024};
//     std::vector<int> MATRIX_SIZES{16};

//     for(unsigned int x=0; x<MATRIX_SIZES.size(); x++){
//     int MATRIX_SIZE = MATRIX_SIZES[x];
//     int TEST_SIZE = MATRIX_SIZE;
//     std::string binaryFile = argv[1];
//     size_t vector_size_bytes = sizeof(float) * BLOCK_MATRIX_SIZE * BLOCK_MATRIX_SIZE;
//     cl_int err;
//     unsigned fileBufSize;
//     // std::vector<int,aligned_allocator<int>> source_in1(DATA_SIZE);

//     // }
// // , aligned_allocator<std::vector>1
//     // Allocate Memory in Host Memory
//     std::vector<float, aligned_allocator<float>> source_in1(MATRIX_SIZE * MATRIX_SIZE);
//     std::vector<float, aligned_allocator<float>> source_in2(MATRIX_SIZE * MATRIX_SIZE);
//     std::vector<float, aligned_allocator<float>> source_hw_results(MATRIX_SIZE * MATRIX_SIZE, 0);
//     std::vector<float, aligned_allocator<float>> source_sw_results(MATRIX_SIZE * MATRIX_SIZE, 0);

//     // float source_in1[MATRIX_SIZE * MATRIX_SIZE];
//     // float source_in2[MATRIX_SIZE * MATRIX_SIZE];
//     // float source_hw_results[MATRIX_SIZE * MATRIX_SIZE] = { 0 };
//     // float source_sw_results[MATRIX_SIZE * MATRIX_SIZE] = { 0 };
//     // Create Random inputs
//     generate_square_matrix(MATRIX_SIZE, source_in1.data());
//     generate_square_matrix(MATRIX_SIZE, source_in2.data());

//     if (DEBUG){
//     // Print Inputs data 
//     std::cout << "IN 1: " << std::endl;
//     print_matrix(source_in1.data(), MATRIX_SIZE);
//     std::cout << "IN 2" << std::endl;
//     print_matrix(source_in2.data(), MATRIX_SIZE);
//     std::cout << "OUT 1" << std::endl;
//     print_matrix(source_hw_results.data(), MATRIX_SIZE);
//     std::cout << "OUT 2" << std::endl;
//     print_matrix(source_sw_results.data(), MATRIX_SIZE);
//     }
//     // for(int i=0; i<TEST_SIZE; i++){
//     //     for(int j=0; j<TEST_SIZE; j++){
//     //         for(int k=0; k<MATRIX_SIZE; k++){
//     //             source_sw_results[MATRIX_SIZE * i + j] += source_in1[MATRIX_SIZE * i + k] * source_in2[MATRIX_SIZE * k + j];
//     //         }
//     //     }
//     // }

//     matrix_multiplication(source_in1.data(), source_in2.data(), source_sw_results.data(), MATRIX_SIZE);

//     std::vector<cl::Device> devices = get_devices("Xilinx");
//     devices.resize(1);
//     cl::Device device = devices[0];


//     OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
//     OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE , &err));
//     char* fileBuf = read_binary_file(binaryFile, fileBufSize);
//     cl::Program::Binaries bins{{fileBuf, fileBufSize}};
//     OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

//     int BLOCK_COLS = int (MATRIX_SIZE/BLOCK_MATRIX_SIZE);
//     int BLOCK_ROWS = int (MATRIX_SIZE/BLOCK_MATRIX_SIZE);
//     int BLOCK_KS = int (MATRIX_SIZE/BLOCK_MATRIX_SIZE);

//     // std::cout << "BLOCK_COLS: " << BLOCK_COLS << std::endl;
//     // std::cout << "BLOCK_ROWS: " << BLOCK_ROWS << std::endl;
//     // std::cout << "BLOCK_KS: " << BLOCK_KS << std::endl;

//     std::vector<float, aligned_allocator<float>> block_in1(BLOCK_MATRIX_SIZE * BLOCK_MATRIX_SIZE);
//     std::vector<float, aligned_allocator<float>> block_in2(BLOCK_MATRIX_SIZE * BLOCK_MATRIX_SIZE);
//     std::vector<float, aligned_allocator<float>> block_out(BLOCK_MATRIX_SIZE * BLOCK_MATRIX_SIZE, 0);
    
//     // Stat Timer
//     auto start = chrono::system_clock::now();

//     // Block Multiplication loop
//     for (int block_col=0; block_col<BLOCK_COLS; block_col++){
//         for (int block_row=0; block_row<BLOCK_ROWS; block_row++){
//             for (int block_k=0; block_k<BLOCK_KS; block_k++){


//                 // Read to Block 1 and 2
//                 for (int i=0; i<BLOCK_MATRIX_SIZE; i++){
//                     for (int j=0; j<BLOCK_MATRIX_SIZE; j++){
//                         // Block 1 (Constant row)
//                         block_in1[i * BLOCK_MATRIX_SIZE + j] = source_in1[(i + (block_row*BLOCK_MATRIX_SIZE)) * MATRIX_SIZE + (j + (block_k*BLOCK_MATRIX_SIZE))];

//                         // Block 2 (Constant col)
//                         block_in2[i * BLOCK_MATRIX_SIZE + j] = source_in2[(i + (block_k*BLOCK_MATRIX_SIZE)) * MATRIX_SIZE + (j + (block_col*BLOCK_MATRIX_SIZE))];
//                     }
//                 }

    
    
    

//     // Read to Block 2
//     OCL_CHECK(err, cl::Kernel krnl_vector_add(program,"vdot", &err));


//     OCL_CHECK(err, cl::Buffer buffer_in1   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
//             vector_size_bytes, block_in1.data(), &err));
//     OCL_CHECK(err, cl::Buffer buffer_in2   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
//             vector_size_bytes, block_in2.data(), &err));
//     OCL_CHECK(err, cl::Buffer buffer_output(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
//             vector_size_bytes, block_out.data(), &err));

//     OCL_CHECK(err, err = krnl_vector_add.setArg(0, buffer_in1));
//     OCL_CHECK(err, err = krnl_vector_add.setArg(1, BLOCK_MATRIX_SIZE));
//     OCL_CHECK(err, err = krnl_vector_add.setArg(2, BLOCK_MATRIX_SIZE));
//     OCL_CHECK(err, err = krnl_vector_add.setArg(3, buffer_in2));
//     OCL_CHECK(err, err = krnl_vector_add.setArg(4, BLOCK_MATRIX_SIZE));
//     OCL_CHECK(err, err = krnl_vector_add.setArg(5, BLOCK_MATRIX_SIZE));
//     OCL_CHECK(err, err = krnl_vector_add.setArg(6, buffer_output));

//     // print_test(2);
//     // print_test(3);
//     // print_test(4);
// 	// print_test(5);

//     std::vector<cl::Event> events;

//     cl::Event write_event;
//     OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1, buffer_in2},0/* 0 means from host*/, NULL, &write_event));	
//     events.push_back(write_event);

//     cl::Event kernel_event;
//     OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add, NULL, &kernel_event));
//     events.push_back(kernel_event);
//     // q.flush();
//     q.finish();

//     cl::Event read_event;
//     OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output},CL_MIGRATE_MEM_OBJECT_HOST, NULL, &read_event));

//     // q.flush();
//     q.finish();
//     // sleep(5);

//     // Write to output
//     for (int i=0; i<BLOCK_MATRIX_SIZE; i++){
//         for (int j=0; j<BLOCK_MATRIX_SIZE; j++){
//             // Block Out
//             source_hw_results[(i + (block_row*BLOCK_MATRIX_SIZE)) * MATRIX_SIZE + (j + (block_col*BLOCK_MATRIX_SIZE))] += block_out[i * BLOCK_MATRIX_SIZE + j];
//         }
//     }

//     // std::cout << "Mutiplied col: " << block_col<< " with row: " << block_row << std::endl;
//     // print_matrix(block_in1.data(), BLOCK_MATRIX_SIZE);
//     // print_matrix(block_in2.data(), BLOCK_MATRIX_SIZE);

//     }
//     }
//     }

//     // End Timer
//     auto end = chrono::system_clock::now();

//     auto elapsed_time = chrono::duration_cast<chrono::nanoseconds>(end - start) / 1000000000.0;

    

// // OPENCL HOST CODE AREA END 12
//     // clock_t start = clock();

//     // clock_t end = clock();
//     // double elapsed_time =(double)(end - start);
//     // double elapsed_time =(double)(end - start)/(double)(CLOCKS_PER_SEC);

//     // Compare the results of the Device to the simulation
//     bool match = true;
//     if (DEBUG){
//         for (int i = 0 ; i < TEST_SIZE ; i++){
//             for (int j =0; j < TEST_SIZE; j++){
//                 if (int(source_hw_results[MATRIX_SIZE * i + j]) != int(source_sw_results[MATRIX_SIZE * i + j])){
//                     std::cout << "Error: Result mismatch" << std::endl;
//                     std::cout << "i = " << i << " CPU result = " << source_sw_results[MATRIX_SIZE * i + j]
//                         << " Device result = " << source_hw_results[MATRIX_SIZE * i + j] << std::endl;
//                     match = false;
//                     // break;
//                 }
//             }
//         }
    
    

//     std::cout << "OUT 1 (END)" << std::endl;
//     print_matrix(source_hw_results.data(), MATRIX_SIZE);
//     std::cout << "OUT 2 (END)" << std::endl;
//     print_matrix(source_sw_results.data(), MATRIX_SIZE);
//     // std::cout << "Matrix Size: " << MATRIX_SIZE << " Time Taken: " << (double)(end - start)/CLOCKS_PER_SEC << std::endl;
//     std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl; 
//     }
//     std::cout << "Matrix Size: " << MATRIX_SIZE << " Time Taken: " << elapsed_time.count() << std::endl;
//     delete[] fileBuf;
// }
//     return (EXIT_SUCCESS);

// }

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