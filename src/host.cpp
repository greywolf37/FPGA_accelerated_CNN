#include "host.hpp"

torch::Tensor forward_sw(torch::Tensor input, torch::Tensor weights);


int main(int argc, char** argv){

    int batches=1;
    int in_channels=2;
    int in_height=3;
    int in_width=3;

    int kernel_height=2;
    int kernel_width=2;

    int out_channels=2;

    float in_array[batches*in_channels*in_height*in_width];
    init_tensor(in_array, batches, in_channels, in_height, in_width);

    float kernel_array[out_channels*in_channels*kernel_height*kernel_width];
    init_tensor(kernel_array, out_channels, in_channels, kernel_height, kernel_width);

    torch::Tensor input = arr2tensor_4d(in_array, batches, in_channels, in_height, in_width);
    torch::Tensor weights = arr2tensor_4d(kernel_array, out_channels, in_channels, kernel_height, kernel_width);

    torch::Tensor output = forward_sw(input, weights);

    std::cout << "Input" << std::endl;
    std::cout << input << std::endl;
    std::cout << "Kernel" << std::endl;
    std::cout << weights << std::endl;
    std::cout << "Output" << std::endl;
    std::cout << output << std::endl;


    // torch::Tensor tensor = torch::rand({1, 1, 3, 3});
    // std::cout << "tensor" << std::endl;
    // std::cout << tensor << std::endl;

    // torch::Tensor output = forward_sw(tensor, tensor);
    // std::cout << "Output" << std::endl;
    // std::cout << output << std::endl;

    // float * output_arr = tensor2arr_4d(output);
    // std::cout << "Output Array" << std::endl;
    // print_tensor(output_arr, output.size(0),output.size(1),output.size(2),output.size(3));

    // torch::Tensor output2= arr2tensor_4d(output_arr, output.size(0),output.size(1),output.size(2),output.size(3));
    // std::cout << "Output Tensor 2" << std::endl;
    // std::cout << output2 << std::endl;

    // return 0;




    // int out_channels=2;
    // int in_channels=2;
    // int in_height=3;
    // int in_width=3;

    // int out_height;
    // int out_width;

    // float input_tensor[out_channels*in_channels*in_height*in_width];
    // std::cout<<"Input:";
    // init_tensor(input_tensor, out_channels, in_channels, in_height, in_width);
    // print_tensor(input_tensor, out_channels, in_channels, in_height, in_width);

    // float *out_tensor = weight2col(input_tensor, out_channels, in_channels, in_height, in_width, &out_height, &out_width);

    // std::cout<<"Output:";
    // print_tensor(out_tensor, 1, 1, out_height, out_width);
    
    // // Remember to delete memory
    // delete[] out_tensor;
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
    

    return output;
}

// vector<torch::Tensor> backward_sw(torch::Tensor output_grad, torch::Tensor input, torch::Tensor weights, torch::Tensor output){
//     return {input, weights};
// }