#include "host.hpp"

torch::Tensor forward_sw(torch::Tensor input, torch::Tensor weights);
std::tuple<torch::Tensor, torch::Tensor> backward_sw(torch::Tensor output_grad,
            torch::Tensor input, torch::Tensor weights);

void forward_sw_test();
void backward_sw_test();
void backward_sw();
void pad_array_test();
void test_Matrix();

int main(int argc, char** argv){
    backward_sw_test();
    return 0;
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

std::tuple<torch::Tensor, torch::Tensor> backward_sw(torch::Tensor output_grad,
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
    float * output_grad_mat = tensor2arr_4d(output_grad, &out_batches, &out_channels_output, &out_height, &out_width);

    // img2col for weight update (using input_arr)
    int weight_img2col_height, weight_img2col_width; /*Shape of output col*/
    int weight_grad_height, weight_grad_width; /*Shape of weight grad*/
    float * weight_update_img2col_arr = weight_update_img2col(input_arr, in_batches, in_channels_input, in_height, in_width, /*input array*/ 
            out_height, out_width, /*We treat the output_grad as kernel height and width*/
            stride, pad, /*pad and stride are identical to what is used in forward*/
            &weight_img2col_height, &weight_img2col_width, &weight_grad_height, &weight_grad_width);


    print_tensor(weight_update_img2col_arr, 1, 1, weight_img2col_height, weight_img2col_width);

    std::cout << "------------*******************-----------------"<< std::endl;

    return {input, weights};
}

void forward_sw_test(){

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
}

void backward_sw_test() {

    int pad = 0;
    int stride = 1;
    int batches=2;
    int in_channels=1;
    int in_height=3;
    int in_width=3;

    int kernel_height=2;
    int kernel_width=2;

    int out_channels=2;

    int out_height = ((in_height - kernel_height + 2*pad)/stride) + 1;
    int out_width = ((in_width - kernel_width + 2*pad)/stride) + 1;

    float in_array[batches*in_channels*in_height*in_width];
    init_tensor(in_array, batches, in_channels, in_height, in_width);

    float kernel_array[out_channels*in_channels*kernel_height*kernel_width];
    init_tensor(kernel_array, out_channels, in_channels, kernel_height, kernel_width);

    float output_grad_array[batches*out_channels*out_height*out_width];
    init_tensor(output_grad_array, batches, out_channels, out_height, out_width);

    torch::Tensor input = arr2tensor_4d(in_array, batches, in_channels, in_height, in_width);
    torch::Tensor weights = arr2tensor_4d(kernel_array, out_channels, in_channels, kernel_height, kernel_width);
    torch::Tensor output_grad = arr2tensor_4d(output_grad_array, batches, out_channels, out_height, out_width);

    auto [input_grad, weights_grad] = backward_sw(output_grad, input, weights);

    std::cout << "Input" << std::endl;
    std::cout << input << std::endl;
    std::cout << "Kernel" << std::endl;
    std::cout << weights << std::endl;
    std::cout << "Output Grad" << std::endl;
    std::cout << output_grad << std::endl;
    std::cout << "Weight Grad" << std::endl;
    std::cout << weights_grad << std::endl;
    std::cout << "Input Grad" << std::endl;
    std::cout << input_grad << std::endl;
}

void pad_array_test(){

    int batches=2;
    int in_channels=2;
    int in_height=3;
    int in_width=3;

    int pad_height = 1;
    int pad_width = 1;

    int kernel_height=2;
    int kernel_width=2;

    int out_channels=2;

    float in_array[batches*in_channels*in_height*in_width];
    init_tensor(in_array, batches, in_channels, in_height, in_width);

    int x_batches, x_channels, x_height, x_width;
    float * output = pad_array(in_array, batches, in_channels, in_height, in_width,
                                pad_height, pad_width, 
                                &x_batches, &x_channels, &x_height, &x_width);

    std::cout << "Input" << std::endl;
    print_tensor(in_array, batches, in_channels, in_height, in_width);

    std::cout << "Output" << std::endl;
    print_tensor(output, x_batches, x_channels, x_height, x_width);

}

void test_Matrix() {
    Matrix mat1 (3, 3, 2, 2);

    init_tensor(mat1.data_ptr(), mat1.dim1, mat1.dim2, mat1.dim3, mat1.dim4);

    std::cout << "Input" << std::endl;
    mat1.print();

    std::cout << "Value" << std::endl;
    std::cout << mat1.get(1, 1, 1, 1) << std::endl;
    
    mat1.set(1000, 1, 1, 1, 1);
    std::cout << "Altered" << std::endl;
    mat1.print();
}