#include "host.hpp"

torch::Tensor forward_sw(torch::Tensor input, torch::Tensor weights);
std::vector<torch::Tensor> backward_sw(torch::Tensor output_grad,
            torch::Tensor input, torch::Tensor weights);


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
    // std::cout<< weight_weight2col_width <<std::endl;
    // std::cout<< weight_grad_height <<std::endl;
    float * weight_grad_col_arr = matmul_sw(weight_update_weight2col_arr, weight_weight2col_height, weight_weight2col_width,
                    weight_update_img2col_arr, weight_img2col_height, weight_img2col_width);
    int weight_grad_col_height = weight_weight2col_height;
    int weight_grad_col_width = weight_img2col_width;

    // col2img for weight update
    float * weight_grad_arr = weight_update_col2img(weight_grad_col_arr, out_channels_output, in_channels_input, weight_grad_height, weight_grad_width);

    // Converting to tensor
    torch::Tensor weight_grad = arr2tensor_4d(weight_grad_arr, out_channels_output, in_channels_input, weight_grad_height, weight_grad_width);
    
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


    // Deleting intermediate arrays
    delete[] og_pad_arr, weights_T_arr, ig_i2c_arr, ig_w2c_arr,
                input_grad_col_arr;

    std::cout << "------------*******************-----------------" << std::endl;

    return {input_grad, weight_grad};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_sw", &forward_sw, "Forward");
  m.def("backward_sw", &backward_sw, "Backward");
}