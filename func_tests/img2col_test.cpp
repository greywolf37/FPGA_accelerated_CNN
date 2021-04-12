#include "../src/host.hpp"

int main(int argc, char** argv){

    int in_batches=1;
    int in_channels=2;
    int in_height=3;
    int in_width=3;

    int kernel_height=2;
    int kernel_width=2;
    int stride=1;
    int pad=0;

    float input_tensor[in_batches*in_channels*in_height*in_width];
    std::cout<<"Input:";
    init_tensor(input_tensor, in_batches, in_channels, in_height, in_width);
    print_tensor(input_tensor, in_batches, in_channels, in_height, in_width);

    int out_height;
    int out_width;

    float * output_tensor = img2col(input_tensor, in_batches, in_channels, in_height, in_width, 
            kernel_height, kernel_width, stride, pad, 
            &out_height, &out_width);

    std::cout<<"Output:";
    print_tensor(output_tensor, 1, 1, out_height, out_width);
    
    // Remember to delete memory
    delete[] output_tensor;

    return 0;
}