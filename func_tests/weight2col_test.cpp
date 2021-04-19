#include "../src/host.hpp"

int main(int argc, char** argv){

    int out_channels=2;
    int in_channels=2;
    int in_height=3;
    int in_width=3;

    int out_height;
    int out_width;

    float input_tensor[out_channels*in_channels*in_height*in_width];
    std::cout<<"Input:";
    init_tensor(input_tensor, out_channels, in_channels, in_height, in_width);
    print_tensor(input_tensor, out_channels, in_channels, in_height, in_width);

    float *out_tensor = weight2col(input_tensor, out_channels, in_channels, in_height, in_width, &out_height, &out_width);

    std::cout<<"Output:";
    print_tensor(out_tensor, 1, 1, out_height, out_width);
    
    // Remember to delete memory
    delete[] out_tensor;

    return 0;
}