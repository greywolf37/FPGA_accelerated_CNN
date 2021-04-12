#include "../src/host.hpp"

int main(int argc, char** argv){

    int out_batches=2;
    int out_channels=2;
    int out_height=3;
    int out_width=3;

    float input_tensor[out_batches*out_channels*out_height*out_width];
    std::cout<<"Input:";
    init_tensor(input_tensor, 1, 1, out_channels, out_batches*out_height*out_width);
    print_tensor(input_tensor, 1, 1, out_channels, out_batches*out_height*out_width);

    float *out_tensor = col2img(input_tensor, out_batches, out_channels, out_height, out_width);

    std::cout<<"Output:";
    print_tensor(out_tensor, out_batches, out_channels, out_height, out_width);
    
    // Remember to delete memory
    delete[] out_tensor;

    return 0;
}