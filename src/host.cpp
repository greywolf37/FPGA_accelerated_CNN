#include "host.hpp"


torch::Tensor forward_sw(torch::Tensor input, torch::Tensor weights);

int main(int argc, char** argv){
    torch::Tensor tensor = torch::rand({2, 2, 3, 3});
    std::cout << "tensor" << std::endl;
    std::cout << tensor << std::endl;

    torch::Tensor output = forward_sw(tensor, tensor);
    std::cout << "Output" << std::endl;
    std::cout << output << std::endl;

    float * output_arr = tensor2arr_4d(output);
    std::cout << "Output Array" << std::endl;
    print_tensor(output_arr, output.size(0),output.size(1),output.size(2),output.size(3));

    torch::Tensor output2= arr2tensor_4d(output_arr, output.size(0),output.size(1),output.size(2),output.size(3));
    std::cout << "Output Tensor 2" << std::endl;
    std::cout << output2 << std::endl;

    return 0;
}


torch::Tensor forward_sw(torch::Tensor input, torch::Tensor weights){
    return input;
}