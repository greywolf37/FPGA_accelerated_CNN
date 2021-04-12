#include "../src/host.hpp"

int main(int argc, char** argv){

    int height1=1;
    int width1=3;
    int height2=3;
    int width2=1;

    float matrix1[height1*width1];
    std::cout<<"Input1:";
    init_tensor(matrix1, 1, 1, height1, width1);
    print_tensor(matrix1, 1, 1, height1, width1);

    float matrix2[height2*width2];
    std::cout<<"Input2:";
    init_tensor(matrix2, 1, 1, height2, width2);
    print_tensor(matrix2, 1, 1, height2, width2);

    float *out_tensor = matmul_sw(matrix1, height1, width1,
                    matrix2, height2, width2);

    std::cout<<"Output:";
    print_tensor(out_tensor, 1, 1, height1, width2);
    
    // Remember to delete memory
    delete[] out_tensor;

    return 0;
}