#include <iostream>
#include <stdexcept>
#include <vector>

#include <torch/torch.h>

// Link for tutorials
// https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_2/ug1414-vitis-ai.pdf

float * img2col(float *in_tensor, int in_batches, int in_channels, int in_height, int in_width, 
            int kernel_height, int kernel_width, int stride, int pad, 
            int *out_height, int *out_width, int *out_shape_height, int *out_shape_width){
    
    int steps_height = ((in_height - kernel_height + 2*pad)/stride) + 1;
    int steps_width = ((in_width - kernel_width + 2*pad)/stride) + 1;
    *out_height = kernel_height*kernel_width*in_channels;
    *out_width = steps_height*steps_width*in_batches;

    *out_shape_height = steps_height;
    *out_shape_width = steps_width;

    float* out_tensor= new float[in_batches*(*out_height)*(*out_width)];

    int i=0; /*Slide number*/
    // Sliding kernel window
    for(int b=0; b<in_batches; b++){
        for(int h_in=0; h_in<in_height-kernel_height+1; h_in+=stride){
            for(int w_in=0; w_in<in_width-kernel_width+1; w_in+=stride){

                // Element in each kernel window
                for(int c=0; c<in_channels; c++){
                    for(int kh=0; kh<kernel_height; kh++){
                        for(int kw=0; kw<kernel_width; kw++){
                            out_tensor[
                                (*out_width)*(kernel_width*kernel_height*c+kernel_width*kh+kw)     /*height  of output*/
                                +(i)]  /*width of output*/
                                = in_tensor[(in_width*in_height*in_batches*b)+(in_width*in_height*c)+(in_width*(h_in+kh))+(w_in+kw)];
                            // std::cout<<(*out_width)*(kernel_width*kernel_height*c+kernel_width*kh+kw)+(i)<< " <- ";
                            // std::cout<<(in_width*in_height*in_batches*b)+(in_width*in_height*c)+(in_width*(h_in+kh))+(w_in+kw)<<std::endl;
                        }
                    }
                }
                i++;
            }
        }
    }

    return out_tensor;
}

float * col2img(float *in_tensor, int out_batches, int out_channels, int out_height, int out_width){

    float* out_tensor= new float[out_batches*out_channels*out_height*out_width];

    for(int b=0; b<out_batches; b++){
        for(int c=0; c<out_channels; c++){
            for(int h=0; h<out_height; h++){
                for(int w=0; w<out_width; w++){
                    out_tensor[(out_width*out_height*out_channels*b)+(out_width*out_height*c)+(out_width*h)+w]
                        = in_tensor[(out_width*out_height*out_batches*c)+((out_width*out_height*b)+(out_width*h) +(w))];
                }
            }
        }
    }

    return out_tensor;
}

float * weight2col(float *kernel, int out_channels, int in_channels, int kernel_height, int kernel_width,
                    int *out_height, int *out_width){

    *out_height = out_channels;
    *out_width = kernel_height * kernel_width * in_channels;

    float * out_tensor = new float[*out_height * (*out_width )];

    for(int o=0; o<out_channels; o++){
        for(int i=0; i<in_channels; i++){
            for(int h=0;h<kernel_height; h++){
                for(int w=0; w<kernel_width; w++){
                    out_tensor[(*out_width)*(o) + (kernel_width*kernel_height*i+ kernel_width*h+ w)] =
                    kernel[(kernel_height * kernel_width * in_channels* o) + (kernel_width*kernel_height*i) + (kernel_width*h) + (w)];
                }
            }
        }
    }

    return out_tensor;
}

float * tensor2arr_4d(torch::Tensor tensor, int *batches, int *in_channels, int *in_height, int *in_width){
    // int batches = tensor.size(0);
    // int channels = tensor.size(1);
    // int height = tensor.size(2);
    // int width = tensor.size(3);

    // float *array= new float[batches*channels*height*width];

    // for(int b=0; b<batches; b++){
    //     for(int c=0; c<channels; c++){
    //         for(int h=0; h<height; h++){
    //             for(int w=0; w<width; w++){
    //                 array[(width*height*channels*b)+(width*height*c)+(width*h)+(w)] =
    //                 tensor.index({b,c,h,w});
    //             }
    //         }
    //     }
    // }

    // return array;
    *batches = tensor.size(0);
    *in_channels = tensor.size(1);
    *in_height = tensor.size(2);
    *in_width = tensor.size(3);
    return tensor.data_ptr<float>();
}

torch::Tensor arr2tensor_4d(float *array, int batches, int channels, int height, int width){
    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    return torch::from_blob(array, {batches, channels, height, width});
}

torch::Tensor transpose_weights(torch::Tensor weights){

    int out_channels, in_channels, kernel_height, kernel_width;
    float * weights_array = tensor2arr_4d(weights, &out_channels, &in_channels, &kernel_height, &kernel_width);

    float * output_array = new float[out_channels*in_channels*kernel_height*kernel_width];

    for(int o=0; o<out_channels; o++){
        for(int i=0; i<in_channels; i++){
            for(int h=0; h<kernel_height; h++){
                for(int w=0; w<kernel_width; w++){
                    output_array[(kernel_width*kernel_height*in_channels*o)+(kernel_height*kernel_width*i)+(kernel_width*(kernel_height-h-1))+(kernel_width-w-1)]=
                    weights_array[(kernel_width*kernel_height*in_channels*o)+(kernel_height*kernel_width*i)+(kernel_width*(h))+(w)];
                }
            }
        }
    }

    return arr2tensor_4d(output_array, out_channels, in_channels, kernel_height, kernel_width);
}

float * matmul_sw(float *matrix1, int height1, int width1,
                    float *matrix2, int height2, int width2){
    if (height2 != width1){
        throw std::invalid_argument("Matrix multiplication dim mismatch");
    }

    float * out_matrix= new float[height1*width2];

    for(int i=0; i<width2; i++){
        for(int j=0; j<height1; j++){
            out_matrix[(width2*j)+(i)] = 0;

            for(int k=0; k<height2; k++){
                out_matrix[(width2*j)+(i)] +=
                matrix1[(width1*j)+k] * matrix2[(width2*k)+i];
            }
        }
    }

    return out_matrix;

}

void print_tensor(float *tensor, int batches, int channels, int height, int width){
    std::cout<<std::endl;
    for(int b=0; b<batches; b++){
        for(int c=0; c<channels; c++){

            for(int h=0; h<height; h++){
                for(int w=0; w<width; w++){
                    std::cout<< tensor[height*width*channels*b + height*width*c + width*h + w];
                    std::cout<<"\t";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }
}

void init_tensor(float *tensor, int batches, int channels, int height, int width){
    int i=0;
    for(int b=0; b<batches; b++){
        for(int c=0; c<channels; c++){
            for(int h=0; h<height; h++){
                for(int w=0; w<width; w++){
                    tensor[height*width*channels*b + height*width*c + width*h + w] = i;
                    i++;

                }
            }
        }
    }
}