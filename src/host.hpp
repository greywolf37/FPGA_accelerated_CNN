#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

//OCL_CHECK doesn't work if call has templatized function call
#define OCL_CHECK(error,call)                                       \
    call;                                                           \
    if (error != CL_SUCCESS) {                                      \
      printf("%s:%d Error calling " #call ", error code is: %d\n",  \
              __FILE__,__LINE__, error);                            \
      exit(EXIT_FAILURE);                                           \
    }
// #define DATA_SIZE 4096
#include <vector>
#include <unistd.h>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <CL/cl2.hpp>
#include <time.h>
#include <string>
#include <stdlib.h>
#include <torch/torch.h>
#include "hls_stream.h" /*For hls*/


template <typename T>
struct aligned_allocator
{
  using value_type = T;
  T* allocate(std::size_t num)
  {
    void* ptr = nullptr;
    if (posix_memalign(&ptr,4096,num*sizeof(T)))
      throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
  }
  void deallocate(T* p, std::size_t num)
  {
    free(p);
  }
};

std::vector<cl::Device> get_devices(const std::string& vendor_name) {

    size_t i;
    cl_int err;
    std::vector<cl::Platform> platforms;
    OCL_CHECK(err, err = cl::Platform::get(&platforms));
    cl::Platform platform;
    for (i  = 0 ; i < platforms.size(); i++){
        platform = platforms[i];
        OCL_CHECK(err, std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(&err));
        if (platformName == vendor_name){
            std::cout << "Found Platform" << std::endl;
            std::cout << "Platform Name: " << platformName.c_str() << std::endl;
            break;
        }
    }
    if (i == platforms.size()) {
        std::cout << "Error: Failed to find Xilinx platform" << std::endl;
        exit(EXIT_FAILURE);
    }
   
    //Getting ACCELERATOR Devices and selecting 1st such device 
    std::vector<cl::Device> devices;
    OCL_CHECK(err, err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices));
    return devices;
}
   
char* read_binary_file(const std::string &xclbin_file_name, unsigned &nb) 
{
    std::cout << "INFO: Reading " << xclbin_file_name << std::endl;

	if(access(xclbin_file_name.c_str(), R_OK) != 0) {
		printf("ERROR: %s xclbin not available please build\n", xclbin_file_name.c_str());
		exit(EXIT_FAILURE);
	}
    //Loading XCL Bin into char buffer 
    std::cout << "Loading: '" << xclbin_file_name.c_str() << "'\n";
    std::ifstream bin_file(xclbin_file_name.c_str(), std::ifstream::binary);
    bin_file.seekg (0, bin_file.end);
    nb = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    char *buf = new char [nb];
    bin_file.read(buf, nb);
    return buf;
}

// int generate_square_matrix(int size, float *output){
//     int range = 100000;
//     for(int i=0; i<size; i++){
//         // output.push_back(std::vector<float> (size));
//         for(int j=0; j<size; j++){
//             output[size * i + j] = i;
//             // output[size * i + j] = (float)((rand() % (range)) - (range/2));
//         }
//     }
//     return 0;
// }

// int matrix_multiplication(float *input1, float *input2, float *output, int size){
//     for(int i=0; i< size; i++){
//         for(int j=0; j<size; j++){
//             for(int k=0; k<size; k++){
//                 output[size * i + j] += input1[size * i + k] * input2[size * k + j];
//             }
//         }
//     }
//     return 0;
// }

// int print_matrix(float *input1, int size){
//     for(int i=0; i<size; i++){
//         for(int j=0; j<size; j++){
//             std::cout << input1[size * i + j] << " ";
//         }
//         std::cout << std::endl;
//     }
//     std::cout << std::endl;
//     return 0;
// }

// void print_test(int i){
//     std::cout << "Test: " << i << std::endl;
// }

// -----------------------------------------------------------------------------------

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

float * weight_update_img2col(float *in_tensor, int in_batches, int in_channels, int in_height, int in_width, 
            int kernel_height, int kernel_width, int stride, int pad, 
            int *out_height, int *out_width, int *out_shape_height, int *out_shape_width){

    int steps_height = ((in_height - kernel_height + 2*pad)/stride) + 1;
    int steps_width = ((in_width - kernel_width + 2*pad)/stride) + 1;
    *out_height = kernel_height*kernel_width*in_batches;
    *out_width = steps_height*steps_width*in_channels;

    *out_shape_height = steps_height;
    *out_shape_width = steps_width;

    float* out_tensor= new float[(*out_height)*(*out_width)];

    int i=0; /*Slide number*/
    // Sliding kernel window
    for(int c=0; c<in_channels; c++){
        for(int h_in=0; h_in<in_height-kernel_height+1; h_in+=stride){
            for(int w_in=0; w_in<in_width-kernel_width+1; w_in+=stride){

                // Element in each kernel window
                for(int b=0; b<in_batches; b++){
                    for(int kh=0; kh<kernel_height; kh++){
                        for(int kw=0; kw<kernel_width; kw++){
                            out_tensor[
                                (*out_width)*(kernel_width*kernel_height*b+kernel_width*kh+kw)     /*height  of output*/
                                +(i)]  /*width of output*/
                                = in_tensor[(in_width*in_height*in_channels*b)+(in_width*in_height*c)+(in_width*(h_in+kh))+(w_in+kw)];
                            // std::cout<<(*out_width)*(kernel_width*kernel_height*b+kernel_width*kh+kw)+(i)<< " <- ";
                            // std::cout<<(in_width*in_height*in_batches*b)+(in_width*in_height*c)+(in_width*(h_in+kh))+(w_in+kw);
                            // std::cout<<"  ("<<in_tensor[(in_width*in_height*in_channels*b)+(in_width*in_height*c)+(in_width*(h_in+kh))+(w_in+kw)]<<")"<<std::endl;
                        }
                    }
                }
                i++;
            }
        }
    }
    return out_tensor;
}

float * weight_update_weight2col(float *out_grad, int out_batches, int out_channels, int out_height, int out_width,
                    int *out_height_shape, int *out_width_shape){

    *out_height_shape = out_channels;
    *out_width_shape = out_height * out_width * out_batches;

    float * out_tensor = new float[*out_height_shape * (*out_width_shape)];

    for(int o=0; o<out_batches; o++){
        for(int i=0; i<out_channels; i++){
            for(int h=0;h<out_height; h++){
                for(int w=0; w<out_width; w++){
                    out_tensor[(*out_width_shape)*(i) + (out_width*out_height*o+ out_width*h+ w)] =
                    out_grad[(out_height * out_width * out_channels* o) + (out_width*out_height*i) + (out_width*h) + (w)];
                }
            }
        }
    }

    return out_tensor;
}

float * weight_update_col2img(float *in_tensor, int out_channels, int in_channels, int out_height, int out_width){

    float* out_tensor= new float[out_channels*in_channels*out_height*out_width];

    for(int b=0; b<out_channels; b++){
        for(int c=0; c<in_channels; c++){
            for(int h=0; h<out_height; h++){
                for(int w=0; w<out_width; w++){
                    out_tensor[(out_width*out_height*in_channels*b)+(out_width*out_height*c)+(out_width*h)+w]
                        = in_tensor[(out_width*out_height*in_channels*b)+((out_width*out_height*c)+(out_width*h) +(w))];
                }
            }
        }
    }

    return out_tensor;
}

float * input_grad_img2col(float *in_tensor, int in_batches, int in_channels, int in_height, int in_width, 
            int kernel_height, int kernel_width, int stride, int pad, 
            int *out_height, int *out_width, int *out_shape_height, int *out_shape_width){

    int steps_height = ((in_height - kernel_height + 2*pad)/stride) + 1;
    int steps_width = ((in_width - kernel_width + 2*pad)/stride) + 1;
    *out_height = kernel_height*kernel_width*in_channels;
    *out_width = steps_height*steps_width*in_batches;

    *out_shape_height = steps_height;
    *out_shape_width = steps_width;

    float* out_tensor= new float[(*out_height)*(*out_width)];

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
                                = in_tensor[(in_width*in_height*in_channels*b)+(in_width*in_height*c)+(in_width*(h_in+kh))+(w_in+kw)];
                            // std::cout<<(*out_width)*(kernel_width*kernel_height*b+kernel_width*kh+kw)+(i)<< " <- ";
                            // std::cout<<(in_width*in_height*in_batches*b)+(in_width*in_height*c)+(in_width*(h_in+kh))+(w_in+kw);
                            // std::cout<<"  ("<<in_tensor[(in_width*in_height*in_channels*b)+(in_width*in_height*c)+(in_width*(h_in+kh))+(w_in+kw)]<<")"<<std::endl;
                        }
                    }
                }
                i++;
            }
        }
    }
    return out_tensor;
}

float * input_grad_weight2col(float *weights, int out_channels, int in_channels, int out_height, int out_width,
                    int *out_height_shape, int *out_width_shape){

    *out_height_shape = in_channels;
    *out_width_shape = out_height * out_width * out_channels;

    float * out_tensor = new float[*out_height_shape * (*out_width_shape)];

    for(int o=0; o<out_channels; o++){
        for(int i=0; i<in_channels; i++){
            for(int h=0;h<out_height; h++){
                for(int w=0; w<out_width; w++){
                    out_tensor[(*out_width_shape)*(i) + (out_width*out_height*o+ out_width*h+ w)] =
                    weights[(out_height * out_width * in_channels* o) + (out_width*out_height*i) + (out_width*h) + (w)];
                }
            }
        }
    }

    return out_tensor;
}

float * input_grad_col2img(float *in_tensor, int out_batches, int in_channels, int out_height, int out_width){

    float* out_tensor= new float[out_batches*in_channels*out_height*out_width];

    for(int b=0; b<out_batches; b++){
        for(int c=0; c<in_channels; c++){
            for(int h=0; h<out_height; h++){
                for(int w=0; w<out_width; w++){
                    out_tensor[(out_width*out_height*in_channels*b)+(out_width*out_height*c)+(out_width*h)+w]
                        = in_tensor[(out_width*out_height*out_batches*c)+((out_width*out_height*b)+(out_width*h) +(w))];
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

Matrix tensor2matrix(torch::Tensor tensor) {
    // float * data_p = new float[tensor.numel()];
    // std::memcpy(data_p, tensor.data_ptr<float>(), sizeof(float)*tensor.numel());
    // return Matrix (data_p, tensor.size(0),tensor.size(1),tensor.size(2),tensor.size(3))

    Matrix matrix = Matrix (tensor.data_ptr<float>(), tensor.size(0),tensor.size(1),tensor.size(2),tensor.size(3));
    return Matrix (tensor.data_ptr<float>(), tensor.size(0),tensor.size(1),tensor.size(2),tensor.size(3));
}

torch::Tensor matrix2tensor(Matrix matrix) {
    // float * data_p = new float[matrix.dim1 * matrix.dim2 * matrix.dim3 *matrix.dim4];
    // std::memcpy(data_p, matrix.data_ptr(), sizeof(float)*matrix.dim1 * matrix.dim2 * matrix.dim3 *matrix.dim4);

    // auto options = torch::TensorOptions().dtype(torch::kFloat64);
    //  return torch::from_blob(data_p, {matrix.dim1, matrix.dim2, matrix.dim3, matrix.dim4});

    auto options = torch::TensorOptions().dtype(torch::kFloat64);
     return torch::from_blob(matrix.data_ptr(), {matrix.dim1, matrix.dim2, matrix.dim3, matrix.dim4});
}

float * transpose_weights(torch::Tensor weights, int *w, int *x, int *y, int *z){

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
    *w = out_channels;
    *x = in_channels;
    *y = kernel_height;
    *z = kernel_width;
    return output_array;
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

float * pad_array(float * array, int batches, int channels, int height, int width, 
                    int pad_height, int pad_width, int *out_batches, int *out_channels, int *out_height, int *out_width){
    *out_batches = batches;
    *out_channels = channels;
    *out_height = height + 2 * pad_height;
    *out_width = width + 2 * pad_width;

    float * out_array = new float[*out_batches * (*out_channels) * (*out_height) * (*out_width)];

    for(int b=0; b<*out_batches;b++){
        for(int c=0; c<*out_channels; c++){
            for(int h_=0; h_<*out_height; h_++){
                for(int w_=0; w_<*out_width; w_++){
                    
                    if((h_<pad_height) || (w_<pad_width) || 
                        (h_>=height+pad_height) || (w_>=width+pad_width)){
                        // Padding with zeroes
                        out_array[(*out_height * (*out_width)*(*out_channels)*b)+(*out_height * (*out_width)*c)+(*out_width*h_)+(w_)]=0;
                    }else{
                        out_array[(*out_height * (*out_width)*(*out_channels)*b)+(*out_height * (*out_width)*c)+(*out_width*h_)+(w_)]=
                        array[(height*width*channels*b)+(height * width*c)+(width*(h_-pad_height))+(w_-pad_width)];
                    }

                }
            }
        }
    }
    return out_array;

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