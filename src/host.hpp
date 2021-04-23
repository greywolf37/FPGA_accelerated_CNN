#include <iostream>
#include <stdexcept>
#include <vector>
#include <tuple>

#include <torch/torch.h>

// Link for tutorials
// https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_2/ug1414-vitis-ai.pdf

class Matrix {
        float * array;
    public:
        int dim1, dim2, dim3, dim4;
        Matrix (int, int, int, int);
        Matrix (float *, int, int, int, int);
        void set(float, int, int, int, int);
        float get(int, int, int, int);
        float * data_ptr(void);
        void print(void);
        void delete_data(void);
        // ~Matrix ();

};

Matrix::Matrix (int a, int b, int c, int d) {
    dim1 = a;
    dim2 = b;
    dim3 = c;
    dim4 = d;
    array = new float[dim1*dim2*dim3*dim4];
}

Matrix::Matrix (float * array_ptr, int a, int b, int c, int d) {
    dim1 = a;
    dim2 = b;
    dim3 = c;
    dim4 = d;
    array = new float[a*b*c*d];
    std::memcpy(array, array_ptr, sizeof(float)*a*b*c*d);
}

void Matrix::set (float value, int a, int b, int c, int d) {
    array[(dim2 * dim3 * dim4 * a) + (dim3 * dim4 * b) + (dim4 * c) + (d)] = value;
}

float Matrix::get (int a, int b, int c, int d) {
    return array[(dim2 * dim3 * dim4 * a) + (dim3 * dim4 * b) + (dim4 * c) + (d)];
}

float * Matrix::data_ptr () {
    return array;
}

void Matrix::print () {
    std::cout<<std::endl;
    for(int b=0; b<dim1; b++){
        for(int c=0; c<dim2; c++){

            for(int h=0; h<dim3; h++){
                for(int w=0; w<dim4; w++){
                    std::cout<< get(b, c, h, w);
                    std::cout<<"\t";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }
}

void Matrix::delete_data () {
    delete[] array;
}

// Matrix::~Matrix () {
//     if (array) {
//     delete[] array;
//     }
// }


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

Matrix weight_update_img2col(Matrix output_grad, Matrix input, int stride, int pad, int *out_shape_height, int *out_shape_width){

    int batches = input.dim1;
    int in_channels = input.dim2;
    int in_height = input.dim3;
    int in_width = input.dim4;

    int kernel_height = output_grad.dim3;
    int kernel_width = output_grad.dim4;

    int steps_height = ((in_height - kernel_height + 2*pad)/stride) + 1;
    int steps_width = ((in_width - kernel_width + 2*pad)/stride) + 1;

    *out_shape_height = steps_height;
    *out_shape_width = steps_width;

    Matrix out_tensor(1, 1, kernel_height*kernel_width*batches, steps_height*steps_width*in_channels);
    
    int i=0; /*Slide number*/
    // Sliding kernel window
    for(int c=0; c<in_channels; c++){
        for(int h_in=0; h_in<in_height-kernel_height+1; h_in+=stride){
            for(int w_in=0; w_in<in_width-kernel_width+1; w_in+=stride){

                // Element in each kernel window
                for(int b=0; b<batches; b++){
                    for(int kh=0; kh<kernel_height; kh++){
                        for(int kw=0; kw<kernel_width; kw++){
                            out_tensor.set(
                                input.get(
                                // Getting indices
                                b, c, h_in+kh, w_in+kw
                                ),
                                // setting indices
                                1, 1, kernel_width*kernel_height*b+kernel_width*kh+kw, /*height  of output*/
                                i /*width of output*/
                            );
            std::cout<<out_tensor.dim1<<std::endl;
            std::cout<<out_tensor.dim2<<std::endl;
            std::cout<<out_tensor.dim3<<std::endl;
            std::cout<<out_tensor.dim4<<std::endl;
            // std::cout<<out_tensor.get(1, 1, kernel_width*kernel_height*b+kernel_width*kh+kw, i )<<std::endl;

                                
                        }
                    }
                }
                i++;
            }
        }
    }

    std::cout<<"in test"<<std::endl;
    out_tensor.print();

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