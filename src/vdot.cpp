
/*
    Vector Addition Kernel Implementation 
    Arguments:
        in1   (input)     --> Input Vector1
        in2   (input)     --> Input Vector2
        out   (output)    --> Output Vector
        size  (input)     --> Size of Vector in Integer
        hkkjk
   */

//    https://ramyadhadidi.github.io/files/asgari-iccd20.pdf
// #define BUFFER_SIZE 3 /*Defining the size of the buffer*/
#define MAX_SIZE 16

extern "C" {
void vdot (const float *input1, const float *input2, 
            float *output, const int size) {

    #pragma HLS INTERFACE m_axi     port=input1 offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi     port=input2 offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi     port=output offset=slave bundle=gmem

    #pragma HLS INTERFACE s_axilite port=input1              bundle=control
    #pragma HLS INTERFACE s_axilite port=input2              bundle=control
    #pragma HLS INTERFACE s_axilite port=output              bundle=control
    #pragma HLS INTERFACE s_axilite port=size             bundle=control
    #pragma HLS INTERFACE s_axilite port=return           bundle=control
    

    float buffer1[MAX_SIZE][MAX_SIZE];
    float buffer2[MAX_SIZE][MAX_SIZE];
    float outbuffer[MAX_SIZE][MAX_SIZE] = { 0 };
    #pragma HLS ARRAY_PARTITION variable=buffer1 dim=2 complete 
    #pragma HLS ARRAY_PARTITION variable=buffer2 dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=outbuffer dim=0 complete

    for (int i=0; i<size; i++){
        #pragma HLS unroll
        #pragma HLS PIPELINE
        for (int j=0; j<size; j++){
            buffer1[i][j] = input1[size * i + j];
        }
    }

    for (int j=0; j<size; j++){
        #pragma HLS unroll
        #pragma HLS PIPELINE
        for (int i=0; i<size; i++){
            buffer2[i][j] = input2[size * i + j];
        }
    }

    for (int i=0; i<size; i++){
        #pragma HLS unroll
        #pragma HLS PIPELINE II=1
        for (int j=0; j<size; j++){
            for (int k=0; k<size; k++){
                outbuffer[i][j] += buffer1[i][k] * buffer2[k][j];
            }
        }
    }

    for (int j=0; j<size; j++){
        #pragma HLS unroll
        #pragma HLS PIPELINE
        for (int i=0; i<size; i++){
            output[size * i + j] = outbuffer[i][j];
        }
    }
}
}

// void vdot (const unsigned int *in1, const unsigned int *in2,
// 		   unsigned int *out, int size)