
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
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

extern "C" {
void vdot (const float *input1, const int input1_h, const int input1_w, 
            const float *input2, const int input2_h, const int input2_w,
            float *output) {

    #pragma HLS INTERFACE m_axi     port=input1 offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi     port=input2 offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi     port=output offset=slave bundle=gmem

    #pragma HLS INTERFACE s_axilite port=input1              bundle=control
    #pragma HLS INTERFACE s_axilite port=input2              bundle=control
    #pragma HLS INTERFACE s_axilite port=output              bundle=control
    #pragma HLS INTERFACE s_axilite port=input1_h             bundle=control
    #pragma HLS INTERFACE s_axilite port=input1_w             bundle=control
    #pragma HLS INTERFACE s_axilite port=input2_h             bundle=control
    #pragma HLS INTERFACE s_axilite port=input2_w             bundle=control
    #pragma HLS INTERFACE s_axilite port=return           bundle=control
    

    float buffer1[MAX_SIZE][MAX_SIZE];
    float buffer2[MAX_SIZE][MAX_SIZE];
    float outbuffer[MAX_SIZE][MAX_SIZE] = { 0 };
    #pragma HLS ARRAY_PARTITION variable=buffer1 dim=2 complete 
    #pragma HLS ARRAY_PARTITION variable=buffer2 dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=outbuffer dim=0 complete



    for(int b_i_1=0; b_i_1 < input1_w; b_i_1+=MAX_SIZE){

        int blk_h_1 = input1_h;
        int blk_w_1 = MIN(MAX_SIZE, input1_w-b_i_1);
        int blk_h_2 = blk_w_1;
        int blk_w_2 = input2_w;
        int blk_h_o = input1_h;
        int blk_w_o = input2_w;

        for (int j=0; j<input1_h; j++){
            #pragma HLS unroll
            #pragma HLS PIPELINE
            for (int i=b_i_1; i<b_i_1+blk_w_1; i++){
                buffer1[j][i] = input1[input1_w * j + (i+b_i_1)];
            }
        }

        for (int i=0; i<input2_w; i++){
            #pragma HLS unroll
            #pragma HLS PIPELINE
            for (int j=b_i_1; j<b_i_1+blk_w_1; j++){
                buffer2[j][i] = input2[input2_w * (j+b_i_1) + i];
            }
        }

        for (int i=0; i<input1_h; i++){
            #pragma HLS unroll
            #pragma HLS PIPELINE II=1
            for (int j=0; j<input2_w; j++){
                for (int k=0; k<input1_w; k++){
                    outbuffer[i][j] += buffer1[i][k] * buffer2[k][j];
                }
            }
        }
    }

    for (int j=0; j<input2_w; j++){
        #pragma HLS unroll
        #pragma HLS PIPELINE
        for (int i=0; i<input1_h; i++){
            output[input2_w * i + j] = outbuffer[i][j];
        }
    }
}
}
// #define MAX_SIZE 16

// extern "C" {
// void vdot (const float *input1, const int input1_h, const int input1_w, 
//             const float *input2, const int input2_h, const int input2_w,
//             float *output) {

//     #pragma HLS INTERFACE m_axi     port=input1 offset=slave bundle=gmem
//     #pragma HLS INTERFACE m_axi     port=input2 offset=slave bundle=gmem
//     #pragma HLS INTERFACE m_axi     port=output offset=slave bundle=gmem

//     #pragma HLS INTERFACE s_axilite port=input1              bundle=control
//     #pragma HLS INTERFACE s_axilite port=input2              bundle=control
//     #pragma HLS INTERFACE s_axilite port=output              bundle=control
//     #pragma HLS INTERFACE s_axilite port=input1_h             bundle=control
//     #pragma HLS INTERFACE s_axilite port=input1_w             bundle=control
//     #pragma HLS INTERFACE s_axilite port=input2_h             bundle=control
//     #pragma HLS INTERFACE s_axilite port=input2_w             bundle=control
//     #pragma HLS INTERFACE s_axilite port=return           bundle=control
    

//     float buffer1[MAX_SIZE][MAX_SIZE];
//     float buffer2[MAX_SIZE][MAX_SIZE];
//     float outbuffer[MAX_SIZE][MAX_SIZE] = { 0 };
//     #pragma HLS ARRAY_PARTITION variable=buffer1 dim=2 complete 
//     #pragma HLS ARRAY_PARTITION variable=buffer2 dim=1 complete
//     #pragma HLS ARRAY_PARTITION variable=outbuffer dim=0 complete

//     for (int i=0; i<input1_h; i++){
//         #pragma HLS unroll
//         #pragma HLS PIPELINE
//         for (int j=0; j<input1_w; j++){
//             buffer1[i][j] = input1[input1_w * i + j];
//         }
//     }

//     for (int j=0; j<input2_w; j++){
//         #pragma HLS unroll
//         #pragma HLS PIPELINE
//         for (int i=0; i<input2_h; i++){
//             buffer2[i][j] = input2[input2_w * i + j];
//         }
//     }

//     for (int i=0; i<input1_h; i++){
//         #pragma HLS unroll
//         #pragma HLS PIPELINE II=1
//         for (int j=0; j<input2_w; j++){
//             for (int k=0; k<input1_w; k++){
//                 outbuffer[i][j] += buffer1[i][k] * buffer2[k][j];
//             }
//         }
//     }

//     for (int j=0; j<input2_w; j++){
//         #pragma HLS unroll
//         #pragma HLS PIPELINE
//         for (int i=0; i<input1_h; i++){
//             output[input2_w * i + j] = outbuffer[i][j];
//         }
//     }
// }
// }

// void vdot (const unsigned int *in1, const unsigned int *in2,
// 		   unsigned int *out, int size)