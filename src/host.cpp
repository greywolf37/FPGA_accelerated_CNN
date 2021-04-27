// #include<time.h>
#include <chrono>
#include <stdio.h>
#include "host.hpp"
#define BLOCK_MATRIX_SIZE 16
#define DEBUG 0

using namespace std;

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
		return EXIT_FAILURE;
	}

    // std::vector<int> MATRIX_SIZES{16,64,256,1024};
    std::vector<int> MATRIX_SIZES{16};

    for(unsigned int x=0; x<MATRIX_SIZES.size(); x++){
    int MATRIX_SIZE = MATRIX_SIZES[x];
    int TEST_SIZE = MATRIX_SIZE;
    std::string binaryFile = argv[1];
    size_t vector_size_bytes = sizeof(float) * BLOCK_MATRIX_SIZE * BLOCK_MATRIX_SIZE;
    cl_int err;
    unsigned fileBufSize;
    // std::vector<int,aligned_allocator<int>> source_in1(DATA_SIZE);

    // }
// , aligned_allocator<std::vector>1
    // Allocate Memory in Host Memory
    std::vector<float, aligned_allocator<float>> source_in1(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<float, aligned_allocator<float>> source_in2(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<float, aligned_allocator<float>> source_hw_results(MATRIX_SIZE * MATRIX_SIZE, 0);
    std::vector<float, aligned_allocator<float>> source_sw_results(MATRIX_SIZE * MATRIX_SIZE, 0);

    // float source_in1[MATRIX_SIZE * MATRIX_SIZE];
    // float source_in2[MATRIX_SIZE * MATRIX_SIZE];
    // float source_hw_results[MATRIX_SIZE * MATRIX_SIZE] = { 0 };
    // float source_sw_results[MATRIX_SIZE * MATRIX_SIZE] = { 0 };
    // Create Random inputs
    generate_square_matrix(MATRIX_SIZE, source_in1.data());
    generate_square_matrix(MATRIX_SIZE, source_in2.data());

    if (DEBUG){
    // Print Inputs data 
    std::cout << "IN 1: " << std::endl;
    print_matrix(source_in1.data(), MATRIX_SIZE);
    std::cout << "IN 2" << std::endl;
    print_matrix(source_in2.data(), MATRIX_SIZE);
    std::cout << "OUT 1" << std::endl;
    print_matrix(source_hw_results.data(), MATRIX_SIZE);
    std::cout << "OUT 2" << std::endl;
    print_matrix(source_sw_results.data(), MATRIX_SIZE);
    }
    // for(int i=0; i<TEST_SIZE; i++){
    //     for(int j=0; j<TEST_SIZE; j++){
    //         for(int k=0; k<MATRIX_SIZE; k++){
    //             source_sw_results[MATRIX_SIZE * i + j] += source_in1[MATRIX_SIZE * i + k] * source_in2[MATRIX_SIZE * k + j];
    //         }
    //     }
    // }

    matrix_multiplication(source_in1.data(), source_in2.data(), source_sw_results.data(), MATRIX_SIZE);

    std::vector<cl::Device> devices = get_devices("Xilinx");
    devices.resize(1);
    cl::Device device = devices[0];


    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE , &err));
    char* fileBuf = read_binary_file(binaryFile, fileBufSize);
    cl::Program::Binaries bins{{fileBuf, fileBufSize}};
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    int BLOCK_COLS = int (MATRIX_SIZE/BLOCK_MATRIX_SIZE);
    int BLOCK_ROWS = int (MATRIX_SIZE/BLOCK_MATRIX_SIZE);
    int BLOCK_KS = int (MATRIX_SIZE/BLOCK_MATRIX_SIZE);

    // std::cout << "BLOCK_COLS: " << BLOCK_COLS << std::endl;
    // std::cout << "BLOCK_ROWS: " << BLOCK_ROWS << std::endl;
    // std::cout << "BLOCK_KS: " << BLOCK_KS << std::endl;

    std::vector<float, aligned_allocator<float>> block_in1(BLOCK_MATRIX_SIZE * BLOCK_MATRIX_SIZE);
    std::vector<float, aligned_allocator<float>> block_in2(BLOCK_MATRIX_SIZE * BLOCK_MATRIX_SIZE);
    std::vector<float, aligned_allocator<float>> block_out(BLOCK_MATRIX_SIZE * BLOCK_MATRIX_SIZE, 0);
    
    // Stat Timer
    auto start = chrono::system_clock::now();

    // Block Multiplication loop
    for (int block_col=0; block_col<BLOCK_COLS; block_col++){
        for (int block_row=0; block_row<BLOCK_ROWS; block_row++){
            for (int block_k=0; block_k<BLOCK_KS; block_k++){


                // Read to Block 1 and 2
                for (int i=0; i<BLOCK_MATRIX_SIZE; i++){
                    for (int j=0; j<BLOCK_MATRIX_SIZE; j++){
                        // Block 1 (Constant row)
                        block_in1[i * BLOCK_MATRIX_SIZE + j] = source_in1[(i + (block_row*BLOCK_MATRIX_SIZE)) * MATRIX_SIZE + (j + (block_k*BLOCK_MATRIX_SIZE))];

                        // Block 2 (Constant col)
                        block_in2[i * BLOCK_MATRIX_SIZE + j] = source_in2[(i + (block_k*BLOCK_MATRIX_SIZE)) * MATRIX_SIZE + (j + (block_col*BLOCK_MATRIX_SIZE))];
                    }
                }

    
    
    

    // Read to Block 2
    OCL_CHECK(err, cl::Kernel krnl_vector_add(program,"vdot", &err));


    OCL_CHECK(err, cl::Buffer buffer_in1   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            vector_size_bytes, block_in1.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_in2   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            vector_size_bytes, block_in2.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_output(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
            vector_size_bytes, block_out.data(), &err));

    OCL_CHECK(err, err = krnl_vector_add.setArg(0, buffer_in1));
    OCL_CHECK(err, err = krnl_vector_add.setArg(1, buffer_in2));
    OCL_CHECK(err, err = krnl_vector_add.setArg(2, buffer_output));
    OCL_CHECK(err, err = krnl_vector_add.setArg(3, BLOCK_MATRIX_SIZE));

    // print_test(2);
    // print_test(3);
    // print_test(4);
	// print_test(5);

    std::vector<cl::Event> events;

    cl::Event write_event;
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1, buffer_in2},0/* 0 means from host*/, NULL, &write_event));	
    events.push_back(write_event);

    cl::Event kernel_event;
    OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add, NULL, &kernel_event));
    events.push_back(kernel_event);
    // q.flush();
    q.finish();

    cl::Event read_event;
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output},CL_MIGRATE_MEM_OBJECT_HOST, NULL, &read_event));

    // q.flush();
    q.finish();
    // sleep(5);

    // Write to output
    for (int i=0; i<BLOCK_MATRIX_SIZE; i++){
        for (int j=0; j<BLOCK_MATRIX_SIZE; j++){
            // Block Out
            source_hw_results[(i + (block_row*BLOCK_MATRIX_SIZE)) * MATRIX_SIZE + (j + (block_col*BLOCK_MATRIX_SIZE))] += block_out[i * BLOCK_MATRIX_SIZE + j];
        }
    }

    // std::cout << "Mutiplied col: " << block_col<< " with row: " << block_row << std::endl;
    // print_matrix(block_in1.data(), BLOCK_MATRIX_SIZE);
    // print_matrix(block_in2.data(), BLOCK_MATRIX_SIZE);

    }
    }
    }

    // End Timer
    auto end = chrono::system_clock::now();

    auto elapsed_time = chrono::duration_cast<chrono::nanoseconds>(end - start) / 1000000000.0;

    

// OPENCL HOST CODE AREA END 12
    // clock_t start = clock();

    // clock_t end = clock();
    // double elapsed_time =(double)(end - start);
    // double elapsed_time =(double)(end - start)/(double)(CLOCKS_PER_SEC);

    // Compare the results of the Device to the simulation
    bool match = true;
    if (DEBUG){
        for (int i = 0 ; i < TEST_SIZE ; i++){
            for (int j =0; j < TEST_SIZE; j++){
                if (int(source_hw_results[MATRIX_SIZE * i + j]) != int(source_sw_results[MATRIX_SIZE * i + j])){
                    std::cout << "Error: Result mismatch" << std::endl;
                    std::cout << "i = " << i << " CPU result = " << source_sw_results[MATRIX_SIZE * i + j]
                        << " Device result = " << source_hw_results[MATRIX_SIZE * i + j] << std::endl;
                    match = false;
                    // break;
                }
            }
        }
    
    

    std::cout << "OUT 1 (END)" << std::endl;
    print_matrix(source_hw_results.data(), MATRIX_SIZE);
    std::cout << "OUT 2 (END)" << std::endl;
    print_matrix(source_sw_results.data(), MATRIX_SIZE);
    // std::cout << "Matrix Size: " << MATRIX_SIZE << " Time Taken: " << (double)(end - start)/CLOCKS_PER_SEC << std::endl;
    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl; 
    }
    std::cout << "Matrix Size: " << MATRIX_SIZE << " Time Taken: " << elapsed_time.count() << std::endl;
    delete[] fileBuf;
}
    return (EXIT_SUCCESS);

}
