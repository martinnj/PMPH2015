#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ScanHost.cu.h"

int gpuMss(int* h_start, unsigned long num_threads, unsigned int block_size) {

    unsigned int mem_start_size = num_threads * sizeof(int);
    unsigned int mem_mapped_size = num_threads * sizeof(MyInt4);
    unsigned int num_blocks = ( (num_threads % block_size) == 0) ?
                    num_threads / block_size     :
                    num_threads / block_size + 1 ;

    MyInt4* h_scanned = (MyInt4*) malloc(mem_mapped_size);


    int* d_start;
    cudaMalloc((void**)&d_start, mem_start_size);
    cudaMemcpy(d_start, h_start, mem_start_size, cudaMemcpyHostToDevice);


    MyInt4* d_mapped;
    cudaMalloc((void**)&d_mapped, mem_mapped_size);

    MssMapKernel<int, MyInt4><<<num_blocks, block_size >>> (d_start, d_mapped, num_threads);
    cudaThreadSynchronize();

    MyInt4* d_scanned;
    cudaMalloc((void**)&d_scanned, mem_mapped_size);
    scanInc<MsspOp, MyInt4> (block_size, num_threads, d_mapped, d_scanned);
    cudaMemcpy(h_scanned, d_scanned, mem_mapped_size, cudaMemcpyDeviceToHost);

    int res = h_scanned[num_threads-1].x;


    cudaFree(d_start);
    cudaFree(d_mapped);
    cudaFree(d_scanned);
    free(h_scanned);

    return res;
}

int cpuMss(int* input, unsigned long input_size) {
    unsigned int mem_int4_size = input_size * sizeof(MyInt4);

    MyInt4* mapped = (MyInt4*) malloc(mem_int4_size);
    MyInt4* scanned = (MyInt4*) malloc(mem_int4_size);

    for (unsigned int i = 0 ; i < input_size ; i++) {
        int x = input[i];
        if (x > 0) {
            mapped[i] = MyInt4(x,x,x,x);
        } else {
            mapped[i] = MyInt4(0,0,0,x);
        }
    }

    MyInt4 identity = CpuMsspOp::identity();
    MyInt4 first = mapped[0];
    scanned[0] = CpuMsspOp::apply(identity, first);

    for (unsigned int i = 1 ; i < input_size ; i++) {
        MyInt4 x = mapped[i];
        MyInt4 y = scanned[i-1];
        scanned[i] = CpuMsspOp::apply(y,x);
    }


    int res = scanned[input_size-1].x;
    free(mapped);
    free(scanned);
    return res;
}

int RandomNumber(int lower, int upper) {
    return lower + rand() % (upper - lower + 1);
}

void mssCompare(int* input, unsigned long input_size) {
    unsigned int block_size  = 512;
    unsigned long num_threads = input_size;

    unsigned long int elapsedGpu, elapsedCpu;

    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);
    int gpuRes = gpuMss(input, num_threads, block_size);
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsedGpu = (t_diff.tv_sec*1e6+t_diff.tv_usec);


    gettimeofday(&t_start, NULL);
    int cpuRes = cpuMss(input, num_threads);
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsedCpu = (t_diff.tv_sec*1e6+t_diff.tv_usec);

    if(gpuRes == cpuRes) {
        printf("Results are VALID, execution times:\n");
    } else {
        printf("Results are INVALID (CPU: %d, GPU: %d) , execution times:\n",cpuRes, gpuRes);
        /*printf("Input: ");
        for (int i = 0 ; i < input_size ; i++) {
            printf("%d, ", input[i]);
        }
        printf("\n");*/
    }

    printf(" - GPU runs in: %lu microsecs\n", elapsedGpu);
    printf(" - CPU runs in: %lu microsecs\n", elapsedCpu);

    return;
}

void mssTest(int iterations, int list_length) {
    printf("==========================================\n");
    printf("================ MSS TEST ================\n");
    printf("==========================================\n");
    printf("Configuration:\n");
    printf(" - # of runs: %d\n", iterations);
    printf(" - list length: %d\n", list_length);

    srand (time(NULL)); // Seed the damn generator.

    int* input = (int*) malloc(list_length * sizeof(int));

    for(int i = 0 ; i < iterations ; i++) {

        for(int j = 0 ; j < list_length ; j++) {
            input[j] = RandomNumber(-1,100);
            //printf("Generated: %d\n", input[j]);
        }
        mssCompare(input, list_length);
    }

    free(input);
}



int main(int argc, char** argv) {
    mssTest(3, 100);
    mssTest(3, 1000);
    mssTest(3, 10000);
    mssTest(3, 100000);
    mssTest(3, 1000000);
}
