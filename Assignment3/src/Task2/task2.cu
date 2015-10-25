#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <sys/time.h>

#include "cpuFunc.cu.h"
#include "gpuFunc.cu.h"

#define COLS 64
#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 32


int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}



template<class T>
unsigned long int gpuNaiveTask2(T* A, T* B, int M, int N) {

    T* d_A = flatMatrixCudaMalloc<T>(M, N);
    T* d_B = flatMatrixCudaMalloc<T>(M, N);

    flatMatrixHostToDevice<T>(d_A, A, M, N);

    int threadsPerBlock = 512;
    int numBlocks = ceil(M/(float) threadsPerBlock);

    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    flatNaiveTask2Kernel<T> <<<numBlocks , threadsPerBlock>>>(d_A, d_B, M, N);
    cudaThreadSynchronize();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);

    flatMatrixDeviceToHost<T>(B, d_B, M, N);

    flatmatrixCudaFree<T>(d_A);
    flatmatrixCudaFree<T>(d_B);
    return (t_diff.tv_sec*1e6+t_diff.tv_usec);
}

template<class T>
unsigned long int gpuTransposedTask2(T* A, T* B, int M, int N) {

    T* d_A = flatMatrixCudaMalloc<T>(M, N);
    T* d_At = flatMatrixCudaMalloc<T>(N, M);
    T* d_B = flatMatrixCudaMalloc<T>(M, N);
    T* d_Bt = flatMatrixCudaMalloc<T>(N, M);

    flatMatrixHostToDevice<T>(d_A, A, M, N);

    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);


    dim3 threadsPerBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    int blocks_X = ceil(M/(float)threadsPerBlock.x);
    int blocks_Y = ceil(N/(float)threadsPerBlock.y);
    dim3 numBlocks(blocks_Y, blocks_X);
    flatSharedTransposeKernel<T> <<<numBlocks , threadsPerBlock>>>(d_A, d_At, M, N);
    cudaThreadSynchronize();

    int threadsPerBlock2 = 512;
    int numBlocks2 = ceil(M/(float) threadsPerBlock2);
    flatTransposedTask2Kernel<T> <<<numBlocks2 , threadsPerBlock2>>>(d_At, d_Bt, M, N);
    cudaThreadSynchronize();

    dim3 threadsPerBlock3(BLOCK_WIDTH, BLOCK_HEIGHT);
    blocks_X = ceil(N/(float)threadsPerBlock.x);
    blocks_Y = ceil(M/(float)threadsPerBlock.y);
    dim3 numBlocks3(blocks_Y, blocks_X);
    flatSharedTransposeKernel<T> <<<numBlocks3 , threadsPerBlock3>>>(d_Bt, d_B, N, M);
    cudaThreadSynchronize();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);

    flatMatrixDeviceToHost<T>(B, d_B, M, N);

    flatmatrixCudaFree<T>(d_A);
    flatmatrixCudaFree<T>(d_B);
    return (t_diff.tv_sec*1e6+t_diff.tv_usec);
}

template<class T>
unsigned long int cpuTask2(T* A, T* B, int M, int N) {

    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    flatMatrixTask2b(A, B, M, N);

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    return (t_diff.tv_sec*1e6+t_diff.tv_usec);
}


void allTask2Test(int iterations, int M, int N) {
    printf("\n");
    printf("===========================================\n");
    printf("=============== TASK 2 TESTS ==============\n");
    printf("===========================================\n");
    printf("Configuration:\n");
    printf(" - # of runs: %d\n", iterations);
    printf(" - Matrix dimensions: A[%d,%d].\n\n", M, N);

    unsigned long int cpu_time  = 0;
    unsigned long int gpu1_time = 0;
    unsigned long int gpu2_time = 0;
    int valid_count1 = 0;
    int valid_count2 = 0;
    int invalid_count1 = 0;
    int invalid_count2 = 0;

    float* cpu_A = flatMatrixCreate<float>(M,N, 1.337);
    float* gpu1_A = flatMatrixCreate<float>(M,N, 1.337);
    float* gpu2_A = flatMatrixCreate<float>(M,N, 1.337);

    for(int i = 0 ; i < iterations ; i++) {
        float* cpu_B = flatMatrixCreate<float>(M,N, -1.0);
        float* gpu1_B = flatMatrixCreate<float>(M,N, -1.0);
        float* gpu2_B = flatMatrixCreate<float>(M,N, -1.0);

        cpu_time += cpuTask2<float>(cpu_A, cpu_B, M, N);
        gpu1_time += gpuNaiveTask2<float>(gpu1_A, gpu1_B, M, N);
        gpu2_time += gpuTransposedTask2<float>(gpu2_A, gpu2_B, M, N);


        bool valid1 = flatMatrixCompare<float>(cpu_B, gpu1_B, M,N);
        bool valid2 = flatMatrixCompare<float>(cpu_B, gpu2_B, M,N);
        if(valid1) { valid_count1++; }
        else {
            invalid_count1++;
            printf("\nERROR - NAIVE COMPARE:\n");
            flatMatrixPrint(cpu_B, M, N);
            flatMatrixPrint(gpu1_B, M, N);
        }

        if(valid2) { valid_count2++; }
        else {
            invalid_count2++;
            printf("\nERROR - SHARED COMPARE:\n");
            flatMatrixPrint(cpu_B, M, N);
            flatMatrixPrint(gpu2_B, M, N);
        }

        flatMatrixFree<float>(cpu_B);
        flatMatrixFree<float>(gpu1_B);
        flatMatrixFree<float>(gpu2_B);
    }

    flatMatrixFree<float>(cpu_A);
    flatMatrixFree<float>(gpu1_A);
    flatMatrixFree<float>(gpu2_A);

    unsigned long int cpu_average = cpu_time / iterations; // average runtime in microseconds
    unsigned long int gpu1_average = gpu1_time / iterations;
    unsigned long int gpu2_average = gpu2_time / iterations;

    if (invalid_count1 < 1 && invalid_count2 < 1) {
        printf("All results are VALID.\n");
    } else {
        printf("INVALID naive results : %d\n", invalid_count1);
        printf("INVALID shared results: %d\n", invalid_count2);
    }

    printf("\n");
    printf("Average CPU runtime        : %d microseconds.\n", cpu_average);
    printf("Average GPU naive runtime  : %d microseconds.\n", gpu1_average);
    printf("Average GPU shared runtime : %d microseconds.\n", gpu2_average);
}

int main(int argc, char** argv) {
    allTask2Test(100, 10, COLS);
    allTask2Test(100, 20, COLS);
    allTask2Test(100, 30, COLS);
    allTask2Test(100, 40, COLS);
    allTask2Test(100, 50, COLS);
    allTask2Test(100, 60, COLS);
    allTask2Test(100, 70, COLS);
    allTask2Test(100, 80, COLS);
    allTask2Test(100, 90, COLS);
    allTask2Test(100, 100, COLS);
    allTask2Test(100, 2000, COLS);
    allTask2Test(100, 3000, COLS);
    allTask2Test(100, 4000, COLS);
    allTask2Test(100, 5000, COLS);
    //allTask2Test(100, 10000, COLS);

    return 0;
}
