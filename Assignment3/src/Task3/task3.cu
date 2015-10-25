#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <sys/time.h>

#include "cpuFunc.cu.h"
#include "gpuFunc.cu.h"

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
unsigned long int cpuMultiply(T* A, T* B, T* C, int M, int N, int U) {
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    flatMatrixMultiply<T>(A, B, C, M, N, U);

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    return (t_diff.tv_sec*1e6+t_diff.tv_usec);
}

template<class T>
unsigned long int gpuNaiveMultiply(T* A, T* B, T* C, int M, int N, int U) {

    T* d_A = flatMatrixCudaMalloc<T>(M, U);
    T* d_B = flatMatrixCudaMalloc<T>(U, N);
    T* d_C = flatMatrixCudaMalloc<T>(M, N);

    flatMatrixHostToDevice<T>(d_A, A, M, U);
    flatMatrixHostToDevice<T>(d_B, B, U, N);

    dim3 threadsPerBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    int blocks_X = ceil(M/(float)threadsPerBlock.x);
    int blocks_Y = ceil(N/(float)threadsPerBlock.y);
    dim3 numBlocks(blocks_Y, blocks_X);

    //printf("numBlocks(%d, %d);\n",blocks_X, blocks_Y);


    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);
    flatNaiveMutliplyKernel<T> <<<numBlocks , threadsPerBlock>>>(d_A, d_B, d_C, M, N, U);
    cudaThreadSynchronize();
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);

    flatMatrixDeviceToHost<T>(C, d_C, M, N);

    flatmatrixCudaFree<T>(d_A);
    flatmatrixCudaFree<T>(d_B);
    flatmatrixCudaFree<T>(d_C);

    return (t_diff.tv_sec*1e6+t_diff.tv_usec);
}

template<class T>
unsigned long int gpuSharedMultiply(T* A, T* B, T* C, int M, int N, int U) {

    T* d_A = flatMatrixCudaMalloc<T>(M, U);
    T* d_B = flatMatrixCudaMalloc<T>(U, N);
    T* d_C = flatMatrixCudaMalloc<T>(M, N);

    flatMatrixHostToDevice<T>(d_A, A, M, U);
    flatMatrixHostToDevice<T>(d_B, B, U, N);

    dim3 threadsPerBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    int blocks_X = ceil(M/(float)threadsPerBlock.x);
    int blocks_Y = ceil(N/(float)threadsPerBlock.y);
    dim3 numBlocks(blocks_Y, blocks_X);

    //printf("numBlocks(%d, %d);\n",blocks_X, blocks_Y);


    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);
    flatSharedMultiplyKernel<T> <<<numBlocks , threadsPerBlock>>>(d_A, d_B, d_C, M, N, U);
    cudaThreadSynchronize();
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);

    flatMatrixDeviceToHost<T>(C, d_C, M, N);

    flatmatrixCudaFree<T>(d_A);
    flatmatrixCudaFree<T>(d_B);
    flatmatrixCudaFree<T>(d_C);

    return (t_diff.tv_sec*1e6+t_diff.tv_usec);
}

void allMultiplyTest(int iterations, int M, int N, int U) {
    printf("\n");
    printf("==========================================\n");
    printf("======= MATRIX MULTIPLICATION TEST =======\n");
    printf("==========================================\n");
    printf("Configuration:\n");
    printf(" - # of runs: %d\n", iterations);
    printf(" - Matrix dimensions: A[%d,%d], B[%d,%d], C[%d,%d].\n\n", M, U, U, N, M, N);

    unsigned long int cpu_time  = 0;
    unsigned long int gpu1_time = 0;
    unsigned long int gpu2_time = 0;
    int valid_count1 = 0;
    int invalid_count1 = 0;
    int valid_count2 = 0;
    int invalid_count2 = 0;

    float* cpu_A = flatMatrixCreate<float>(M,U, 0.0);
    float* gpu1_A = flatMatrixCreate<float>(M,U, 0.0);
    float* gpu2_A = flatMatrixCreate<float>(M,U, 0.0);

    float* cpu_B = flatMatrixCreate<float>(U,N, 0.0);
    float* gpu1_B = flatMatrixCreate<float>(U,N, 0.0);
    float* gpu2_B = flatMatrixCreate<float>(U,N, 0.0);

    for(int i = 0 ; i < M*U ; i++) {
        cpu_A[i] = i + 1.0;
        gpu1_A[i] = i + 1.0;
        gpu2_A[i] = i + 1.0;
    }

    for(int i = 0 ; i < U*N ; i++) {
        cpu_B[i] = i + 1.0;
        gpu1_B[i] = i + 1.0;
        gpu2_B[i] = i + 1.0;
    }

    for(int i = 0 ; i < iterations ; i++) {
        float* cpu_C = flatMatrixCreate<float>(M,N, -1.0);
        float* gpu1_C = flatMatrixCreate<float>(M,N, -1.0);
        float* gpu2_C = flatMatrixCreate<float>(M,N, -1.0);

        cpu_time += cpuMultiply<float>(cpu_A, cpu_B, cpu_C, M, N, U);
        gpu1_time += gpuNaiveMultiply<float>(gpu1_A, gpu1_B, gpu1_C, M, N, U);
        gpu2_time += gpuSharedMultiply<float>(gpu2_A, gpu2_B, gpu2_C, M, N, U);

        bool valid1 = flatMatrixCompare<float>(cpu_C, gpu1_C, M, N);
        bool valid2 = flatMatrixCompare<float>(cpu_C, gpu2_C, M, N);
        if(valid1) { valid_count1++; }
        else {
            invalid_count1++;
            printf("\nERROR - NAIVE COMPARE:\n");
            flatMatrixPrint(cpu_C, M, N);
            flatMatrixPrint(gpu1_C, M, N);
        }


        if(valid2) { valid_count2++; }
        else {
            invalid_count2++;
            printf("\nERROR - SHARED COMPARE:\n");
            flatMatrixPrint(cpu_C, M, N);
            flatMatrixPrint(gpu2_C, M, N);
        }

        flatMatrixFree<float>(cpu_C);
        flatMatrixFree<float>(gpu1_C);
        flatMatrixFree<float>(gpu2_C);
    }

    flatMatrixFree<float>(cpu_A);
    flatMatrixFree<float>(cpu_B);
    flatMatrixFree<float>(gpu1_A);
    flatMatrixFree<float>(gpu1_B);
    flatMatrixFree<float>(gpu2_A);
    flatMatrixFree<float>(gpu2_B);

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


    allMultiplyTest(10, 2, 3, 4);
    allMultiplyTest(10, 40, 8, 10);

    allMultiplyTest(10, 10, 10, 10);
    allMultiplyTest(10, 20, 20, 20);
    allMultiplyTest(10, 30, 30, 30);
    allMultiplyTest(10, 40, 40, 40);
    allMultiplyTest(10, 50, 50, 50);
    allMultiplyTest(10, 60, 60, 60);

    return 0;
}
