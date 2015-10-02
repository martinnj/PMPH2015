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
unsigned long int cpuTranspose(T* A, T* B, int M, int N) {

    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    flatMatrixTranspose<T>(A, B, M, N);

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    return (t_diff.tv_sec*1e6+t_diff.tv_usec);
}

template<class T>
unsigned long int gpuSharedTranspose(T* A, T* B, int M, int N) {

    T* d_A = flatMatrixCudaMalloc<T>(M, N);
    T* d_B = flatMatrixCudaMalloc<T>(N, M);

    flatMatrixHostToDevice<T>(d_A, A, M, N);

    dim3 threadsPerBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    int blocks_X = ceil(M/(float)threadsPerBlock.x);
    int blocks_Y = ceil(N/(float)threadsPerBlock.y);
    dim3 numBlocks(blocks_X, blocks_Y);

    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    flatSharedTransposeKernel<T> <<<numBlocks , threadsPerBlock>>>(d_A, d_B, M, N);

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);

    flatMatrixDeviceToHost<T>(B, d_B, M, N);

    flatmatrixCudaFree<T>(d_A);
    flatmatrixCudaFree<T>(d_B);
    return (t_diff.tv_sec*1e6+t_diff.tv_usec);
}

template<class T>
unsigned long int gpuNaiveTranspose(T* A, T* B, int M, int N) {

    T* d_A = flatMatrixCudaMalloc<T>(M, N);
    T* d_B = flatMatrixCudaMalloc<T>(M, N);

    flatMatrixHostToDevice<T>(d_A, A, M, N);

    dim3 threadsPerBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    int blocks_X = ceil(M/(float)threadsPerBlock.x);
    int blocks_Y = ceil(N/(float)threadsPerBlock.y);
    dim3 numBlocks(blocks_X, blocks_Y);

    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    flatNaiveTransposeKernel<T> <<<numBlocks , threadsPerBlock>>>(d_A, d_B, M, N);

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);

    flatMatrixDeviceToHost<T>(B, d_B, M, N);

    flatmatrixCudaFree<T>(d_A);
    flatmatrixCudaFree<T>(d_B);
    return (t_diff.tv_sec*1e6+t_diff.tv_usec);
}


void allTransposeTest(int iterations, int M, int N) {
    printf("\n");
    printf("==========================================\n");
    printf("========== MATRIX TRANSPOSE TEST =========\n");
    printf("==========================================\n");
    printf("Configuration:\n");
    printf(" - # of runs: %d\n", iterations);
    printf(" - Matrix dimensions: A[%d,%d].\n\n", M, N);

    unsigned long int cpu_time  = 0;
    unsigned long int gpu1_time = 0;
    unsigned long int gpu2_time = 0;
    int valid_count1 = 0;
    int invalid_count1 = 0;
    int valid_count2 = 0;
    int invalid_count2 = 0;

    float* cpu_A = flatMatrixCreate<float>(M,N, 0.0);
    float* gpu1_A = flatMatrixCreate<float>(M,N, 0.0);
    float* gpu2_A = flatMatrixCreate<float>(M,N, 0.0);

    for(int i = 0 ; i < M*N ; i++) {
        cpu_A[i] = i + 1.0;
        gpu1_A[i] = i + 1.0;
        gpu2_A[i] = i + 1.0;
    }

    for(int i = 0 ; i < iterations ; i++) {
        float* cpu_B = flatMatrixCreate<float>(N,M, -1.0);
        float* gpu1_B = flatMatrixCreate<float>(N,M, -1.0);
        float* gpu2_B = flatMatrixCreate<float>(N,M, -1.0);

        cpu_time += cpuTranspose<float>(cpu_A, cpu_B, M, N);
        gpu1_time += gpuNaiveTranspose<float>(gpu1_A, gpu1_B, M, N);
        gpu2_time += gpuSharedTranspose<float>(gpu2_A, gpu2_B, M, N);

        bool valid1 = flatMatrixCompare<float>(cpu_B, gpu1_B, N,M);
        bool valid2 = flatMatrixCompare<float>(cpu_B, gpu2_B, N,M);
        if(valid1) { valid_count1++; }
        else {
            invalid_count1++;
            printf("\nERROR - NAIVE COMPARE:\n");
            flatMatrixPrint(cpu_B, N, M);
            flatMatrixPrint(gpu1_B, N, M);
        }


        if(valid2) { valid_count2++; }
        else {
            invalid_count2++;
            printf("\nERROR - SHARED COMPARE:\n");
            flatMatrixPrint(cpu_B, N, M);
            flatMatrixPrint(gpu2_B, N, M);
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

    allTransposeTest(1000, 10, 10);
    allTransposeTest(1000, 50, 50);
    allTransposeTest(1000, 100, 100);
    allTransposeTest(1000, 250, 250);

    return 0;
}
