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
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    T* d_A = flatMatrixCudaMalloc<T>(M, N);
    T* d_B = flatMatrixCudaMalloc<T>(N, M);

    flatMatrixHostToDevice<T>(d_A, A, M, N);

    dim3 threadsPerBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    int blocks_X = ceil(M/(float)threadsPerBlock.x);
    int blocks_Y = ceil(N/(float)threadsPerBlock.y);
    dim3 numBlocks(blocks_X, blocks_Y);

    //printf("numBlocks(%d, %d);\n",blocks_X, blocks_Y);


    flatSharedTransposeKernel<T> <<<numBlocks , threadsPerBlock>>>(d_A, d_B, M, N);
    flatMatrixDeviceToHost<T>(B, d_B, M, N);

    flatmatrixCudaFree<T>(d_A);
    flatmatrixCudaFree<T>(d_B);

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    return (t_diff.tv_sec*1e6+t_diff.tv_usec);
}

void sharedTransposeTest(int iterations, int M, int N) {
    printf("\n");
    printf("==========================================\n");
    printf("========== SHARED TRANSPOSE TEST =========\n");
    printf("==========================================\n");
    printf("Configuration:\n");
    printf(" - # of runs: %d\n", iterations);
    printf(" - Matrix dimensions: [%d,%d]\n\n", M, N);

    unsigned long int cpu_time;
    unsigned long int gpu_time;
    int valid_count = 0;
    int invalid_count = 0;

    float* cpu_A = flatMatrixCreate<float>(M,N, 0.0);
    float* gpu_A = flatMatrixCreate<float>(M,N, 0.0);
    for(int i = 0 ; i < M ; i++) {
        for(int j = 0 ; j < N ; j++) {
            cpu_A[i*N+j] = i*N+j + 1.0;
            gpu_A[i*N+j] = i*N+j + 1.0;
        }
    }

    for(int i = 0 ; i < iterations ; i++) {
        float* cpu_B = flatMatrixCreate<float>(N,M, -1.0);
        float* gpu_B = flatMatrixCreate<float>(N,M, -1.0);

        //flatMatrixPrint(cpu_A, M ,N);
        //flatMatrixPrint(gpu_A, M ,N);

        cpu_time += cpuTranspose<float>(cpu_A, cpu_B, M, N);
        gpu_time += gpuSharedTranspose<float>(gpu_A, gpu_B, M, N);

        bool valid = flatMatrixCompare<float>(cpu_B, gpu_B, M, N);
        if(valid) { valid_count++; }
        else {
            invalid_count++;
            //flatMatrixPrint(cpu_B, M, N);
            //flatMatrixPrint(gpu_B, M, N);
        }

        flatMatrixFree<float>(cpu_B);
        flatMatrixFree<float>(gpu_B);
    }

    flatMatrixFree<float>(cpu_A);
    flatMatrixFree<float>(gpu_A);

    unsigned long int cpu_average = cpu_time / iterations; // average runtime in microseconds
    unsigned long int gpu_average = gpu_time / iterations;

    if(invalid_count > 0) {
        printf("INVALID results where found. # of invalid results: %d\n", invalid_count);
    } else {
        printf("All results were VALID.\n");
    }
    printf("\n");
    printf("Average CPU runtime: %.2f ms\n", cpu_average/1000.0);
    printf("Average GPU runtime: %.2f ms\n", gpu_average/1000.0);
}

template<class T>
unsigned long int gpuNaiveTranspose(T* A, T* B, int M, int N) {
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    T* d_A = flatMatrixCudaMalloc<T>(M, N);
    T* d_B = flatMatrixCudaMalloc<T>(M, N);

    flatMatrixHostToDevice<T>(d_A, A, M, N);

    dim3 threadsPerBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    int blocks_X = ceil(M/(float)threadsPerBlock.x);
    int blocks_Y = ceil(N/(float)threadsPerBlock.y);
    dim3 numBlocks(blocks_X, blocks_Y);

    //printf("numBlocks(%d, %d);\n",blocks_X, blocks_Y);


    flatNaiveTransposeKernel<T> <<<numBlocks , threadsPerBlock>>>(d_A, d_B, M, N);
    flatMatrixDeviceToHost<T>(B, d_B, M, N);

    flatmatrixCudaFree<T>(d_A);
    flatmatrixCudaFree<T>(d_B);

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    return (t_diff.tv_sec*1e6+t_diff.tv_usec);
}


void naiveTransposeTest(int iterations, int M, int N) {
    printf("\n");
    printf("==========================================\n");
    printf("========== NAIVE TRANSPOSE TEST ==========\n");
    printf("==========================================\n");
    printf("Configuration:\n");
    printf(" - # of runs: %d\n", iterations);
    printf(" - Matrix dimensions: [%d,%d]\n\n", M, N);

    unsigned long int cpu_time;
    unsigned long int gpu_time;
    int valid_count = 0;
    int invalid_count = 0;

    float* cpu_A = flatMatrixCreate<float>(M,N, 0.0);
    float* gpu_A = flatMatrixCreate<float>(M,N, 0.0);
    for(int i = 0 ; i < M ; i++) {
        for(int j = 0 ; j < N ; j++) {
            cpu_A[i*N+j] = i*N+j + 1.0;
            gpu_A[i*N+j] = i*N+j + 1.0;
        }
    }

    for(int i = 0 ; i < iterations ; i++) {
        float* cpu_B = flatMatrixCreate<float>(N,M, -1.0);
        float* gpu_B = flatMatrixCreate<float>(N,M, -1.0);

        //flatMatrixPrint(cpu_A, M ,N);
        //flatMatrixPrint(gpu_A, M ,N);

        cpu_time += cpuTranspose<float>(cpu_A, cpu_B, M, N);
        gpu_time += gpuNaiveTranspose<float>(gpu_A, gpu_B, M, N);

        bool valid = flatMatrixCompare<float>(cpu_B, gpu_B, M, N);
        if(valid) { valid_count++; }
        else {
            invalid_count++;
            //flatMatrixPrint(cpu_B, M, N);
            //flatMatrixPrint(gpu_B, M, N);
        }

        flatMatrixFree<float>(cpu_B);
        flatMatrixFree<float>(gpu_B);
    }

    flatMatrixFree<float>(cpu_A);
    flatMatrixFree<float>(gpu_A);

    unsigned long int cpu_average = cpu_time / iterations; // average runtime in microseconds
    unsigned long int gpu_average = gpu_time / iterations;

    if(invalid_count > 0) {
        printf("INVALID results where found. # of invalid results: %d\n", invalid_count);
    } else {
        printf("All results were VALID.\n");
    }
    printf("\n");
    printf("Average CPU runtime: %.2f ms\n", cpu_average/1000.0);
    printf("Average GPU runtime: %.2f ms\n", gpu_average/1000.0);
}

int main(int argc, char** argv) {

    // Shows where naive GPU overtakes CPU.
    // naiveTransposeTest(10, 2500, 2500);
    // naiveTransposeTest(10, 2600, 2600);
    // naiveTransposeTest(10, 2700, 2700);
    // naiveTransposeTest(10, 3000, 3000);

    // Shows where shared-memory GPU overtakes CPU.
    // sharedTransposeTest(10, 2500, 2500);
    // sharedTransposeTest(10, 2600, 2600);
    // sharedTransposeTest(10, 2700, 2700);
    // sharedTransposeTest(10, 3000, 3000);

    // Odd matrix sizes.
    naiveTransposeTest(10, 100, 4000);
    sharedTransposeTest(10, 100, 4000);

    return 0;
}
