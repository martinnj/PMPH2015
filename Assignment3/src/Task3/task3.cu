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

}

template<class T>
unsigned long int gpuNaiveMultiply(T* A, T* B, T* C, int M, int N, int U) {

}

template<class T>
unsigned long int gpuSharedMultiply(T* A, T* B, T* C, int M, int N, int U) {

}


int main(int argc, char** argv) {

    int M = 2;
    int N = 2;
    int U = 2;

    float* A = flatMatrixCreate<float>(M,U,0.0);
    float* B = flatMatrixCreate<float>(U,N,0.0);
    float* C = flatMatrixCreate<float>(M,N,-1.0);

    for(int i = 0 ; i < M ; i++) {
        for(int j = 0 ; j < N ; j++) {
            int offset = fmo(i,j,N);
            A[offset] = offset + 1.0;
            B[offset] = offset + 1.0;
        }
    }

    flatMatrixPrint<float>(A, M, U);
    flatMatrixPrint<float>(B, U, N);

    flatMatrixMultiply<float>(A,B,C,M,N,U);

    flatMatrixPrint<float>(C, M, N);

    flatMatrixFree<float>(A);
    flatMatrixFree<float>(B);
    flatMatrixFree<float>(C);

    return 0;
}
