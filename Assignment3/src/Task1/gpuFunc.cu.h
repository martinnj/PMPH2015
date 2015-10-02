#ifndef _GPU_FUNC
#define _GPU_FUNC

#include <cuda_runtime.h>

template<class T>
__global__ void flatNaiveTransposeKernel(T* A, T* B, int M, int N) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if((m < M) && (n < N))
        B[n*M+m] = A[m*N+n];
}

template<class T>
__global__ void flatSharedTransposeKernel(T* A, T* B, int M, int N) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    __shared__ T s[1024]; // The blocks are 32x32 in size, so we ned enough memory for each of them to store something.

    if((m < M) && (n < N)) {
        s[tid] = A[m*N+n];
        __syncthreads();
        B[n*M+m] = s[tid];
    } else {
        // We need to sync up to the working threads or they might stall forever.
        // Even if we do not want to do any work.
        __syncthreads();
    }
}

template<class T>
T* flatMatrixCudaMalloc(int M, int N) {
    T* A;
    cudaMalloc((void**)&A, sizeof(T)*(M*N));
    return A;
}

template<class T>
void flatmatrixCudaFree(T* A) {
    cudaFree((void**)&A);
}

template<class T>
void flatMatrixHostToDevice(T* deviceA, T* hostA, int M, int N) {
    cudaMemcpy(deviceA, hostA, sizeof(T)*(M*N), cudaMemcpyHostToDevice);
}

template<class T>
void flatMatrixDeviceToHost(T* hostA, T* deviceA, int M, int N) {
    cudaMemcpy(hostA, deviceA, sizeof(T)*(M*N), cudaMemcpyDeviceToHost);
}

#endif
