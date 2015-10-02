#ifndef _GPU_FUNC
#define _GPU_FUNC

#include <cuda_runtime.h>

// Flat Matrix Offset
__global__ int fmo(int row, int col, int matrixWidth) {
    return row * matrixWidth + col;
}

template<class T>
__global__ void flatNaiveMutliplyKernel(T* A, T* B, T* C, int M, int N, int U) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if((i < M) && (j < N)) {
        float tmp = 0.0;
        for (int k = 0 ; k < U ; k++) { // Sequential loop
            tmp = tmp + A[fmo(i,k,U)] * B[fmo(k,j,N)];
        }
        C[fmo(i,j,U)] = tmp;
    }
}

template<class T>
__global__ void flatSharedMultiplyKernel(T* A, T* B, T* C, int M, int N, int U) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    __shared__ T s[1024];

    if((i < M) && (j < N)) {
        s[tid] = 0.0;
        __syncthreads();
        for (int k = 0 ; k < U ; k++) { // Sequential loop
            s[tid] = s[tid] + A[fmo(i,k,U)] * B[fmo(k,j,N)];
            __syncthreads();
        }
        C[fmo(i,j,U)] = s[tid];
    } else {
        __syncthreads();
        for (int k = 0 ; k < U ; k++) { // We still need to sync as many times
                                        // as the threads in the block that are
                                        // working.
            __syncthreads();
        }
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
