#ifndef _GPU_FUNC
#define _GPU_FUNC

#include <cuda_runtime.h>

// Flat Matrix Offset
__device__ int gpuFmo(int row, int col, int matrixWidth) {
    return row * matrixWidth + col;
}

template<class T>
__global__ void flatNaiveTask2Kernel(T* A, T* B, int M, int N) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) { return; }

    T tmp = A[gpuFmo(i, 0, N)];
    T accum = tmp * tmp;
    B[gpuFmo(i, 0, N)] = accum;

    for (int j = 1 ; j < N ; j++) {
        T tmpA = A[gpuFmo(i, j, N)];
        accum = sqrt(accum) + tmpA * tmpA;
        B[gpuFmo(i, j, N)] = accum;
    }
}

template<class T>
__global__ void flatTransposedTask2Kernel(T* At, T* Bt, int M, int N) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) { return; }

    T tmp = At[gpuFmo(0, i, M)];
    T accum = tmp * tmp;
    __syncthreads();
    Bt[gpuFmo(0, i, M)] = accum;

    for (int j = 1 ; j < N ; j++) {
        __syncthreads();
        T tmpA = At[gpuFmo(j, i, M)];
        accum = sqrt(accum) + tmpA * tmpA;

        __syncthreads();
        Bt[gpuFmo(j, i, M)] = accum;
    }
}

template<class T>
__global__ void flatNaiveTransposeKernel(T* A, T* B, int M, int N) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if((m < M) && (n < N))
        B[n*M+m] = A[m*N+n];
}

template<class T>
__global__ void flatSharedTransposeKernel(T* A, T* B, int M, int N) {
    const int tile_size = 32;
    __shared__ T tile[tile_size][tile_size];
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int j    = blockIdx.x*tile_size + tidx;
    int i    = blockIdx.y*tile_size + tidy;

    if( j < N && i < M )
        tile[tidy][tidx] = A[i*N+j];
    __syncthreads();

    i = blockIdx.y*tile_size + threadIdx.x;
    j = blockIdx.x*tile_size + threadIdx.y;
    if( j < N && i < M )
        B[j*M+i] = tile[tidx][tidy];
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
