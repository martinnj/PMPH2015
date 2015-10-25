#ifndef _GPU_FUNC
#define _GPU_FUNC

#include <cuda_runtime.h>

// Flat Matrix Offset
__device__ int gpuFmo(int row, int col, int matrixWidth) {
    return row * matrixWidth + col;
}

template<class T>
__global__ void flatNaiveMutliplyKernel(T* A, T* B, T* C, int M, int N, int U) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if((i < M) && (j < N)) {
        float tmp = 0.0;
        for (int k = 0 ; k < U ; k++) { // Sequential loop
            tmp = tmp + A[gpuFmo(i,k,U)] * B[gpuFmo(k,j,N)];
        }
        C[gpuFmo(i,j,N)] = tmp;
    }
}

template<class T>
__global__ void flatSharedMultiplyKernel(T* A, T* B, T* C, int M, int N, int U) {

    const int tile_size = 32;
    __shared__ T Atile[tile_size][tile_size];
    __shared__ T Btile[tile_size][tile_size];

    int ii = blockIdx.y * blockDim.x;
    int jj = blockIdx.x * blockDim.y;

    int tidy = threadIdx.y;
    int i = tidy+ii;

    int tidx = threadIdx.x;
    int j = tidx+jj;

    float tmp = 0.0;

    for(int kk=0; kk<U; kk+=tile_size) {
        Atile[tidy][tidx] = (i<M && kk+tidx<U) ? A[gpuFmo(i,kk+tidx,U)] : 0.0 ;
        Btile[tidy][tidx] = (j<N && kk+tidy<U) ? B[gpuFmo(kk+tidy,j,N)] : 0.0 ;
        __syncthreads();

        for(int k=0; k<tile_size; k++) {
            tmp += Atile[tidy][k] * Btile[k][tidx];
        }
        __syncthreads();
    }

    if (i<M && j<N) {
        C[gpuFmo(i,j,N)] = tmp;
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
