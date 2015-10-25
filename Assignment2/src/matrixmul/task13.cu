#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <sys/time.h>

#include "cpuFunc.cu.h"
#include "gpuFunc.cu.h"
#include "ScanHost.cu.h"

using namespace std;

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

unsigned long int gpuFlatMatVecMult(int* flags, int* idxs, float* matIn, int matLen, float* vIn, int vecLen, float* vOut) {

    unsigned int block_size  = 1024;
    unsigned int num_blocks  = ceil(((matLen + (block_size -1)) / block_size));


    // Allocate device memory
    int intMatLen = sizeof(int)*matLen;
    int floatMatLen = sizeof(float)*matLen;
    int floatVecLen = sizeof(float)*vecLen;

    int* d_flags;
    int* d_idxs;
    float* d_matIn;
    float* d_vIn;
    float* d_vOut;
    float* d_prods;
    float* d_sums;
    int* d_end_flags;
    int* d_segms;

    cudaMalloc((void**) &d_flags, intMatLen);
    cudaMalloc((void**) &d_idxs, intMatLen);
    cudaMalloc((void**) &d_matIn, floatMatLen);
    cudaMalloc((void**) &d_vIn, floatVecLen);
    cudaMalloc((void**) &d_vOut, floatVecLen);
    cudaMalloc((void**) &d_prods, floatMatLen);
    cudaMalloc((void**) &d_sums, floatMatLen);
    cudaMalloc((void**) &d_end_flags, intMatLen);
    cudaMalloc((void**) &d_segms, intMatLen);

    // Copy data to device memory.
    cudaMemcpy(d_flags, flags, intMatLen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_idxs, idxs, intMatLen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matIn, matIn, floatMatLen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vIn, vIn, floatVecLen, cudaMemcpyHostToDevice);


    //prods = map (\(a,b) -> b*(x!!a)) mat
    prodsMapKernel<<<num_blocks, block_size>>>(d_idxs, d_matIn, matLen, d_vIn, d_prods);
    cudaThreadSynchronize();
    //float* h_prods = (float*) malloc(floatMatLen);
    //float* h_matIn = (float*) malloc(floatMatLen);
    //cudaMemcpy(h_prods, d_prods, floatMatLen, cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_matIn, d_matIn, floatMatLen, cudaMemcpyDeviceToHost);
    //vecFPrint(h_matIn, matLen);
    //vecFPrint(h_prods, matLen);

    //sums = segmScanInc (+) 0 flags prods
    sgmScanInc<Add<float>, float>(block_size, matLen, d_prods, d_flags, d_sums);
    cudaThreadSynchronize();
    //float* h_sums = (float*) malloc(floatMatLen);
    //cudaMemcpy(h_sums, d_sums, floatMatLen, cudaMemcpyDeviceToHost);
    //vecFPrint(h_sums, matLen);

    //end_flags = tail flags ++ [head flags] -- flags now marks ends of segments.
    flagsToEndFlagsKernel<<<num_blocks, block_size>>>(d_flags, d_end_flags, matLen);
    cudaThreadSynchronize();
    //int* end_flags = (int*) malloc(intMatLen);
    //cudaMemcpy(end_flags, d_end_flags, intMatLen, cudaMemcpyDeviceToHost);
    //vecIPrint(end_flags, matLen);

    //foo = zip end_flags sums
    //(vals, ff) = parFilter (\(a,b) -> a == 1) foo
    //(_, res) = unzip $ take (head ff) vals
    // From here the code is not the same as in haskell.

    // Create a list of what vector location each element is part of.
    scanInc<Add<int>,int>(block_size, matLen, d_end_flags, d_segms);
    cudaThreadSynchronize();
    //int* h_segms = (int*) malloc(intMatLen);
    //cudaMemcpy(h_segms, d_segms, intMatLen, cudaMemcpyDeviceToHost);
    //vecIPrint(h_segms, matLen);

    fooKernel<<<num_blocks, block_size>>>(d_sums, d_end_flags, matLen, d_segms, d_vOut);
    cudaThreadSynchronize();

    // Copy the result from device to host.
    cudaMemcpy(vOut, d_vOut, floatVecLen, cudaMemcpyDeviceToHost);

    // Free the device memory.
    cudaFree(d_flags);
    cudaFree(d_idxs);
    cudaFree(d_matIn);
    cudaFree(d_vIn);
    cudaFree(d_vOut);
    cudaFree(d_prods);
    cudaFree(d_sums);
    cudaFree(d_end_flags);
    cudaFree(d_segms);

    return 0;
}

int main(int argc, char** argv) {
    /**
    matrix_flag = [1,       0,        1,        0,        0,        1,        0,        0,        1,         0      ]
    matrix_flat = [(0,2.0), (1,-1.0), (0,-1.0), (1, 2.0), (2,-1.0), (1,-1.0), (2, 2.0), (3,-1.0), (2,-1.0), (3, 2.0)]
    x_vector    = [2.0, 1.0, 0.0, 3.0]
    */
    int flags[] = {1,0,1,0,0,1,0,0,1,0};
    int cols[]  = {0,1,0,1,2,1,2,3,2,3};
    float mat[] = {2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0};
    float vec[] = {2.0, 1.0, 0.0, 3.0};
    float out[] = {-10.0, -10.0, -10.0, -10.0};

    gpuFlatMatVecMult(flags, cols, mat, 10, vec, 4, out);

    // Should be: [ 3.0, 0.0, -4.0, 6.0 ]
    printf("[ %.1f, %.1f, %.1f, %.1f ]\n", out[0], out[1], out[2], out[3]);


    return 0;
}
