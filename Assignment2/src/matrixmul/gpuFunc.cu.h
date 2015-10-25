#ifndef _GPU_FUNC
#define _GPU_FUNC

__global__ void prodsMapKernel(int* idxs, float* matIn, int matLen, float* vIn, float* prodOut) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < matLen) {
        int i = idxs[tid];
        float v = vIn[i];
        float m = matIn[tid];
        prodOut[tid] = m * v;
    }
}

__global__ void flagsToEndFlagsKernel(int* flags_in, int* flags_out, int matLen) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0 && tid < matLen) {
        flags_out[tid-1] = flags_in[tid];
    }
    if (tid == 0) {
        flags_out[matLen-1] = flags_in[0];
    }
}

__global__ void fooKernel(float* sums, int* endFlags, int matLen, int* segms, float* vOut) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < matLen) {
        if(endFlags[tid] == 1) {
            vOut[segms[tid]-1] = sums[tid];
        }
    }
}

#endif // _GPU_FUNC
