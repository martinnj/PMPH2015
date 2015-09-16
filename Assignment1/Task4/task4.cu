#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits>

#include <sys/time.h>
#include <time.h>

#include <cuda_runtime.h>

using namespace std;

__global__ void assignmentKernel(float* d_in, float* d_out, int threads_num) {
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < threads_num) {
    float x = d_in[tid];
    d_out[tid] = pow((x/(x-2.3)),3.0);
  }
}

int timeval_subtract( struct timeval* result, struct timeval* t2,struct timeval* t1) {
  unsigned int resolution=1000000;
  long int diff = (t2->tv_usec + resolution * t2->tv_sec) -
    (t1->tv_usec + resolution * t1->tv_sec) ;
  result->tv_sec = diff / resolution;
  result->tv_usec = diff % resolution;
  return (diff<0);
}

unsigned long int cudaCalculation(unsigned int num_threads, unsigned int mem_size, float* h_in, float* h_out) {
  unsigned int block_size  = 1024;
  unsigned int num_blocks  = ceil(((num_threads + (block_size -1)) / block_size));

  // Allocate device memory
  float* d_in;
  cudaMalloc((void**) &d_in,  mem_size);
  float* d_out;
  cudaMalloc((void**) &d_out, mem_size);

  // Copy host memory to device
  cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);

  // Start timing the CUDA run.
  unsigned long int elapsed;
  struct timeval t_start, t_end, t_diff;
  gettimeofday(&t_start, NULL);

  // Execute the kernel
  assignmentKernel<<<num_blocks, block_size>>>(d_in, d_out, num_threads);
  cudaThreadSynchronize();

  // End timing the CUDA run,
  gettimeofday(&t_end, NULL);
  timeval_subtract(&t_diff, &t_end, &t_start);
  elapsed = t_diff.tv_sec*1e6+t_diff.tv_usec;

  // Copy the result form device to host
  cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

  // Free device memory.
  cudaFree(d_in);
  cudaFree(d_out);

  return elapsed;
}

unsigned long int cpuCalculation(unsigned int array_length, float* h_in, float* h_out) {

  // Start timing the CPU run.
  unsigned long int elapsed;
  struct timeval t_start, t_end, t_diff;
  gettimeofday(&t_start, NULL);

  // Perform serial calculations
  for (unsigned int i=0 ; i < array_length ; i++) {
      float x = h_in[i];
      h_out[i] = pow((x/(x-2.3)),3.0);
  }

  // End timing the CPU run,
  gettimeofday(&t_end, NULL);
  timeval_subtract(&t_diff, &t_end, &t_start);
  elapsed = t_diff.tv_sec*1e6+t_diff.tv_usec;

  return elapsed;
}

bool areSame(float* h_in, float* cpu_arr, float* cuda_arr, unsigned int array_length) {
  for(unsigned int i = 0 ; i < array_length ; i++) {
    if(!(std::abs(cpu_arr[i] - cuda_arr[i]) < std::numeric_limits<float>::epsilon())) {
      printf("INVALID:\nInput: %f\nCPU: %.15f\nGPU: %.15f\n", h_in[i], cpu_arr[i], cuda_arr[i]);
      return false;
    }
  }
  return true;
}

int main(int argc, char** argv) {
  unsigned int array_length = 753411;
  unsigned int mem_size   = array_length*sizeof(float);

  // Allocate host memory
  float* h_in  = (float*) malloc(mem_size);
  float* cuda_out = (float*) malloc(mem_size);
  float* cpu_out = (float*) malloc(mem_size);

  // Initialize the memory
  for(unsigned int i=0 ; i < array_length ; ++i) {
    h_in[i] = (float)i+1;
  }

  // Call the CUDA code.
  unsigned long int cuda_elapsed = cudaCalculation(array_length, mem_size, h_in, cuda_out);
  printf("CUDA took %d microseconds (%.2fms)\n",cuda_elapsed,cuda_elapsed/1000.0);

  // Call the CPU code.
  unsigned long int cpu_elapsed = cpuCalculation(array_length, h_in, cpu_out);
  printf("CPU took %d microseconds (%.2fms)\n",cpu_elapsed,cpu_elapsed/1000.0);

  if(areSame(h_in, cpu_out, cuda_out, array_length)) {
    printf("VALID: CPU and GPU agreed on results.\n");
  }

  // print result
  //for(unsigned int i=0; i<array_length; ++i) printf("%.6f\n", cuda_out[i]);

  // Clean up memory
  free(h_in);
  free(cuda_out);
  free(cpu_out);

  return 0;
}
