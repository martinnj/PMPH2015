#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ScanHost.cu.h"


int scanIncTest(bool is_segmented) {
    const unsigned int num_threads = 8353455;
    const unsigned int block_size  = 512;
    unsigned int mem_size = num_threads * sizeof(int);

    int* h_in    = (int*) malloc(mem_size);
    int* h_out   = (int*) malloc(mem_size);
    int* flags_h = (int*) malloc(num_threads*sizeof(int));

    int sgm_size = 123;
    { // init segments and flags
        for(unsigned int i=0; i<num_threads; i++) {
            h_in   [i] = 1;
            flags_h[i] = (i % sgm_size == 0) ? 1 : 0;
        }
    }

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);


    { // calling exclusive (segmented) scan
        int* d_in;
        int* d_out;
        int* flags_d;
        cudaMalloc((void**)&d_in ,   mem_size);
        cudaMalloc((void**)&d_out,   mem_size);
        cudaMalloc((void**)&flags_d, num_threads*sizeof(int));

        // copy host memory to device
        cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
        cudaMemcpy(flags_d, flags_h, num_threads*sizeof(int), cudaMemcpyHostToDevice);

        // execute kernel
        if(is_segmented)
            sgmScanInc< Add<int>,int > ( block_size, num_threads, d_in, flags_d, d_out );
        else
            scanInc< Add<int>,int > ( block_size, num_threads, d_in, d_out );

        // copy host memory to device
        cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

        // cleanup memory
        cudaFree(d_in );
        cudaFree(d_out);
        cudaFree(flags_d);
    }

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Scan Inclusive on GPU runs in: %lu microsecs\n", elapsed);

    // validation
    bool success = true;
    int  accum   = 0;
    if(is_segmented) {
        for(int i=0; i<num_threads; i++) {
            if (i % sgm_size == 0) accum  = 0;
            accum += 1;

            if ( accum != h_out[i] ) {
                success = false;
                //printf("Scan Exclusive Violation: %.1d should be %.1d\n", h_out[i], accum);
            }
        }
    } else {
        for(int i=0; i<num_threads; i++) {
            accum += 1;

            if ( accum != h_out[i] ) {
                success = false;
                //printf("Scan Exclusive Violation: %.1d should be %.1d\n", h_out[i], accum);
            }
        }
    }

    if(success) printf("\nScan Inclusive +   VALID RESULT!\n");
    else        printf("\nScan Inclusive + INVALID RESULT!\n");


    // cleanup memory
    free(h_in );
    free(h_out);
    free(flags_h);

    return 0;
}


/*
int scanExcTest(bool is_segmented) {
    const unsigned int num_threads = 8353455;
    const unsigned int block_size  = 512;
    unsigned int mem_size = num_threads * sizeof(int);

    int* h_in    = (int*) malloc(mem_size);
    int* h_out   = (int*) malloc(mem_size);
    int* flags_h = (int*) malloc(num_threads*sizeof(int));

    int sgm_size = 123;
    { // init segments and flags
        for(unsigned int i=0; i<num_threads; i++) {
            h_in   [i] = 1;
            flags_h[i] = (i % sgm_size == 0) ? 1 : 0;
        }
    }

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);


    { // calling exclusive (segmented) scan
        int* d_in;
        int* d_out;
        int* flags_d;
        cudaMalloc((void**)&d_in ,   mem_size);
        cudaMalloc((void**)&d_out,   mem_size);
        cudaMalloc((void**)&flags_d, num_threads*sizeof(int));

        // copy host memory to device
        cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
        cudaMemcpy(flags_d, flags_h, num_threads*sizeof(int), cudaMemcpyHostToDevice);

        // execute kernel
        if(is_segmented)
            sgmScanExc< Add<int>,int > ( block_size, num_threads, d_in, flags_d, d_out );
        else
            scanExc< Add<int>,int > ( block_size, num_threads, d_in, d_out );

        // copy host memory to device
        cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

        // cleanup memory
        cudaFree(d_in );
        cudaFree(d_out);
        cudaFree(flags_d);
    }

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Scan Exclusive on GPU runs in: %lu microsecs\n", elapsed);

    // validation
    bool success = true;
    int  accum   = 0;
    if(is_segmented) {
        for(int i=0; i<num_threads; i++) {
            if (i % sgm_size == 0) accum  = 0;
            accum += 1;

            if ( accum != h_out[i] ) {
                success = false;
                //printf("Scan Exclusive Violation: %.1d should be %.1d\n", h_out[i], accum);
            }
        }
    } else {
        for(int i=0; i<num_threads-1; i++) {
            accum += 1;

            if ( accum != h_out[i] ) {
                success = false;
                //printf("Scan Exclusive Violation: %.1d should be %.1d\n", h_out[i], accum);
            }
        }
    }

    if(success) printf("\nScan Exclusive +   VALID RESULT!\n");
    else        printf("\nScan Exclusive + INVALID RESULT!\n");


    // cleanup memory
    free(h_in );
    free(h_out);
    free(flags_h);

    return 0;
}*/

int gpuMss(int* h_start, unsigned long num_threads, unsigned int block_size) {
//int gpuMss() {
    //unsigned int num_threads = 8353455;
    //unsigned int block_size  = 512;
    unsigned int mem_start_size = num_threads * sizeof(int);
    unsigned int mem_mapped_size = num_threads * sizeof(MyInt4);
    unsigned int num_blocks = ( (num_threads % block_size) == 0) ?
                    num_threads / block_size     :
                    num_threads / block_size + 1 ;

    //int* h_start  = (int*) malloc(mem_start_size);
    //MyInt4* h_mapped = (MyInt4*) malloc(mem_mapped_size);
    MyInt4* h_scanned = (MyInt4*) malloc(mem_mapped_size);

    //for(unsigned int i=0; i<num_threads; i++) {
    //    h_start[i] = 1;
    //}

    int* d_start;
    cudaMalloc((void**)&d_start, mem_start_size);
    cudaMemcpy(d_start, h_start, mem_start_size, cudaMemcpyHostToDevice);


    MyInt4* d_mapped;
    cudaMalloc((void**)&d_mapped, mem_mapped_size);


    // Perform the map to create the 4-tubles.
    MssMapKernel<int, MyInt4><<<num_blocks, block_size >>> (d_start, d_mapped, num_threads);
    //cudaMemcpy(h_mapped, d_mapped, mem_mapped_size, cudaMemcpyDeviceToHost);


    MyInt4* d_scanned;
    cudaMalloc((void**)&d_scanned, mem_mapped_size);
    scanInc<MsspOp, MyInt4> (block_size, num_threads, d_mapped, d_scanned);
    cudaMemcpy(h_scanned, d_scanned, mem_mapped_size, cudaMemcpyDeviceToHost);

    int res = h_scanned[num_threads-1].x;


    cudaFree(d_start);
    cudaFree(d_mapped);
    cudaFree(d_scanned);
    //free(h_start);
    free(h_scanned);

    return res;
}

int cpuMss(int* input, unsigned long input_size) {
    unsigned int mem_int4_size = input_size * sizeof(MyInt4);

    MyInt4* mapped = (MyInt4*) malloc(mem_int4_size);
    MyInt4* scanned = (MyInt4*) malloc(mem_int4_size);

    for (unsigned int i = 0 ; i < input_size ; i++) {
        int x = input[i];
        mapped[i] = MyInt4(x,x,x,x);
    }

    /*for (unsigned int i = 0; i < input_size ; i++) {
        MyInt4 res = mapped[i];
        printf("mapped[%d]: %d, %d, %d, %d\n", i, res.x, res.y, res.z, res.w);
    }*/

    MyInt4 identity = CpuMsspOp::identity();
    MyInt4 first = mapped[0];
    scanned[0] = CpuMsspOp::apply(identity, first);

    for (unsigned int i = 1 ; i < input_size ; i++) {
        MyInt4 x = mapped[i];
        MyInt4 y = scanned[i-1];
        scanned[i] = CpuMsspOp::apply(y,x);
    }

    /*for (unsigned int i = 0; i < input_size ; i++) {
        MyInt4 res = scanned[i];
        printf("scanned[%d]: %d, %d, %d, %d\n", i, res.x, res.y, res.z, res.w);
    }*/


    int res = scanned[input_size-1].x;
    free(mapped);
    free(scanned);
    return res;
}

int RandomNumber(int lower, int upper) {
    return lower + rand() % (upper - lower + 1);
}

void mssCompare(int* input, unsigned long input_size) {
    unsigned int block_size  = 512;
    unsigned long num_threads = input_size;

    unsigned long int elapsedGpu, elapsedCpu;

    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);
    int gpuRes = gpuMss(input, num_threads, block_size);
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsedGpu = (t_diff.tv_sec*1e6+t_diff.tv_usec);


    gettimeofday(&t_start, NULL);
    int cpuRes = cpuMss(input, num_threads);
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsedCpu = (t_diff.tv_sec*1e6+t_diff.tv_usec);

    if(gpuRes == cpuRes) {
        printf("Results are VALID, execution times:\n");
        printf(" - GPU runs in: %lu microsecs\n", elapsedGpu);
        printf(" - CPU runs in: %lu microsecs\n", elapsedCpu);
    } else {
        printf("Results are INVALID (CPU: %d, GPU: %d) , execution times:\n",cpuRes, gpuRes);
        //printf(" - GPU runs in: %lu microsecs\n", elapsedGpu);
        //printf(" - CPU runs in: %lu microsecs\n", elapsedCpu);
        printf("Input: ");
        for (int i = 0 ; i < input_size ; i++) {
            printf("%d, ", input[i]);
        }
        printf("\n");
    }



    //Print here.

    return;
}

void mssTest(int iterations, int list_length) {

    int* input = (int*) malloc(list_length * sizeof(int));

    for(int i = 0 ; i < iterations ; i++) {

        srand (time(NULL)); // Seed the damn generator.

        for(int j = 0 ; j < list_length ; j++) {
            input[j] = RandomNumber(0,100);
            //printf("Generated: %d\n", input[j]);
        }
        mssCompare(input, list_length);
    }

    free(input);
}



int main(int argc, char** argv) {
    //scanIncTest(true);
    //scanIncTest(true);
    //scanIncTest(false);

    mssTest(10, 10000000);
}
