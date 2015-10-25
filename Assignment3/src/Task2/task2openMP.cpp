#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <sys/time.h>

#include "cpuFunc.cu.h"

#define COLS 64

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

template<class T>
unsigned long int task2b(T* A, T* B, int rows) {

    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    #pragma omp parallel for default(shared) schedule(static)
    for (unsigned i = 0 ; i < rows ; i++) {
        T accum = A[fmo(i, 0, COLS)] * A[fmo(i, 0, COLS)];
        B[fmo(i, 0, COLS)] = accum;

        for (unsigned j = 1 ; j < 64 ; j++) {
            T tmpA = A[fmo(i, j, COLS)];
            accum = sqrt(accum) + tmpA * tmpA;
            B[fmo(i, j, COLS)] = accum;
        }
    }

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    return (t_diff.tv_sec*1e6+t_diff.tv_usec);
}

void task2btest(int iterations, int rows) {

    printf("\n");
    printf("==========================================\n");
    printf("============== TASK 2 BONUS ==============\n");
    printf("==========================================\n");
    printf("Configuration:\n");
    printf(" - # of runs: %d\n", iterations);
    printf(" - Matrix dimensions: A[%d,%d].\n\n", rows, COLS);

    float* A = flatMatrixCreate<float>(rows, COLS, 1.0);
    unsigned long int timespent = 0;

    for ( int i = 0 ; i < iterations ; i++) {
        float* B = flatMatrixCreate<float>(rows, COLS, -1.0);
        timespent += task2b<float>(A, B, rows);
        //flatMatrixPrint(B, rows, COLS);
        flatMatrixFree(B);
    }

    flatMatrixFree(A);
    printf("\n");
    printf("Average OpenMP runtime: %d microseconds.\n", (int)(timespent/iterations));
    return;

}

int main(int argc, char** argv) {
    task2btest(100, 10);
    task2btest(100, 20);
    task2btest(100, 30);
    task2btest(100, 40);
    task2btest(100, 50);
    task2btest(100, 60);
    task2btest(100, 70);
    task2btest(100, 80);
    task2btest(100, 90);
    task2btest(100, 100);
    task2btest(100, 2000);
    task2btest(100, 3000);
    task2btest(100, 4000);
    task2btest(100, 5000);
    //task2btest(100, 10000);
    return 0;
}
