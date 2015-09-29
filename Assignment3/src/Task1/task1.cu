#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "cpuFunc.cu.h"


int main(int argc, char** argv) {

    float M = 3;
    float N = 3;
    float** A = createMatrix<float>(M,N,0.0);
    float** B = createMatrix<float>(M,N,0.0);

    for (int i = 0 ; i < M ; i++) {
        for (int j = 0 ; j < N ; j++) {
            A[i][j] = i*M+j+1.0;
        }
    }


    matrixPrint(A,M,N);
    cpuTranspose<float>(A,B,3,3);
    printf("------\n");
    matrixPrint(B,M,N);

    // TODO: FREE THE STUFF
    freeMatrix<float>(A,M);
    freeMatrix<float>(B,M);

    return 0;
}
