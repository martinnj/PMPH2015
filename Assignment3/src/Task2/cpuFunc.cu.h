#ifndef _CPU_FUNC
#define _CPU_FUNC


#include <limits>

/**
 * Create an MxN matrix filled with the element eye.
 */
template<class T>
T* flatMatrixCreate(int M, int N, T eye) {
    T* A = (T*) malloc(sizeof(T)*(M*N));
    int MN = M*N;
    for(int i = 0 ; i < MN ; i++) {
        A[i] = eye;
    }
    return A;
}

/**
 * Frees the given matrix.
 */
template<class T>
void flatMatrixFree(T* A) {
    free(A);
}

// Flat Matrix Offset
int fmo(int row, int col, int matrixWidth) {
    return row * matrixWidth + col;
}

template<class T>
void flatMatrixTask2b(T* A, T* B, int rows, int cols) {

    for (int i = 0 ; i < rows ; i++) {
        T accum = A[fmo(i, 0, cols)] * A[fmo(i, 0, cols)];
        B[fmo(i, 0, cols)] = accum;

        for (int j = 1 ; j < cols ; j++) {
            T tmpA = A[fmo(i, j, cols)];
            accum = sqrt(accum) + tmpA * tmpA;
            B[fmo(i, j, cols)] = accum;
        }
    }
}

/**
 * Transposes the MxN matrix contained in A and stores the result in B.
 * A and B must have a length corresponding to M*N.
 * N = number of rows, M = number of columns.
 */
template<class T>
void flatMatrixTranspose(T* A, T* B, int M, int N) {
    for(int i = 0 ; i < M ; i++) {
        for(int j = 0 ; j < N ; j++) {
            B[j*M+i] = A[i*N+j];
        }
    }
}

/**
 * Prints a given M x N matrix.
 */
template<class T>
void flatMatrixPrint(T* A, int M, int N) {
    printf("\n");
    for (int i = 0 ; i < M ; i++) {
        for (int j = 0 ; j < N ; j++) {
            printf("%.2f ",A[i*N+j]);
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * Compares matrix A & B on a per-value basis, both matrices must have the same
 * M x N dimensions.
 */
template<class T>
bool flatMatrixCompare(T* A, T* B, int M, int N) {
    for (int i = 0 ; i < M ; i++) {
        for (int j = 0 ; j < N ; j++) {
            if(abs(A[i*N+j] - B[i*N+j]) > 0.0001) {//std::numeric_limits<float>::epsilon()) {
                printf("Data before error = [%d,%d]:\nCPU: %.20f\nGPU: %.20f\n", i, j-1, A[i*N+j-1], B[i*N+j-1]);
                printf("Compare error at [%d,%d]:\nCPU: %.20f\nGPU: %.20f\n", i, j, A[i*N+j], B[i*N+j]);
                return false;
            }
        }
    }
    return true;
}

#endif // _CPU_FUNC
