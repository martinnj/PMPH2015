#ifndef _CPU_FUNC
#define _CPU_FUNC

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
            if(A[i*N+j] != B[i*N+j]) {
                //printf("Compare error at [%d,%d]:\nCPU: %.2f\nGPU: %.2f\n", i, j, A[i*M+j], B[i*M+j]);
                return false;
            }
        }
    }
    return true;
}

#endif // _CPU_FUNC
