#ifndef _CPU_FUNC
#define _CPU_FUNC

/**
 * Create an MxN matrix filled with the element eye.
 */
template<class T>
T** createMatrix(int M, int N, T eye) {
    T** A = new T*[M];
    for(int i = 0; i < M; i++)
        A[i] = new T[N];

    for (int i = 0 ; i < M ; i++) {
        for (int j = 0 ; j < N ; j++) {
            A[i][j] = eye;
        }
    }

    return A;
}

template<class T>
void freeMatrix(T** A, int M) {
    for(int i = 0; i < M; i++)
        free(A[i]);
    free(A);
}

/**
 * Transposes the MxN matrix contained in A and stores the result in B.
 * A and B must have a length corresponding to M*N.
 * N = number of rows, M = number of columns.
 */
template<class T>
void cpuTranspose(T** A, T** B, int M, int N) {
    for(int i = 0 ; i < M ; i++) {
        for(int j = 0 ; j < N ; j++) {
            B[j][i] = A[i][j];
        }
    }
}

template<class T>
void matrixPrint(T** A, int M, int N) {
    for (int i = 0 ; i < M ; i++) {
        for (int j = 0 ; j < N ; j++) {
            printf("%.2f ",A[i][j]);
        }
        printf("\n");
    }
}

#endif // _CPU_FUNC
