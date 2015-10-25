#ifndef _CPU_FUNC
#define _CPU_FUNC

void vecFPrint(float* v, int vLen) {
    printf("[ ");
    for( unsigned i = 0 ; i < vLen ; i++) {
        printf(" %.1f ", v[i]);
    }
    printf("]\n");
}

void vecIPrint(int* v, int vLen) {
    printf("[ ");
    for( unsigned i = 0 ; i < vLen ; i++) {
        printf(" %d ", v[i]);
    }
    printf("]\n");
}

#endif // _CPU_FUNC
