/* File: example.c */

#include "example.h"

int fact(int *a,int n) {
    printf("%d\n", n);
    if (n < 0){ /* This should probably return an error, but this is simpler */
        *a = 9;
        return -1;
    }
    else if (n == 0) {
        *a = 1;
        return 0;
    }
    else {
        /* testing for overflow would be a good idea here */
        int p;
        int j = fact(&p, n-1);
        *a = p * n ;
        return 1;
    }
}
