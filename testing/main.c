#include <stdio.h>
#include <stdlib.h>
#include "matrices.h"
#include <math.h>


int main(void) {
    mat *A = mat_user_fill(mat_init(3, 1));
    mat *B = mat_user_fill(mat_init(3, 1));
    if (!A || !B) {
        printf("Failed to allocate memory for A.\n");
        return 1;
    }

    mat *C = vec_cross(A, B);

    printf("A x B is:\n\n");
    mat_print(C);



    mat_free(A);
    mat_free(B);
    mat_free(C);
    return 0;
}