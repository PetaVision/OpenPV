#include <stdio.h>
#include <stdlib.h>


int read_header(FILE *inputfp, int *params);
int write_header(FILE *outputfp, int *params);
int transfer_patch_data(FILE *inputfp, FILE *outputfp, int *params);
int transfer_patch(FILE *inputfp, FILE *outputfp, int nf, int minVal, int maxVal);
