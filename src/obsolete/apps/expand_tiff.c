/*
 * expand_tiff.c
 *
 * Expands the tiff image by the given amount, filling in border with 0.0s
 *
 */

#include "src/io/io.h"
#include "src/io/tiff.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
   int nx, ny, nxIn, nyIn;
   int ii, i, j, i0, j0;
   char * infile, * outfile;
   float * V, * input;

   if (argc < 5) {
     printf("usage: expand_tiff_image infile outfile nx ny\n");
     exit(1);
   }

   infile  = argv[1];
   outfile = argv[2];
   nx = atoi(argv[3]);
   ny = atoi(argv[4]);

   V = (float *) malloc(nx * ny * sizeof(float));
   assert(V != NULL);

   input = (float *) malloc(nx * ny * sizeof(float));
   assert(input != NULL);

   nxIn = nx;
   nyIn = ny;
   tiff_read_file(infile, input, &nxIn, &nyIn);

   i0 = nx/2 - nxIn/2;
   j0 = ny/2 - nyIn/2;

   for (i = 0; i < nx*ny; i++) {
      V[i] = 0;
   }

   ii = 0;
   for (j = j0; j < j0+nyIn; j++) {
      for (i = i0; i < i0+nxIn; i++) {
         V[i+nx*j] = input[ii++];
      }
   }

   tiff_write_file(outfile, V, nx, ny);

   free(input);
   free(V);

   return 0;
}
