/*
 * create_input.c
 *
 *  Created on: Nov 3, 2008
 *      Author: rasmussn
 */


#include "src/include/pv_common.h"
#include <stdio.h>

int main(int argc, char* argv[])
{
   int k;
   const int nx = NX;
   const int ny = NY;
   const char* filename = "input/constant_ones_64x64.bin";

   float V[nx*ny];

   FILE* fp = fopen(filename, "wb");
   if (fp == NULL) {
      fprintf(stderr, "%s: ERROR opening file %s\n", argv[0], filename);
   }

   for (k = 0; k < nx*ny; k++) {
      V[k] = 1.0;
   }

   fwrite(V, sizeof(float), nx*ny, fp);
   fclose(fp);

   return 0;
}
