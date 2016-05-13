/*
 * bin_to_tiff.c
 *
 *  Created on: Jan 5, 2009
 *      Author: rasmussn
 */

#include "../tiff.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @argc
 * @argv[]
 */
int main(int argc, char* argv[])
{
   FILE * fd;
   int n, nx, ny;
   char * infile, * outfile;
   float * V;

   if (argc < 5) {
     printf("usage: bin_to_tiff infile outfile nx ny\n");
     exit(1);
   }

   infile  = argv[1];
   outfile = argv[2];
   nx = atoi(argv[3]);
   ny = atoi(argv[4]);

   V = (float *) malloc(nx * ny * sizeof(float));
   assert(V != NULL);

   fd = fopen(infile, "rb");
   if (fd == NULL) {
      fprintf(stderr, "%s: ERROR opening file %s\n", argv[0], infile);
   }

   n = fread(V, sizeof(float), nx*ny, fd);
   assert(n == nx*ny);
   fclose(fd);

   tiff_write_file(outfile, V, nx, ny);

   free(V);

   return 0;
}
