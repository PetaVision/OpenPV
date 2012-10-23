/*
 * io.c
 *
 *  Created on: Aug 3, 2008
 *      Author: dcoates
 */

// Shared input and output routines

#include "io.h"
#include "tiff.h"

#include <assert.h>
#include <float.h>	// FLT_MAX/MIN
#include <math.h>
#include <string.h>     // memcpy

void usage()
{
   printf("\nUsage:\n");
   printf(" -p <parameters filename>\n");
   printf(" [-n <number of timesteps>]\n");
   printf(" [-o <output directory>\n");
   printf(" [-s <random number generator seed>]\n");
   printf(" [-d <OpenCL device>]\n");
   printf(" [-w <working directory>]\n");
}

/**
 * @argc
 * @argv
 * @input_file
 * @param_file
 * @n_time_steps
 * @device
 */
int parse_options(int argc, char * argv[], char ** output_path,
                  char ** param_file, long int * n_time_steps, int * opencl_device,
                  unsigned long * random_seed, char ** working_dir)
{
   if (argc < 2) {
      usage();
      return -1;
   }

   // *n_time_steps = 1;
   // parse_options should not set defaults; calling routine should set default
   // before calling parse_options.

   pv_getopt_long(argc, argv, "-n", n_time_steps);
   pv_getopt_int(argc, argv, "-d", opencl_device);
   pv_getopt_str(argc, argv, "-o", output_path);
   pv_getopt_str(argc, argv, "-p", param_file);
   pv_getopt_unsigned_long(argc, argv, "-s", random_seed);
   pv_getopt_str(argc, argv, "-w", working_dir);

   return 0;
}

/**
 * @argc
 * @argv
 * @opt
 * @iVal
 */
int pv_getopt_int(int argc, char * argv[], const char * opt, int * iVal)
{
   int i;
   for (i = 1; i < argc; i += 1) {
      if (i+1 < argc && strcmp(argv[i], opt) == 0) {
         if( iVal != NULL ) *iVal = atoi(argv[i+1]);
         return 0;
      }
   }
   return -1;  // not found
}

/**
 * @argc
 * @argv
 * @opt
 * @iVal
 */
int pv_getopt_long(int argc, char * argv[], const char * opt, long int * iVal)
{
   int i;
   for (i = 1; i < argc; i += 1) {
      if (i+1 < argc && strcmp(argv[i], opt) == 0) {
         if( iVal != NULL ) *iVal = strtol(argv[i+1], NULL, 0);
         return 0;
      }
   }
   return -1;  // not found
}

/**
 * @argc
 * @argv
 * @opt
 * @iVal
 */
int pv_getopt_unsigned_long(int argc, char * argv[], const char * opt, unsigned long * iVal)
{
   int i;
   for (i = 1; i < argc; i += 1) {
      if (i+1 < argc && strcmp(argv[i], opt) == 0) {
         if( iVal != NULL ) *iVal = strtoul(argv[i+1], NULL, 0);
         return 0;
      }
   }
   return -1;  // not found
}

/**
 * @argc
 * @argv
 * @opt
 * @sVal
 */
int pv_getopt_str(int argc, char * argv[], const char * opt, char ** sVal)
{
   int i;
   for (i = 1; i < argc; i += 1) {
      if (i+1 < argc && strcmp(argv[i], opt) == 0) {
         if( sVal != NULL ) *sVal = strdup(argv[i+1]);
         return 0;
      }
   }
   return -1;  // not found
}

#ifdef OBSOLETE
// For MATLAB use, just save a number with name as comment
#define LOGINTPARM(paramfile, which) \
    fprintf(paramfile, "%d %% %s\n", which, #which)
#define LOGFPARM(paramfile, which) \
    fprintf(paramfile, "%f %% %s\n", which, #which)
#define LOGSPARM(paramfile, which) \
    fprintf(paramfile, #which "=%s\n", which)
#endif // OBSOLETE


#define TIFF_FILE_TYPE    1
#define BINARY_FILE_TYPE  2

/**
 * @filename
 */
static int filetype(const char * filename)
{
   int n = strlen(filename);
   if (strncmp(&filename[n-4], ".tif", 4) == 0) return TIFF_FILE_TYPE;
   if (strncmp(&filename[n-4], ".bin", 4) == 0) return BINARY_FILE_TYPE;
   return 0;
}

/**
 * @V
 * @nx0
 * @ny0
 * @nx
 * @ny
 */
int pv_center_image(float * V, int nx0, int ny0, int nx, int ny)
{
   int i0, j0, i, j, ii;

   float * buf = (float *) malloc(nx0 * ny0 * sizeof(float));
   assert(buf != NULL);

   assert(nx0 <= nx);
   assert(ny0 <= ny);

   memcpy(buf, V, nx0 * nx0 * sizeof(float));

   i0 = nx/2 - nx0/2;
   j0 = ny/2 - ny0/2;

   for (i = 0; i < nx*ny; i++) {
      V[i] = 0;
   }

   ii = 0;
   for (j = j0; j < j0+ny0; j++) {
      for (i = i0; i < i0+nx0; i++) {
         V[i+nx*j] = buf[ii++];
      }
   }
   free(buf);

   return 0;
}

/**
 * @filename
 * @buf
 * @nx
 * @ny
 */
int readFile(const char * filename, float * buf, int * nx, int * ny)
{
   int result, nItems;
   int status = 0;
   FILE * fd;
   const char * altfile = INPUT_PATH "const_one_64x64.bin";

   if (filename == NULL) {
      filename = altfile;
      fprintf(stderr, "[ ]: Warning: Input file is NULL -- using %s.\n", filename);
   }

   if (filetype(filename) == TIFF_FILE_TYPE) {
      return tiff_read_file(filename, buf, nx, ny);
   }

   fd = fopen(filename, "rb");

   if (fd == NULL) {
      fprintf(stderr, "[ ]: readFile: ERROR, Input file %s not found.\n", filename);
      return 1;
   }
   else {
      nItems = (*nx) * (*ny);

      // assume binary file
      assert(filetype(filename) == BINARY_FILE_TYPE);
      result = fread(buf, sizeof(float), nItems, fd);
      fclose(fd);

      if (result != nItems) {
         pv_log(stderr, "[ ]: Warning: readFile %s, expected %d, got %d.\n",
                filename, nItems, result);
         status = 1;
      }
      else {
#ifdef DEBUG_OUTPUT
         printf("[ ]: readFile: Successfully read %d items from %s\n", nItems, filename);  fflush(stdout);
#endif // DEBUG_OUTPUT
      }
   }

   return status;
}

/**
 * @fp
 * @numParams
 * @params
 */
int pv_read_binary_params(FILE * fp, int numParams, int params[])
{
   if ((size_t) numParams > NUM_BIN_PARAMS) {
      numParams = NUM_BIN_PARAMS;
   }
   rewind(fp);

   return fread(params, sizeof(int), numParams, fp);
}

/**
 * Open a PV binary file for reading.
 * @numParams contains the number of integer parameters on return
 * @nx contains the number of items in the x direction on return
 * @ny contains the number of items in the y direction on return
 * @nf contains the number of features on return
 * returns the opened file (NULL if an error occurred)
 */
FILE * pv_open_binary(const char * filename, int * numParams, int * type, int * nx, int * ny, int * nf)
{
   int params[MIN_BIN_PARAMS];

   FILE * fp = fopen(filename, "rb");
   if (fp == NULL) {
      pv_log(stderr, "pv_open_binary: couldn't open input file %s\n", filename);
      return NULL;
   }

   if ( fread(params, sizeof(int), MIN_BIN_PARAMS, fp) != MIN_BIN_PARAMS ) {
      fclose(fp);
      return NULL;
   }

   *numParams = params[INDEX_NUM_PARAMS];
   *type      = params[INDEX_FILE_TYPE];
   *nx        = params[INDEX_NX];
   *ny        = params[INDEX_NY];
   *nf        = params[INDEX_NF];

   return fp;
}

/**
 * Close a PV binary file.
 * @fp
 */
int pv_close_binary(FILE * fp)
{
   return fclose(fp);
}

/**
 * @fd
 * @patch
 */
int pv_text_write_patch(FILE * fd, PVPatch * patch, pvdata_t * data, int nf, int sx, int sy, int sf)
{
   int f, i, j;

   const int nx = (int) patch->nx;
   const int ny = (int) patch->ny;
   //const int nf = (int) patch->nf;

   //const int sx = (int) patch->sx;  assert(sx == nf);
   //const int sy = (int) patch->sy;  //assert(sy == nf*nx);
   //const int sf = (int) patch->sf;  assert(sf == 1);

   assert(fd != NULL);

   for (f = 0; f < nf; f++) {
      for (j = 0; j < ny; j++) {
         for (i = 0; i < nx; i++) {
            fprintf(fd, "%7.5f ", data[i*sx + j*sy + f*sf]);
         }
         fprintf(fd, "\n");
      }
      fprintf(fd, "\n");
   }

   return 0;
}
