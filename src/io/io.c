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
#include <float.h>  // FLT_MAX/MIN
#include <math.h>
#include <string.h>     // memcpy

void usage()
{
   printf("\nUsage:\n");
   printf(" -p <parameters filename>\n");
   // printf(" [-n <number of timesteps>]\n");
   printf(" [-o <output directory>\n");
   printf(" [-s <random number generator seed>]\n");
   printf(" [-d <OpenCL device>]\n");
   printf(" [-w <working directory>]\n");
   printf(" [-r|-c <checkpoint directory>]\n");
}

/**
 * @argc
 * @argv
 * @input_file
 * @param_file
 * @n_time_steps
 * @device
 */
int parse_options(int argc, char * argv[], bool * require_return,
                  char ** output_path, char ** param_file, int * opencl_device,
                  unsigned int * random_seed, char ** working_dir, int * restart, char ** checkpointReadDir, int * numthreads)
{
   if (argc < 2) {
      usage();
      return -1;
   }

   bool reqrtn = false; 
   int arg;
   for(arg=1; arg<argc; arg++) { 
      if( !strcmp(argv[arg], "--require-return")) { 
         reqrtn = true; 
         break; 
      } 
   } 
   *require_return = reqrtn;

   pv_getopt_int(argc, argv, "-d", opencl_device);
   pv_getoptionalopt_int(argc, argv, "-t", numthreads, 0);
   pv_getopt_str(argc, argv, "-o", output_path);
   pv_getopt_str(argc, argv, "-p", param_file);
   pv_getopt_unsigned(argc, argv, "-s", random_seed);
   pv_getopt_str(argc, argv, "-w", working_dir);
   if (pv_getopt(argc, argv, "-r") == 0) { *restart = 1; }
   pv_getopt_str(argc, argv, "-c", checkpointReadDir);

   return 0;
}

/*
 * @argc
 * @argv
 * @opt
 */
int pv_getopt(int argc, char * argv[], const char * opt)
{
   int i;
   for (i = 1; i < argc; i++) {
      if (strcmp(argv[i], opt)==0) {
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

int pv_getoptionalopt_int(int argc, char * argv[], const char * opt, int * iVal, int defaultVal)
{
   int i;
   for (i = 1; i < argc; i += 1) {
      if(strcmp(argv[i], opt) == 0){
         //Default parameter
         if (i+1 >= argc || argv[i+1][0] == '-') {
            if( iVal != NULL) *iVal = defaultVal;
         }
         else{
            if( iVal != NULL ) *iVal = atoi(argv[i+1]);
         }
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
int pv_getopt_unsigned(int argc, char * argv[], const char * opt, unsigned int * uVal)
{
   int i;
   for (i = 1; i < argc; i += 1) {
      if (i+1 < argc && strcmp(argv[i], opt) == 0) {
         if( uVal != NULL ) *uVal = (unsigned int) strtoul(argv[i+1], NULL, 0);
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
   // sVal can be NULL.  If sVal is not null and the option is found,
   // the value of the option is put into sVal and the calling routine is
   // responsible for freeing it.
   // Example: if argv[1] is "-p" and argv[2] is "params.pv", and opt is "-p",
   // *sVal will be "params.pv".
   int i;
   for (i = 1; i < argc; i += 1) {
      if (i+1 < argc && strcmp(argv[i], opt) == 0) {
         if( sVal != NULL ) *sVal = strdup(argv[i+1]);
         return 0;
      }
   }
   if (sVal != NULL) *sVal = NULL;
   return -1;  // not found
}

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

char * expandLeadingTilde(char const * path) {
   char * newpath = NULL;
   if (path != NULL) {
      int len = strlen(path);
      if (len==1 && path[0]=='~') {
         newpath = strdup(getenv("HOME"));
         if (newpath==NULL) {
            fprintf(stderr, "Error expanding \"%s\": home directory not defined\n", path);
            exit(EXIT_FAILURE);
         }
      }
      else if (len>1 && path[0]=='~' && path[1]=='/') {
         char * homedir = getenv("HOME");
         if (homedir==NULL) {
            fprintf(stderr, "Error expanding \"%s\": home directory not defined\n", path);
         }
         char dummy;
         int chars_needed = snprintf(&dummy, 0, "%s/%s", homedir, &path[2]);
         newpath = (char *) malloc(chars_needed+1);
         if (newpath==NULL) {
            fprintf(stderr, "Unable to allocate memory for path \"%s/%s\"\n", homedir, &path[2]);
            exit(EXIT_FAILURE);
         }
         int chars_used = snprintf(newpath, chars_needed+1, "%s/%s", homedir, &path[2]);
         assert(chars_used == chars_needed);
      }
      else {
         newpath = strdup(path);
      }
   }
   return newpath;
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
   FILE * fp;
   const char * altfile = INPUT_PATH "const_one_64x64.bin";

   if (filename == NULL) {
      filename = altfile;
      fprintf(stderr, "[ ]: Warning: Input file is NULL -- using %s.\n", filename);
   }

   if (filetype(filename) == TIFF_FILE_TYPE) {
      return tiff_read_file(filename, buf, nx, ny);
   }

   fp = fopen(filename, "rb");

   if (fp == NULL) {
      fprintf(stderr, "[ ]: readFile: ERROR opening input file %s: %s\n", filename, strerror(errno));
      return 1;
   }
   else {
      nItems = (*nx) * (*ny);

      // assume binary file
      assert(filetype(filename) == BINARY_FILE_TYPE);
      result = fread(buf, sizeof(float), nItems, fp);
      fclose(fp);

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
 * @fd
 * @patch
 */
int pv_text_write_patch(PV_Stream * pvstream, PVPatch * patch, pvwdata_t * data, int nf, int sx, int sy, int sf)
{
   int f, i, j;

   const int nx = (int) patch->nx;
   const int ny = (int) patch->ny;
   //const int nf = (int) patch->nf;

   //const int sx = (int) patch->sx;  assert(sx == nf);
   //const int sy = (int) patch->sy;  //assert(sy == nf*nx);
   //const int sf = (int) patch->sf;  assert(sf == 1);

   assert(pvstream != NULL);

   FILE * fd = pvstream->fp;
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
