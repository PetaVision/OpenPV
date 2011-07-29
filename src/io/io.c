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

static int pv_getopt_int(int argc, char * argv[], char * opt, int *   iVal);
static int pv_getopt_str(int argc, char * argv[], char * opt, char ** sVal);
static int pv_getopt_unsigned_long(int argc, char * argv[], char * opt, unsigned long * ulVal);

void usage()
{
   printf("\nUsage:\n");
   printf(" -p <parameters filename>\n");
   printf(" [-n <number of timesteps>]\n");
   printf(" [-o <output directory>\n");
   printf(" [-i <input image filename>]\n");
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
int parse_options(int argc, char * argv[], char ** output_path, char ** input_file,
                  char ** param_file, int * n_time_steps, int * opencl_device, unsigned long * random_seed, char ** working_dir)
{
   if (argc < 2) {
      usage();
      return -1;
   }

   // *n_time_steps = 1;
   // parse_options should not set defaults; calling routine should set default
   // before calling parse_options.

   pv_getopt_int(argc, argv, "-n", n_time_steps);
   pv_getopt_int(argc, argv, "-d", opencl_device);
   pv_getopt_str(argc, argv, "-i", input_file);
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
static int pv_getopt_int(int argc, char * argv[], char * opt, int * iVal)
{
   int i;
   for (i = 1; i < argc; i += 1) {
      if (i+1 < argc && strcmp(argv[i], opt) == 0) {
         *iVal = atoi(argv[i+1]);
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
static int pv_getopt_unsigned_long(int argc, char * argv[], char * opt, unsigned long * iVal)
{
   int i;
   for (i = 1; i < argc; i += 1) {
      if (i+1 < argc && strcmp(argv[i], opt) == 0) {
         *iVal = strtoul(argv[i+1], NULL, 0);
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
static int pv_getopt_str(int argc, char * argv[], char * opt, char ** sVal)
{
   int i;
   for (i = 1; i < argc; i += 1) {
      if (i+1 < argc && strcmp(argv[i], opt) == 0) {
         *sVal = strdup(argv[i+1]);
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
#endif


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
#endif
      }
   }

   return status;
}

/**
 * copy local portion of globalBuf to localBuf
 * @l
 * @globalBuf
 * @localBuf
 * @comm
 */
int scatterReadBuf(PVLayer* l, float* globalBuf, float* localBuf, MPI_Comm comm)
{
  // everyone gets a copy of the entire input
#ifdef PV_USE_MPI
   int nTotal = l->loc.nxGlobal * l->loc.nyGlobal;
   MPI_Bcast(globalBuf, nTotal, MPI_FLOAT, 0, comm);
#endif // PV_USE_MPI

   // everyone copies only their local patch of input
   int kl, kg, status = 0;
   int nLocal = l->loc.nx * l->loc.ny;
   for (kl = 0; kl < nLocal; kl++) {
      kg = globalIndexFromLocal(kl, l->loc);
      localBuf[kl] = globalBuf[kg];
   }
   return status;
}

/**
 * @filename
 * @l
 * @localBuf
 * @comm
 */
int scatterReadFile(const char* filename, PVLayer* l, float* localBuf, MPI_Comm comm)
{
   int status = 0;
   int nx, ny;

   int nTotal = l->loc.nxGlobal * l->loc.nyGlobal;

   float* globalBuf = (float*) malloc(nTotal * sizeof(float));
   if (globalBuf == NULL) return -1;

   if (l->columnId == 0) {
      nx = l->loc.nxGlobal;
      ny = l->loc.nyGlobal;
      status = readFile(filename, globalBuf, &nx, &ny);
      if (status == 0) {
         assert(nx <= l->loc.nxGlobal);
         assert(ny <= l->loc.nyGlobal);
         if (nx < l->loc.nxGlobal || ny < l->loc.nyGlobal) {
            status = pv_center_image(globalBuf, nx, ny, l->loc.nxGlobal, l->loc.nyGlobal);
         }
      }
   }

   if (status != 0) return status;

   // everyone gets a copy of the entire input
   scatterReadBuf(l, globalBuf, localBuf, comm);

//   // everyone gets a copy of the entire input
//   MPI_Bcast(tmp, nTotal, MPI_FLOAT, 0, comm);
//
//   // everyone copies only their local patch of input
//   for (kl = 0; kl < nLocal; kl++) {
//      kg = globalIndexFromLocal(kl, l->loc, l->numFeatures);
//      buf[kl] = tmp[kg];
//   }
   if (globalBuf) free(globalBuf);

   return status;
}

/**
 * @filename
 * @l
 * @ibuf
 * @comm
 */
int gatherWriteFile(const char* filename, PVLayer* l, float* ibuf, MPI_Comm comm)
{
   int commRank, commSize, status = 0;
   int c, kl, nLocal, nTotal;
   float kxy[2], *kbuf, *tbuf;

   // WARNING, this assumes layers in separate hypercolumns have the same (nx,ny)

#ifdef PV_USE_MPI
   MPI_Comm_rank(comm, &commRank);
   MPI_Comm_size(comm, &commSize);
#else // PV_USE_MPI
   commRank = 0;
   commSize = 1;
#endif // PV_USE_MPI
   assert(commRank == l->columnId);

   nTotal = l->loc.nxGlobal * l->loc.nyGlobal;
   nLocal = l->loc.nx * l->loc.ny;

#ifdef DEBUG_OUTPUT
   printf("[%d]: gatherWriteFile: nTotal=%d nLocal=%d\n", commRank, nTotal, nLocal);  fflush(stdout);
#endif

   kbuf = (float*) malloc(2 * commSize * sizeof(float));
   tbuf = (float*) malloc(nTotal * sizeof(float));

   // need kx0, ky0 and buf from all layers
   kxy[0] = l->loc.kx0;
   kxy[1] = l->loc.ky0;

#ifdef PV_USE_MPI
   MPI_Gather(kxy, 2, MPI_FLOAT, kbuf, 2, MPI_FLOAT, 0, comm);
   MPI_Gather(ibuf, nLocal, MPI_FLOAT, tbuf, nLocal, MPI_FLOAT, 0, comm);
#else // PV_USE_MPI
   memcpy(kbuf, kxy, 2*sizeof(float));
   memcpy(tbuf, ibuf, nLocal*sizeof(float));
#endif // PV_USE_MPI

   if (commRank == 0) {
      FILE* fp;
      int result;
      PVLayerLoc loc;

      loc = l->loc;

      // copy image patch to global layer output buffer
      float * obuf = (float*) malloc(nTotal * sizeof(float));

      for (c = 0; c < commSize; c++) {
         for (kl = 0; kl < nLocal; kl++) {
            loc.kx0 = kbuf[2 * c];
            loc.ky0 = kbuf[2 * c + 1];
            int kg = globalIndexFromLocal(kl, loc);
            float* loc = tbuf + c * nLocal;
            obuf[kg] = loc[kl];
         }
      }

      // write output buffer to file

      fp = fopen(filename, "wb");
      if (fp == NULL) {
         status = -1;
         fprintf(stderr, "[0]: ERROR: gatherWriteFile: open failure on file %s\n",
               filename);
         // TODO - exit mpi gracefully
         free(obuf);
         return status;
      }
      result = fwrite(obuf, sizeof(float), nTotal, fp);
      fclose(fp);
      free(obuf);

      if (result != nTotal) {
         status = -1;
         fprintf(stderr, "[%d]: ERROR: gatherWriteFile: write result wrong = %d\n",
               commRank, result);
         return status;
      }
   }

   free(tbuf);
   free(kbuf);

   return status;
}

/**
 * @buf
 * @nItems
 * @msg
 */
int printStats(pvdata_t * buf, int nItems, char * msg)
{
   int n;
   float fMin = FLT_MAX, fMax = FLT_MIN;
   double tot = 0.0;
   char txt[128];
#ifdef _MSC_VER
   FILE * f;
#endif

   for (n = 0; n < nItems; n++) {
      tot += buf[n];

      if (buf[n] < fMin) fMin = buf[n];
      if (buf[n] > fMax) fMax = buf[n];
   }

   sprintf(txt, "%s (N=%d) Total=%f Min=%f, Avg=%f, Max=%f\n", msg, n, (float)tot, fMin,
           (float)(tot / n), fMax);

#ifdef _MSC_VER
   f = fopen( "log.txt", "a" );
   fprintf(f, txt);
   fclose(f);
#else
   printf("%s", txt);  fflush(stdout);
#endif

   return 0;
}

/**
 * @filename
 * @append
 * @I
 * @nx
 * @ny
 * @nf
 */
int pv_dump(const char * output_path, const char * filename,
            int append, pvdata_t * I, int nx, int ny, int nf)
{
   char fullpath[PV_PATH_MAX];
   int params[NUM_BIN_PARAMS];
   int status = 0;
   FILE * fp;

   int nItems = nx * ny * nf;

   sprintf(fullpath, "%s/%s.bin", output_path, filename);

   if (append) fp = fopen(fullpath, "ab");
   else        fp = fopen(fullpath, "wb");

   if (!append) {
      const int nParams = 3;
      params[0] = nParams;
      params[1] = nx;
      params[2] = ny;
      params[3] = nf;
      if ( fwrite(params, sizeof(int), nParams+1, fp) != nParams+1) status = -3;
      if (status != 0) {
         pv_log(stderr, "pv_dump: error writing params header\n");
      }
   }

   if (fp != NULL) {
      if ( fwrite(I, sizeof(pvdata_t), nItems, fp) != nItems) status = -2;
      fclose(fp);
   }
   else {
      status = -1;
      pv_log(stderr, "pv_dump: couldn't open output file %s\n", fullpath);
   }

   if (status == -2) {
      pv_log(stderr, "pv_dump: error writing %d items\n", nItems);
   }

   return status;
}

/**
 * @filename
 * @append
 * @I
 * @nx
 * @ny
 * @nf
 */
int pv_dump_sparse(const char * output_path, const char * filename,
                   int append, pvdata_t * I, int nx, int ny, int nf)
{
   char fullpath[PV_PATH_MAX];
   int params[NUM_BIN_PARAMS];
   int status = 0;
   FILE * fp;
   int k, nSpikes = 0;

   int nItems = nx * ny * nf;

   sprintf(fullpath, "%s/%s_sparse.bin", output_path, filename);

   if (append) fp = fopen(fullpath, "ab");
   else        fp = fopen(fullpath, "wb");

   if (!append) {
      int nParams = 3;
      params[0] = nx;
      params[1] = ny;
      params[2] = nf;
      if ( fwrite(&nParams, sizeof(int), 1, fp) != 1) status = -3;
      if ( fwrite(params, sizeof(int), nParams, fp) != nParams) status = -3;
      if (status != 0) {
         pv_log(stderr, "pv_dump_sparse: error writing params header\n");
      }
   }

   if (fp != NULL) {
      for (k = 0; k < nItems; k++) {
         if (I[k] > 0.0f) nSpikes++;
      }

      if ( fwrite(&nSpikes, sizeof(int), 1, fp) != 1) status = -2;

      for (k = 0; k < nItems; k++)
         if (I[k] > 0.0f) {
            if ( fwrite(&k, sizeof(int), 1, fp) != 1) status = -2;
         }

      fclose(fp);
   }
   else {
      status = -1;
      pv_log(stderr, "pv_dump_sparse: couldn't open output file %s\n", fullpath);
   }

   if (status == -2) {
      pv_log(stderr, "pv_dump_sparse: error writing data\n");
   }

   return status;
}

/**
 * @fp
 * @minVal
 * @maxVal
 * @p
 *
 * NOTES: p->nx and p->ny reflect the location inside PVPatch that information
 * is extracted from. For example, for a neuron in the padding layer, that
 * sees only a corner of the "retina" we have p->nx=1 and p->ny=1 and
 * we only write one weight (nItems=1).
 *         +++
 *         +++
 *         ++o
 * For the following neuron we have
 *         +++
 *         +++
 *         +oo
 * p->nx =2 and p->ny =1 (nItems=2) and we write 2 weights.
 * Think of p->nx and p->ny as defining the size of the neuron's
 * receptive field that receives spikes from the retina.
 */
int pv_write_patch(FILE * fp, int numTotal, float minVal, float maxVal, PVPatch * p)
{
   int i, nItems, nExtra;
   unsigned short nxny[2];

   const pvdata_t * data = p->data;

   nxny[0] = (unsigned short) p->nx;
   nxny[1] = (unsigned short) p->ny;

   nItems = (int) nxny[0] * (int) nxny[1] * (int) p->nf;

   if ( fwrite(nxny, sizeof(unsigned short), 2, fp) != 2 ) return -1;

   for (i = 0; i < nItems; i++) {
      float val = 255.0 * (data[i] - minVal) / (maxVal - minVal);
//      unsigned char c = (unsigned char) fval;
//      if ( fwrite(&c, sizeof(unsigned char), 1, fp) != 1 ) return -2;
      int c = (unsigned char) val;
      if (putc_unlocked(c, fp) == EOF) return -2;
   }

   // write leftover null characters
   nExtra = numTotal - nItems;

   for (i = 0; i < nExtra; i++) {
      int c = 0;
      if (putc_unlocked(c, fp) == EOF) return -2;
   }
   fflush(fp);

   return numTotal;
}

/**
 * @fp        file pointer
 * @nxp       size of patch in x dimension
 * @nyp       size of patch in y dimension
 * @nfp       size of patch in f dimension
 * @minVal    minimum of a patch value
 * @maxVal    maximum of a patch value
 * @p         the patch
 * @buf       a temporary buffer for patch compressed char values
 */
int pv_read_patch(FILE * fp, int nxp, int nyp, int nfp,
                  float minVal, float maxVal, PVPatch * p, unsigned char * buf)
{
   int i;
   unsigned short nxny[2];

   pvdata_t * data = p->data;

   if ( fread(nxny, sizeof(unsigned short), 2, fp) != 2 ) return -1;

   // read the patch values which are padded to full size
   int nItems = nxp * nyp * nfp;
   if ( fread(buf, sizeof(unsigned char), nItems, fp) != nItems ) return -2;

   nItems = (int) nxny[0] * (int) nxny[1] * (int) nfp;

   p->nx = (float) nxny[0];
   p->ny = (float) nxny[1];
   p->nf = nfp;

   p->sf = 1;
   p->sx = nfp;
   p->sy = (float) ( (int) p->nf * (int) p->nx );

   for (i = 0; i < nItems; i++) {
      // data are packed into chars
      float val = (float) buf[i];
      data[i] = minVal + (maxVal - minVal) * (val / 255.0);
   }

   return nItems;
}

#ifdef DONT_COMPILE
int pv_read_patch_old(FILE * fp, float nf, float minVal, float maxVal, PVPatch * p)
{
   const int bufSize = 4;
   int i, ii, nItems;
   unsigned char buf[bufSize];
   unsigned short nxny[2];

   pvdata_t * data = p->data;

   if ( fread(nxny, sizeof(unsigned short), 2, fp) != 2 ) return -1;

   nItems = (int) nxny[0] * (int) nxny[1] * (int) nf;

   p->nx = (float) nxny[0];
   p->ny = (float) nxny[1];
   p->nf = nf;

   p->sf = 1;
   p->sx = nf;
   p->sy = (float) ( (int) p->nf * (int) p->nx );

   i = 0;
   while (i < nItems) {
      if ( fread(buf, sizeof(unsigned char), bufSize, fp) != bufSize ) return -2;
      // data are packed into chars
      for (ii = 0; ii < bufSize; ii++) {
         float val = (float) buf[ii];
         data[i++] = minVal + (maxVal - minVal) * (val / 255.0);
         if (i >= nItems) break;
      }
   }

   return nItems;
}
#endif

/**
 * @filename
 * @append
 * @nx
 * @ny
 * @nf
 * @minVal
 * @maxVal
 * @numPatches
 * @patches
 */
int pv_write_patches(const char * output_path, const char * filename, int append,
                     int nxp, int nyp, int nfp, float minVal, float maxVal,
                     int numPatches, PVPatch ** patches)
{
   char fullpath[PV_PATH_MAX];
   int params[NUM_WEIGHT_PARAMS];
   int i, status = 0;
   FILE * fp;

   assert(numPatches > 0);

   // write maxVal as an integer so round up
   const float newMaxVal = ceilf(maxVal);

   sprintf(fullpath, "%s/%s.bin", output_path, filename);

   if (append) fp = fopen(fullpath, "ab");
   else        fp = fopen(fullpath, "wb");

   if (fp == NULL) {
      pv_log(stderr, "pv_dump_patches: couldn't open output file %s\n", fullpath);
      return -1;
   }

   if (!append) {
      const int nParams = NUM_WEIGHT_PARAMS;
      assert(nParams == MIN_BIN_PARAMS + 3);
      params[INDEX_HEADER_SIZE] = nParams * sizeof(int);
      params[INDEX_NUM_PARAMS]  = nParams;
      params[INDEX_FILE_TYPE]   = PV_WEIGHTS_FILE_TYPE;
      params[INDEX_NX] = nxp;
      params[INDEX_NY] = nyp;
      params[INDEX_NF] = nfp;
      params[MIN_BIN_PARAMS + 0] = (int) minVal;
      params[MIN_BIN_PARAMS + 1] = (int) newMaxVal;
      params[MIN_BIN_PARAMS + 2] = numPatches;
      if ( fwrite(params, sizeof(int), nParams, fp) != nParams ) status = -3;
      if (status != 0) {
         pv_log(stderr, "pv_dump_patches: error writing params header\n");
         return status;
      }
   }

   // numPatches - each neuron has a patch; these are neurons that live in the extended layer
   for (i = 0; i < numPatches; i++) {
      int numItems = pv_write_patch(fp, nxp*nyp*nfp, minVal, newMaxVal, patches[i]);
      if (numItems < 0) {
         status = numItems;
         pv_log(stderr, "pv_write_patches: error writing patch %d\n", i);
         break;
      }
   }
   fclose(fp);

   return status;
}

/**
 * @fp
 * @nf
 * @minVal
 * @maxVal
 * @patches
 * @numPatches
 */
int pv_read_patches(FILE * fp, int nxp, int nyp, int nfp, float minVal, float maxVal,
                    int numPatches, PVPatch ** patches)
{
   int i, status = 0;

   const int total = nxp * nyp * nfp;

   unsigned char * buf = (unsigned char *) malloc(total * sizeof(unsigned char));
   assert(buf != NULL);

   for (i = 0; i < numPatches; i++) {
      int numItems = pv_read_patch(fp, nxp, nyp, nfp, minVal, maxVal, patches[i], buf);
      if (numItems < 0) {
         status = numItems;
         pv_log(stderr, "pv_read_patches: error reading patch %d\n", i);
         break;
      }
   }
   free(buf);

   return status;
}

/**
 * @fp
 * @numParams
 * @params
 */
int pv_read_binary_params(FILE * fp, int numParams, int params[])
{
   if (numParams > NUM_BIN_PARAMS) {
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
 * Read a PV binary record.  This is a three dimensional array [ny][nx][nf].
 * @fp a FILE pointer to the file
 * @buf a buffer containing the record on return
 * @numItems
 * returns the number of items read (if 0, likely is end of file)
 */
size_t pv_read_binary_record(FILE * fp, pvdata_t * buf, int numItems)
{
   return fread(buf, sizeof(pvdata_t), numItems, fp);
}

/**
 * @filename
 * @cube
 * @nx
 * @ny
 * @nf
 */
int pv_tiff_write_cube(const char * filename, PVLayerCube * cube, int nx, int ny, int nf)
{
   int f, i, j, k;
   long nextLoc;

   float scale = 1.0;
   float max = -1.0e99;
   float min =  1.0e99;

   const int sx = nf;
   const int sy = nf*nx;
   const int sf = 1;

   pvdata_t * buf = (pvdata_t *) malloc(nx * ny * sizeof(pvdata_t));

   FILE * fd = fopen(filename, "wb");
   if (fd == NULL) {
      fprintf(stderr, "pv_tiff_write_cube: ERROR opening file %s\n", filename);
      return 1;
   }

   assert(nx*ny*nf == cube->numItems);
   for (k = 0; k < cube->numItems; k++) {
      float val = cube->data[k];
      if (val < min) min = val;
      if (val > max) max = val;
   }
   //   scale = 1.0 / (max - min);

   if (min < 0.0 || min > 1.0) {
      fprintf(stderr, "[ ]: pv_tiff_write_cube: mininum value out of bounds=%f\n", min);
   }
   if (max < 0.0 || max > 1.0) {
      fprintf(stderr, "[ ]: pv_tiff_write_cube: maximum value out of bounds=%f\n", max);
   }

   min = 0.0;
   max = 1.0;
   scale = 1.0;

   tiff_write_header(fd, &nextLoc);

   for (f = 0; f < nf; f++) {
      k = 0;
      for (j = 0; j < ny; j++) {
         for (i = 0; i < nx; i++) {
            float val = cube->data[i*sx + j*sy + f*sf];
            buf[k++] = scale * (val - min);
         }
      }

      tiff_write_ifd(fd, &nextLoc, nx, ny);
      tiff_write_image(fd, buf, nx, ny);
   }

   tiff_write_finish(fd, nextLoc);
   fclose(fd);
   free(buf);

   return 0;
}

/**
 * @fd
 * @patch
 */
int pv_tiff_write_patch(FILE * fd, PVPatch * patch)
{
   int f, i, j, k;
   long nextLoc;

   const int nx = (int) patch->nx;
   const int ny = (int) patch->ny;
   const int nf = (int) patch->nf;

   const int sx = (int) patch->sx;  assert(sx == nf);
   const int sy = (int) patch->sy;  assert(sy == nf*nx);
   const int sf = (int) patch->sf;  assert(sf == 1);

   float * buf = (float *) malloc(nx * ny * sizeof(float));

   tiff_write_header(fd, &nextLoc);

   for (f = 0; f < nf; f++) {
      float scale = 1.0;
      float max = -1.0e99;
      float min =  1.0e99;

      for (j = 0; j < ny; j++) {
         for (i = 0; i < nx; i++) {
            float val = patch->data[i*sx + j*sy + f*sf];
            if (val < min) min = val;
            if (val > max) max = val;
         }
      }

      k = 0;

      // assume this is a weight patch (need scaling type)
      min = 0.0;

      if (min == max) {
         for (j = 0; j < ny; j++) {
            for (i = 0; i < nx; i++) {
               // if there is only one value, just use whatever it is
               buf[k++] = patch->data[i*sx + j*sy + f*sf];
            }
         }
      }
      else {
         scale = 1.0 / (max - 0.0);
         for (j = 0; j < ny; j++) {
            for (i = 0; i < nx; i++) {
               float val = patch->data[i*sx + j*sy + f*sf];
               buf[k++] = scale * (val - min);
            }
         }
      }

      tiff_write_ifd(fd, &nextLoc, nx, ny);
      tiff_write_image(fd, buf, nx, ny);
   }

   tiff_write_finish(fd, nextLoc);
   free(buf);

   return 0;
}

/**
 * @fd
 * @patch
 */
int pv_text_write_patch(FILE * fd, PVPatch * patch)
{
   int f, i, j;

   const int nx = (int) patch->nx;
   const int ny = (int) patch->ny;
   const int nf = (int) patch->nf;

   const int sx = (int) patch->sx;  assert(sx == nf);
   const int sy = (int) patch->sy;  //assert(sy == nf*nx);
   const int sf = (int) patch->sf;  assert(sf == 1);

   assert(fd != NULL);

   for (f = 0; f < nf; f++) {
      for (j = 0; j < ny; j++) {
         for (i = 0; i < nx; i++) {
            fprintf(fd, "%7.5f ", patch->data[i*sx + j*sy + f*sf]);
         }
         fprintf(fd, "\n");
      }
      fprintf(fd, "\n");
   }

   return 0;
}
