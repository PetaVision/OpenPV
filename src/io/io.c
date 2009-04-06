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

static int pv_getopt_int(int argc, char * argv[], char * opt, int *   iVal);
static int pv_getopt_str(int argc, char * argv[], char * opt, char ** sVal);

void usage()
{
   printf("\nUsage:\n");
   printf(" -n <number of timesteps>\n");
   printf(" -i <input filename>\n");
   printf(" -p <parameters filename>\n");
   printf(" -t <number of threads for shared memory parallelism>\n");
   printf("\nA good test is:\n");
   printf(" ./Debug/pv -n 100 -p src/input/params.pv -i tests/input/horizontal-lines.tif\n");
   printf("\nThen check results in Octave/MATLAB using analysis/pv_analyze.m\n\n");
}

int parse_options(int argc, char * argv[], char ** input_file,
                  char ** param_file, int * n_time_steps, int * threads)
{
   if (argc < 3) {
      usage();
      return -1;
   }

   pv_getopt_int(argc, argv, "-n", n_time_steps);
   pv_getopt_int(argc, argv, "-t", threads);
   pv_getopt_str(argc, argv, "-i", input_file);
   pv_getopt_str(argc, argv, "-p", param_file);

   return 0;
}

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


// For MATLAB use, just save a number with name as comment
#define LOGINTPARM(paramfile, which) \
    fprintf(paramfile, "%d %% %s\n", which, #which)
#define LOGFPARM(paramfile, which) \
    fprintf(paramfile, "%f %% %s\n", which, #which)
#define LOGSPARM(paramfile, which) \
    fprintf(paramfile, #which "=%s\n", which)

void log_parameters(int n_time_steps, char *input_filename)
{
   // Write our runtime parameters to a logfile, so that
   // anyone using using the output can extract this
   // information for analysis, rather than hardcoding.
   char param_filename[PV_PATH_MAX];
   FILE * paramfile;
   sprintf(param_filename, OUTPUT_PATH "/" PARAMS_FILE);

   paramfile = fopen(param_filename, "w");

   if (paramfile == NULL) {
      printf("Couldn't open parameter logfile %s. Aborting.\n", param_filename);
      exit(-1);
   }

   // TODO: need to get these per-layer
   LOGINTPARM(paramfile,NX);
   LOGINTPARM(paramfile,NY);
   LOGINTPARM(paramfile,NO);
   LOGINTPARM(paramfile,NK);
   LOGINTPARM(paramfile,(NX*NY*NO*NK));
   LOGFPARM(paramfile,DTH);
   LOGINTPARM(paramfile,n_time_steps);
   //LOGSPARM(paramfile,input_filename); //causes MATLAB problems
   fclose(paramfile);
}

#define TIFF_FILE_TYPE    1
#define BINARY_FILE_TYPE  2

static int filetype(const char * filename)
{
   int n = strlen(filename);
   if (strncmp(&filename[n-4], ".tif", 4) == 0) return TIFF_FILE_TYPE;
   if (strncmp(&filename[n-4], ".bin", 4) == 0) return BINARY_FILE_TYPE;
   return 0;
}

static int centerImage(float * V, int nx0, int ny0, int nx, int ny)
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


int readFile(const char * filename, float * buf, int * nx, int * ny)
{
   int result, nItems;
   int err = 0;
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
         err = 1;
      }
      else {
#ifdef DEBUG_OUTPUT
         printf("[ ]: readFile: Successfully read %d items from %s\n", nItems, filename);  fflush(stdout);
#endif
      }
   }

   return err;
}

int scatterReadFile(const char* filename, PVLayer* l, float* buf, MPI_Comm comm)
{
   int kl, kg, err = 0;
   int nx, ny;

   int nTotal = l->loc.nxGlobal * l->loc.nyGlobal;
   int nLocal = l->loc.nx * l->loc.ny;

   float* tmp = (float*) malloc(nTotal * sizeof(float));
   if (tmp == NULL) return -1;

   if (l->columnId == 0) {
      nx = l->loc.nxGlobal;
      ny = l->loc.nyGlobal;
      err = readFile(filename, tmp, &nx, &ny);
      if (err == 0) {
         assert(nx <= l->loc.nxGlobal);
         assert(ny <= l->loc.nyGlobal);
         if (nx < l->loc.nxGlobal || ny < l->loc.nyGlobal) {
            err = centerImage(tmp, nx, ny, l->loc.nxGlobal, l->loc.nyGlobal);
         }
      }
   }

   if (err != 0) return err;

   // everyone gets a copy of the entire input
   MPI_Bcast(tmp, nTotal, MPI_FLOAT, 0, comm);

   // everyone copies only their local patch of input
   for (kl = 0; kl < nLocal; kl++) {
      kg = globalIndexFromLocal(kl, l->loc, l->numFeatures);
      buf[kl] = tmp[kg];
   }
   if (tmp) free(tmp);

   return err;
}

int gatherWriteFile(const char* filename, PVLayer* l, float* ibuf, MPI_Comm comm)
{
   int commRank, commSize, err = 0;
   int c, kl, nLocal, nTotal;
   float kxy[2], *kbuf, *tbuf;

   // WARNING, this assumes layers in separate hypercolumns have the same (nx,ny)

   MPI_Comm_rank(comm, &commRank);
   MPI_Comm_size(comm, &commSize);
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

   MPI_Gather(kxy, 2, MPI_FLOAT, kbuf, 2, MPI_FLOAT, 0, comm);
   MPI_Gather(ibuf, nLocal, MPI_FLOAT, tbuf, nLocal, MPI_FLOAT, 0, comm);

   if (commRank == 0) {
      FILE* fp;
      int result;
      PVLayerLoc loc;

      loc = l->loc;

      // copy image patch to global layer output buffer
      float* obuf = (float*) malloc(nTotal * sizeof(float));

      for (c = 0; c < commSize; c++) {
         for (kl = 0; kl < nLocal; kl++) {
            loc.kx0 = kbuf[2 * c];
            loc.ky0 = kbuf[2 * c + 1];
            int kg = globalIndexFromLocal(kl, loc, l->numFeatures);
            float* loc = tbuf + c * nLocal;
            obuf[kg] = loc[kl];
         }
      }

      // write output buffer to file

      fp = fopen(filename, "wb");
      if (fp == NULL) {
         err = -1;
         fprintf(stderr, "[0]: ERROR: gatherWriteFile: open failure on file %s\n",
               filename);
         // TODO - exit mpi gracefully
         free(obuf);
         return err;
      }
      result = fwrite(obuf, sizeof(float), nTotal, fp);
      fclose(fp);
      free(obuf);

      if (result != nTotal) {
         err = -1;
         fprintf(stderr, "[%d]: ERROR: gatherWriteFile: write result wrong = %d\n",
               commRank, result);
         return err;
      }
   }

   free(tbuf);
   free(kbuf);

   return err;
}

int printStats(pvdata_t * buf, int nItems, char * msg)
{
   int n;
   float fMin = FLT_MAX, fMax = FLT_MIN;
   double tot = 0.0;
   char txt[128];
#ifdef _MSC_VER
   FILE *f;
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
   printf(txt);  fflush(stdout);
#endif

   return 0;
}

int pv_dump(char * filename, int append, pvdata_t * I, int nx, int ny, int nf)
{
   char fullpath[PV_PATH_MAX];
   int err = 0;
   FILE* fd;

   int nItems = nx * ny * nf;

   sprintf(fullpath, "%s/%s.bin", OUTPUT_PATH, filename);

   if (append) fd = fopen(fullpath, "ab");
   else        fd = fopen(fullpath, "wb");

   if (!append) {
      int nParams = 3;
      int params[3];
      params[0] = nx;
      params[1] = ny;
      params[2] = nf;
      if ( fwrite(&nParams, sizeof(int), 1, fd) != 1) err = -3;
      if ( fwrite(params, sizeof(int), nParams, fd) != nParams) err = -3;
      if (err != 0) {
         pv_log(stderr, "pv_dump: error writing params header\n");
      }
   }

   if (fd != NULL) {
      if ( fwrite(I, sizeof(pvdata_t), nItems, fd) != nItems) err = -2;
      fclose(fd);
   }
   else {
      err = -1;
      pv_log(stderr, "pv_dump: couldn't open output file %s\n", fullpath);
   }

   if (err == -2) {
      pv_log(stderr, "pv_dump: error writing %d items\n", nItems);
   }

   return err;
}

int pv_dump_sparse(char * filename, int append, pvdata_t * I, int nx, int ny, int nf)
{
   char fullpath[PV_PATH_MAX];
   int err = 0;
   FILE * fd;
   float m;
   float nSpikes = 0;

   int nItems = nx * ny * nf;

   sprintf(fullpath, "%s/%s_sparse.bin", OUTPUT_PATH, filename);

   if (append) fd = fopen(fullpath, "ab");
   else        fd = fopen(fullpath, "wb");

   if (!append) {
      int nParams = 3;
      int params[3];
      params[0] = nx;
      params[1] = ny;
      params[2] = nf;
      if ( fwrite(&nParams, sizeof(int), 1, fd) != 1) err = -3;
      if ( fwrite(params, sizeof(int), nParams, fd) != nParams) err = -3;
      if (err != 0) {
         pv_log(stderr, "pv_dump_sparse: error writing params header\n");
      }
   }

   if (fd != NULL) {
      for (m = 0; m < nItems; m++) {
         if (I[(int) m]) nSpikes++;
      }

      if ( fwrite(&nSpikes, sizeof(float), 1, fd) != 1) err = -2;

      for (m = 0; m < nItems; m++)
         if (I[(int) m]) {
            if ( fwrite(&m, sizeof(float), 1, fd) != 1) err = -2;
         }

      fclose(fd);
   }
   else {
      err = -1;
      pv_log(stderr, "pv_dump_sparse: couldn't open output file %s\n", fullpath);
   }

   if (err == -2) {
      pv_log(stderr, "pv_dump_sparse: error writing data\n");
   }

   return err;
}

FILE * pv_open_binary(char * filename, int * nx, int * ny, int * nf)
{
   int nParams;
   int * params;

   FILE * fd = fopen(filename, "rb");
   if (fd == NULL) return NULL;

   if ( fread(&nParams, sizeof(int), 1, fd) != 1 ) {
      fclose(fd);
      return NULL;
   }

   params = (int *) malloc(nParams * sizeof(int));
   if (params == NULL) return NULL;

   if ( fread(params, sizeof(int), nParams, fd) != nParams ) {
      fclose(fd);
      return NULL;
   }

   *nx = params[0];
   *ny = params[1];
   *nf = params[2];

   return fd;
}

int pv_close_binary(FILE * fd)
{
   return fclose(fd);
}

int pv_read_binary_record(FILE * fd, pvdata_t * buf, int nItems)
{
   if ( fread(buf, sizeof(pvdata_t), nItems, fd) != nItems ) {
      return -1;
   }
   return 0;
}

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
      fprintf(stderr, "pv_tiff_write_patch: ERROR opening file %s\n", filename);
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

int pv_tiff_write_patch(const char * filename, PVPatch * patch)
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

   FILE * fd = fopen(filename, "wb");
   if (fd == NULL) {
      fprintf(stderr, "pv_tiff_write_patch: ERROR opening file %s\n", filename);
      return 1;
   }

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
   fclose(fd);
   free(buf);

   return 0;
}

int pv_text_write_patch(const char * filename, PVPatch * patch)
{
   int f, i, j;

   const int nx = (int) patch->nx;
   const int ny = (int) patch->ny;
   const int nf = (int) patch->nf;

   const int sx = (int) patch->sx;  assert(sx == nf);
   const int sy = (int) patch->sy;  assert(sy == nf*nx);
   const int sf = (int) patch->sf;  assert(sf == 1);

   FILE * fd = fopen(filename, "w");
   if (fd == NULL) {
      fprintf(stderr, "pv_text_write_patch: ERROR opening file %s\n", filename);
      return 1;
   }

   for (f = 0; f < nf; f++) {
      for (j = 0; j < ny; j++) {
         for (i = 0; i < nx; i++) {
            fprintf(fd, "%5.3f ", patch->data[i*sx + j*sy + f*sf]);
         }
         fprintf(fd, "\n");
      }
      fprintf(fd, "\n");
   }
   fclose(fd);

   return 0;
}
