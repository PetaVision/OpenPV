/*
 * fileio.cpp
 *
 *  Created on: Oct 21, 2009
 *      Author: Craig Rasmussen
 */

#include "fileio.hpp"
#include "../layers/HyPerLayer.hpp"

#include <assert.h>

#undef DEBUG_OUTPUT

namespace PV {

// timeToParams and timeFromParams use memcpy instead of casting pointers
// because casting pointers violates strict aliasing.
void timeToParams(double time, void * params)
{
   memcpy(params, &time, sizeof(double));
}

double timeFromParams(void * params)
{
   double x;
   memcpy(&x, params, sizeof(double));
   return x;
}

size_t pv_sizeof(int datatype)
{
   if (datatype == PV_INT_TYPE) {
      return sizeof(int);
   }
   if (datatype == PV_FLOAT_TYPE) {
      return sizeof(float);
   }
   if (datatype == PV_BYTE_TYPE) {
      return sizeof(unsigned char);
   }

   // shouldn't arrive here
   assert(false);
   return 0;
}

/**
 * Returns the size of a patch when read or written. The size includes the size
 * of the (nxp,nyp) patch header.
 */
size_t pv_sizeof_patch(int count, int datatype)
{
   return ( 2*sizeof(unsigned short) + sizeof(unsigned int) + count*pv_sizeof(datatype) );
   // return ( 2*sizeof(unsigned short) + count*pv_sizeof(datatype) );
}

PV_Stream * PV_fopen(const char * path, const char * mode) {
   int fopencounts = 0;
   PV_Stream * streampointer = NULL;
   FILE * fp = NULL;
   while (fp == NULL) {
      errno = 0;
      fp = fopen(path, mode);
      if (fp != NULL) break;
      fopencounts++;
      fprintf(stderr, "fopen failure on attempt %d: %s\n", fopencounts, strerror(errno));
      if (fopencounts < MAX_FILESYSTEMCALL_TRIES) {
         sleep(1);
      }
      else {
         break;
      }
   }
   if (fp == NULL) {
      fprintf(stderr, "PV_fopen error for \"%s\": MAX_FILESYSTEMCALL_TRIES = %d exceeded\n", path, MAX_FILESYSTEMCALL_TRIES);
   }
   else {
      streampointer = (PV_Stream *) calloc(1, sizeof(PV_Stream));
      if (streampointer != NULL) {
         streampointer->name = strdup(path);
         streampointer->fp = fp;
         streampointer->isfile = 1;
      }
      else {
         fprintf(stderr, "PV_fopen failure for \"%s\": %s\n", path, strerror(errno));
         fclose(fp);
      }
   }
   return streampointer;
}

long int PV_ftell(PV_Stream * pvstream) {
   int ftellcounts = 0;
   long filepos = -1;
   while (filepos < 0) {
      errno = 0;
      filepos = ftell(pvstream->fp);
      if (filepos >= 0) break;
      ftellcounts++;
      fprintf(stderr, "ftell failure for \"%s\" on attempt %d: %s\n", pvstream->name, ftellcounts, strerror(errno));
      if (ftellcounts < MAX_FILESYSTEMCALL_TRIES) {
         sleep(1);
      }
      else {
         break;
      }
   }
   if (filepos<0) {
      fprintf(stderr, "PV_ftell failure for \"%s\": MAX_FILESYSTEMCALL_TRIES = %d exceeded\n", pvstream->name, MAX_FILESYSTEMCALL_TRIES);
   }
   return filepos;
}

int PV_fseek(PV_Stream * pvstream, long offset, int whence) {
   int fseekcounts = 0;
   int fseekstatus = -1;
   while (fseekstatus != 0) {
      errno = 0;
      fseekstatus = fseek(pvstream->fp, offset, whence);
      if (fseekstatus==0) break;
      fseekcounts++;
      fprintf(stderr, "fseek failure for \"%s\" on attempt %d\n", pvstream->name, fseekcounts);
      if (fseekcounts<MAX_FILESYSTEMCALL_TRIES) {
         sleep(1);
      }
      else {
         break;
      }
   }
   if (fseekstatus!=0) {
      fprintf(stderr, "PV_fseek failure for \"%s\": MAX_FILESYSTEMCALL_TRIES = %d exceeded\n", pvstream->name, MAX_FILESYSTEMCALL_TRIES);
   }
   return fseekstatus;
}

size_t PV_fwrite(const void * RESTRICT ptr, size_t size, size_t nitems, PV_Stream * RESTRICT pvstream) {
   int fwritecounts = 0;
   size_t fwritten = nitems - 1;
   while (fwritten != nitems) {
      long int fpos = PV_ftell(pvstream);
      if (fpos<0) {
         fprintf(stderr, "PV_fwrite error: unable to determine file position of \"%s\".  Fatal error\n", pvstream->name);
         exit(EXIT_FAILURE);
      }
      fwritten = fwrite(ptr, size, nitems, pvstream->fp);
      if (fwritten == nitems) {
    	  return fwritten;
      }
      fwritecounts++;
      if (fwritecounts<MAX_FILESYSTEMCALL_TRIES) {
         fprintf(stderr, "fwrite failure for \"%s\" on attempt %d.  Attempting to return to original position\n", pvstream->name, fwritecounts);
         sleep(1);
         int fseekstatus = PV_fseek(pvstream, fpos, SEEK_SET);
         if (fseekstatus!=0) {
            fprintf(stderr, "PV_fwrite error: unable to return to original position after failed fwrite call for \"%s\".  Fatal error.\n", pvstream->name);
            exit(EXIT_FAILURE);
         }
      }
      else {
    	 fprintf(stderr, "PV_fwrite failure for \"%s\": MAX_FILESYSTEMCALL_TRIES = %d exceeded\n", pvstream->name, MAX_FILESYSTEMCALL_TRIES);
         assert(fwritten == nitems);
      }
   }
   return fwritten;
}

// TODO PV_fread

int PV_fclose(PV_Stream * pvstream) {
   int status = PV_SUCCESS;
   if (pvstream) {
      if (pvstream->fp && pvstream->isfile) {
         status = fclose(pvstream->fp);
         if (status!=0) {
            fprintf(stderr, "fclose failure for \"%s\"", pvstream->name);
         }
      }
      free(pvstream->name);
      free(pvstream); pvstream = NULL;
   }
   return status;
}

PV_Stream * PV_stdout() {
   PV_Stream * pvstream = (PV_Stream *) calloc(1, sizeof(PV_Stream));
   if (pvstream != NULL) {
      pvstream->name = strdup("stdout");
      pvstream->fp = stdout;
      pvstream->isfile = 0;
   }
   else {
      fprintf(stderr, "PV_stdout failure: %s\n", strerror(errno));
   }
   return pvstream;
}

/**
 * Copy patches into an unsigned char buffer
 */
int pvp_copy_patches(unsigned char * buf, PVPatch ** patches, pvdata_t * dataStart, int numDataPatches,
                     int nxp, int nyp, int nfp, float minVal, float maxVal,
                     bool compressed=true) {
   // Copies data from patches and dataStart to buf.
   // buf should point to a buffer of size numDataPatches*pv_sizeof_patch(numweights,datatype) characters,
   // where numweights is nxp*nyp*nfp; and datatype is PV_FLOAT_TYPE for uncompressed weights and PV_BYTE_TYPE for compressed.
   // The calling routine is responsible for allocating and for freeing buf.
   // For PVP_KERNEL_FILE_TYPE, patches should be null.  For PVP_WGT_FILE_TYPE, patches should point to the weight patches for one arbor.
   // Each patch takes up pv_sizeof_patch(numweights,datatype) chars in buf --- even for shrunken patches.
   // The values in patches[k] will be written to &buf[k].  (For PVP_KERNEL_FILE_TYPE, the values are always nx=nxp, ny=nyp, offset=0).
   // The numweights values from dataStart+k*numweights will be copied to buf starting at &buf[k*(numweights*datasize+2*sizeof(short)+sizeof(int))].
   unsigned char * cptr = buf;
   const int patchsize = nxp * nyp * nfp;
   int nx = nxp;
   int ny = nyp;
   int offset = 0;
   for (int k = 0; k < numDataPatches; k++) {
      if( patches != NULL ) {
         PVPatch * p = patches[k];
         nx = p->nx;
         ny = p->ny;
         offset = p->offset;
      }
      // const pvdata_t * data = p->data;
      const pvdata_t * data = dataStart + k*patchsize; // + offset; // Don't include offset as the entire patch will be copied

      // const int sxp = nfp; //p->sx;
      // const int syp = nfp * nxp; //p->sy;
      // const int sfp = 1; //p->sf;

      unsigned short * nxny = (unsigned short *) cptr;
      nxny[0] = (unsigned short) nx;
      nxny[1] = (unsigned short) ny;
      cptr += 2 * sizeof(unsigned short);

      unsigned int * offsetptr = (unsigned int *) cptr;
      *offsetptr = offset;
      cptr += sizeof(unsigned int);

      if( compressed ) {
         for (int k = 0; k < patchsize; k++) {
            *cptr++ = compressWeight(data[k], minVal, maxVal);
         }
      }
      else {
         float * fptr = (float *) cptr;
         for (int k = 0; k < patchsize; k++) {
            *fptr++ = data[k];
         }
         cptr = (unsigned char *) fptr;
      }
   }

   return PV_SUCCESS;
}

/**
 * Set patches given an unsigned char input buffer
 */
int pvp_set_patches(unsigned char * buf, PVPatch ** patches, pvdata_t * dataStart, int numDataPatches,
                    int nxp, int nyp, int nfp, float minVal, float maxVal,
                    bool compress=true,
                    bool shmget_owner=true, bool shmget_flag=false)
{
   // Copies data from patches and dataStart to buf.
   // buf should point to a buffer of size numDataPatches*pv_sizeof_patch(numweights,datatype) characters,
   // where numweights is nxp*nyp*nfp; and datatype is PV_FLOAT_TYPE for uncompressed weights and PV_BYTE_TYPE for compressed.
   // The calling routine is responsible for allocating and for freeing buf.
   // For PVP_KERNEL_FILE_TYPE, patches should be null.  For PVP_WGT_FILE_TYPE, patches should point to the weight patches for one arbor.
   // Each patch takes up pv_sizeof_patch(numweights,datatype) chars in buf --- even for shrunken patches.
   // The values in patches[k] are compared to &buf[k] in an assert statement; they should be equal.  (For PVP_KERNEL_FILE_TYPE, these tests are skipped).
   // The numweights values from dataStart+k*numweights will be copied from buf starting at &buf[k*(numweights*datasize+2*sizeof(short)+sizeof(int))].
   unsigned char * cptr = buf;

   // const int sfp = 1;
   // const int sxp = nfp;
   // const int syp = nfp * nxp;
   const int patchsize = nxp * nyp * nfp; // syp * nyp;

   unsigned short nx = nxp;
   unsigned short ny = nyp;
   unsigned int offset = 0;
   for (int k = 0; k < numDataPatches; k++) {
      if( patches != NULL ) {
         PVPatch * p = patches[k];
         nx = p->nx;
         ny = p->ny;
         offset = p->offset;
      }
      // pvdata_t * data = p->data;
      pvdata_t * data = dataStart + k*patchsize; // + offset; // Don't include offset as entire patch will be read from buf
#ifdef USE_SHMGET
      volatile pvdata_t * data_volatile = dataStart + k*patchsize;
#endif

      unsigned short * nxny = (unsigned short *) cptr;
      nx = nxny[0];
      ny = nxny[1];
      cptr += 2 * sizeof(unsigned short);

      unsigned int * offsetptr = (unsigned int *) cptr;
      offset = *offsetptr;
      cptr += sizeof(unsigned int);
      assert( patches==NULL || (patches[k]->nx==nx && patches[k]->ny==ny && patches[k]->offset == offset) );

      if( compress ) {
         for (int k = 0; k < patchsize; k++) {
            // values in buf are packed into chars
#ifndef USE_SHMGET
            data[k] += uncompressWeight(*cptr++, minVal, maxVal);
#else
            if (!shmget_flag){
               data[k] += uncompressWeight(*cptr++, minVal, maxVal);
            }
            else{
               data_volatile[k] += uncompressWeight(*cptr++, minVal, maxVal);
            }
#endif
         }
      }
      else {
         float * fptr = (float *) cptr;
         for (int k = 0; k < patchsize; k++) {
#ifndef USE_SHMGET
            data[k] += *fptr++;
#else
            if (!shmget_flag){
               data[k] += *fptr++;
            }
            else{
               data_volatile[k] += *fptr++;
            }
#endif
         }
         cptr = (unsigned char *) fptr;
      }
   }

   return PV_SUCCESS;
}

PV_Stream * pvp_open_read_file(const char * filename, Communicator * comm)
{
   PV_Stream * pvstream = NULL;
   if (comm->commRank() == 0) {
      pvstream = PV_fopen(filename, "rb");
      if (pvstream==NULL) {
        fprintf(stderr, "pvp_open_read_file failed for \"%s\": %s\n", filename, strerror(errno));
      }
   }
   return pvstream;
}

PV_Stream * pvp_open_write_file(const char * filename, Communicator * comm, bool append)
{
   PV_Stream * pvstream = NULL;
   if (comm->commRank() == 0) {
      bool rwmode = false;
      if (append) {
         // If the file exists, need to use read/write mode (r+) since we'll navigate back to the header to update nbands
         // If the file does not exist, mode r+ gives an error
         struct stat filestat;
         int status = stat(filename, &filestat);
         if (status==0) {
            rwmode = true;
         }
         else {
            if (errno==ENOENT) {
               fprintf(stderr, "Warning: activity file \"%s\" does not exist.  File will be created\n", filename);
               rwmode = false;
            }
            else {
               fprintf(stderr, "Error opening activity file \"%s\": %s", filename, strerror(errno));
               abort();
            }
         }
      }
      if (rwmode) {
         pvstream = PV_fopen(filename, "r+b");
         if (pvstream==NULL) {
            fprintf(stderr, "pvp_open_write_file failed for \"%s\": %s\n", filename, strerror(errno));
         }
      }
      else {
         pvstream = PV_fopen(filename, "wb");
         if (pvstream==NULL) {
            fprintf(stderr, "pvp_open_write_file failed for \"%s\": %s\n", filename, strerror(errno));
         }
      }
   }
   return pvstream;
}

int pvp_close_file(PV_Stream * pvstream, Communicator * comm)
{
   int status = PV_SUCCESS;
   if (comm->commRank()==0) {
      status = PV_fclose(pvstream);
   }
   return status;
}

int pvp_check_file_header(Communicator * comm, const PVLayerLoc * loc, int params[], int numParams)
{
   int status = PV_SUCCESS;
   int tmp_status = PV_SUCCESS;

   int nxProcs = comm->numCommColumns();
   int nyProcs = comm->numCommRows();

   if (loc->nx       != params[INDEX_NX])        {status = PV_FAILURE; tmp_status = INDEX_NX;}
   if (tmp_status == INDEX_NX) {
      if (params[INDEX_FILE_TYPE] != PVP_KERNEL_FILE_TYPE){
         fprintf(stderr, "nx = %d != params[%d]==%d ", loc->nx, INDEX_NX, params[INDEX_NX]);
         fprintf(stderr, "\n");
      }
      else {
         status = PV_SUCCESS; // kernels can be used regardless of layer size
      }
   }
   if (loc->ny       != params[INDEX_NY])        {status = PV_FAILURE; tmp_status = INDEX_NY;}
   if (tmp_status == INDEX_NY) {
      if (params[INDEX_FILE_TYPE] != PVP_KERNEL_FILE_TYPE){
         fprintf(stderr, "ny = %d != params[%d]==%d ", loc->ny, INDEX_NY, params[INDEX_NY]);
         fprintf(stderr, "\n");
      }
      else {
         status = PV_SUCCESS; // kernels can be used regardless of layer size
      }
   }
   if (loc->nf != params[INDEX_NF]) {status = PV_FAILURE; tmp_status = INDEX_NF;}
   if (tmp_status == INDEX_NF) {
         fprintf(stderr, "nBands = %d != params[%d]==%d ", loc->nf, INDEX_NF, params[INDEX_NF]);
         fprintf(stderr, "\n");
   }
   if (loc->nxGlobal != params[INDEX_NX_GLOBAL]) {status = PV_FAILURE; tmp_status = INDEX_NX_GLOBAL;}
   if (tmp_status == INDEX_NX_GLOBAL) {
      if (params[INDEX_FILE_TYPE] != PVP_KERNEL_FILE_TYPE){
         fprintf(stderr, "nxGlobal = %d != params[%d]==%d ", loc->nxGlobal, INDEX_NX_GLOBAL, params[INDEX_NX_GLOBAL]);
         fprintf(stderr, "\n");
      }
      else {
         status = PV_SUCCESS; // kernels can be used regardless of layer size
      }
   }
   if (loc->nyGlobal != params[INDEX_NY_GLOBAL]) {status = PV_FAILURE; tmp_status = INDEX_NY_GLOBAL;}
   if (tmp_status == INDEX_NY_GLOBAL) {
      if (params[INDEX_FILE_TYPE] != PVP_KERNEL_FILE_TYPE){
         fprintf(stderr, "nyGlobal = %d != params[%d]==%d ", loc->nyGlobal, INDEX_NY_GLOBAL, params[INDEX_NY_GLOBAL]);
         fprintf(stderr, "\n");
      }
      else {
         status = PV_SUCCESS; // kernels can be used regardless of layer size
      }
   }
   if (nxProcs != params[INDEX_NX_PROCS]) {status = PV_FAILURE; tmp_status = INDEX_NX_PROCS;}
   if (tmp_status == INDEX_NX_PROCS) {
      if (params[INDEX_FILE_TYPE] != PVP_KERNEL_FILE_TYPE){
         fprintf(stderr, "nxProcs = %d != params[%d]==%d ", nxProcs, INDEX_NX_PROCS, params[INDEX_NX_PROCS]);
         fprintf(stderr, "\n");
      }
      else {
         status = PV_SUCCESS; // kernels can be used regardless of num procs
      }
   }
   if (nyProcs != params[INDEX_NY_PROCS]) {status = PV_FAILURE; tmp_status = INDEX_NY_PROCS;}
   if (tmp_status == INDEX_NY_PROCS) {
      if (params[INDEX_FILE_TYPE] != PVP_KERNEL_FILE_TYPE){
         fprintf(stderr, "nyProcs = %d != params[%d]==%d ", nyProcs, INDEX_NY_PROCS, params[INDEX_NY_PROCS]);
         fprintf(stderr, "\n");
      }
      else {
         status = PV_SUCCESS; // kernels can be used regardless of num procs
      }
   }
   if (loc->nb != params[INDEX_NB]) {status = PV_FAILURE; tmp_status = INDEX_NB;}
   if (tmp_status == INDEX_NB) {
      if (params[INDEX_FILE_TYPE] != PVP_KERNEL_FILE_TYPE){
         fprintf(stderr, "nPad = %d != params[%d]==%d ", loc->nb, INDEX_NB, params[INDEX_NB]);
         fprintf(stderr, "\n");
      }
      else {
         status = PV_SUCCESS; // kernels can be used regardless of margin size
      }
   }
   if (loc->nf != params[INDEX_NF]) {status = PV_FAILURE; tmp_status = INDEX_NF;}
   if (tmp_status == INDEX_NF) {
         fprintf(stderr, "nBands = %d != params[%d]==%d ", loc->nf, INDEX_NF, params[INDEX_NF]);
      fprintf(stderr, "\n");
   }

   // (kx0,ky0) is for node 0 only (can be calculated otherwise)
   //
   //   if (loc->kx0      != params[INDEX_KX0])       status = -1;
   //   if (loc->ky0      != params[INDEX_KY0])       status = -1;

   if (status != 0) {
      for (int i = 0; i < numParams; i++) {
         fprintf(stderr, "params[%d]==%d ", i, params[i]);
      }
      fprintf(stderr, "\n");
   }

   return status;
} // pvp_check_file_header

int pvp_read_header(PV_Stream * pvstream, Communicator * comm, int * params, int * numParams) {
   // Under MPI, called by all processes; nonroot processes should have pvstream==NULL
   // On entry, numParams is the size of the params buffer.
   // All process should have the same numParams on entry.
   // On exit, numParams is the number of params actually read, they're read into params[0] through params[(*numParams)-1]
   // All processes receive the same params, the same numParams, and the same return value (PV_SUCCESS or PV_FAILURE).
   // If the return value is PV_FAILURE, *numParams has information on the type of failure.
   int status = PV_SUCCESS;
   int numParamsRead = 0;
   int * mpi_buffer = (int *) calloc((size_t)(*numParams+2), sizeof(int));
   // int mpi_buffer[*numParams+2]; // space for params to be MPI_Bcast, along with space for status and number of params read
   if (comm->commRank()==0) {
      if (pvstream==NULL) {
         fprintf(stderr, "pvp_read_header error: pvstream==NULL for rank zero");
         status = PV_FAILURE;
      }
      if (*numParams < 2) {
         numParamsRead = 0;
         status = PV_FAILURE;
      }

      // find out how many parameters there are
      //
      if (status == PV_SUCCESS) {
         int numread = fread(params, sizeof(int), 2, pvstream->fp);
         if (numread != 2) {
            numParamsRead = -1;
            status = PV_FAILURE;
         }
      }
      int nParams = 0;
      if (status == PV_SUCCESS) {
         nParams = params[INDEX_NUM_PARAMS];
         if (params[INDEX_HEADER_SIZE] != nParams * (int)sizeof(int)) {
            numParamsRead = -2;
            status = PV_FAILURE;
         }
      }
      if (status == PV_SUCCESS) {
         if (nParams > *numParams) {
            numParamsRead = nParams;
            status = PV_FAILURE;
         }
      }

      // read the rest
      //
      if (status == PV_SUCCESS && *numParams > 2) {
         size_t numRead = fread(&params[2], sizeof(int), nParams - 2, pvstream->fp);
         if (numRead != (size_t) nParams - 2) {
            status = PV_FAILURE;
            *numParams = numRead;
         }
      }
      if (status == PV_SUCCESS) {
         numParamsRead  = params[INDEX_NUM_PARAMS];
      }
      mpi_buffer[0] = status;
      mpi_buffer[1] = numParamsRead;
      memcpy(&mpi_buffer[2], params, sizeof(int)*(*numParams));
      MPI_Bcast(mpi_buffer, 22, MPI_INT, 0/*root*/, comm->communicator());
   } // comm->communicator()==0
   else {
      MPI_Bcast(mpi_buffer, 22, MPI_INT, 0/*root*/, comm->communicator());
      status = mpi_buffer[0];
      memcpy(params, &mpi_buffer[2], sizeof(int)*(*numParams));
   }
   *numParams = mpi_buffer[1];
   free(mpi_buffer);
   return status;
}

void read_header_err(const char * filename, Communicator * comm, int returned_num_params, int * params) {
   if (comm->commRank() != 0) {
      fprintf(stderr, "readBufferFile error while reading \"%s\"\n", filename);
      switch(returned_num_params) {
      case 0:
         fprintf(stderr, "   Called with fewer than 2 params (%d); at least two are required.\n", returned_num_params);
         break;
      case -1:
         fprintf(stderr, "   Error reading first two params from file");
         break;
      case -2:
         fprintf(stderr, "   Header size %d and number of params %d in file are not compatible.\n", params[INDEX_HEADER_SIZE], params[INDEX_NUM_PARAMS]);
         break;
      default:
         if (returned_num_params < (int) NUM_BIN_PARAMS) {
            fprintf(stderr, "   Called with %d params but only %d params could be read from file.\n", (int) NUM_BIN_PARAMS, returned_num_params);
         }
         else {
            fprintf(stderr, "   Called with %d params but file contains %d params.\n", (int) NUM_BIN_PARAMS, returned_num_params);
         }
         break;
      }
   }
   abort();
}

static
int pvp_read_header(PV_Stream * pvstream, double * time, int * filetype,
                    int * datatype, int params[], int * numParams)
{
   int status = PV_SUCCESS;

   if (*numParams < 2) {
      *numParams = 0;
      return -1;
   }

   // find out how many parameters there are
   //
   if ( fread(params, sizeof(int), 2, pvstream->fp) != 2 ) return -1;

   int nParams = params[INDEX_NUM_PARAMS];
   assert(params[INDEX_HEADER_SIZE] == (int) (nParams * sizeof(int)));
   if (nParams > *numParams) {
      *numParams = 2;
      return -1;
   }

   // read the rest
   //
   if (fread(&params[2], sizeof(int), nParams - 2, pvstream->fp) != (unsigned int) nParams - 2) return -1;

   *numParams  = params[INDEX_NUM_PARAMS];
   *filetype   = params[INDEX_FILE_TYPE];
   *datatype   = params[INDEX_DATA_TYPE];

   // make sure the parameters are what we are expecting
   //

   assert(params[INDEX_DATA_SIZE] == (int) pv_sizeof(*datatype));

   *time = timeFromParams(&params[INDEX_TIME]);

   return status;
}

int pvp_read_header(const char * filename, Communicator * comm, double * time,
                    int * filetype, int * datatype, int params[], int * numParams)
{
   int status = PV_SUCCESS;
   const int icRank = comm->commRank();

   if (icRank == 0) {
       PV_Stream * pvstream = pvp_open_read_file(filename, comm);
       if (pvstream == NULL) {
          fprintf(stderr, "[%2d]: pvp_read_header: pvp_open_read_file failed to open file \"%s\"\n",
                  comm->commRank(), filename);
          return -1;
       }

       status = pvp_read_header(pvstream, time, filetype, datatype, params, numParams);
       pvp_close_file(pvstream, comm);
       if (status != 0) return status;
   }

#ifdef PV_USE_MPI
   const int icRoot = 0;
#ifdef DEBUG_OUTPUT
   fprintf(stderr, "[%2d]: pvp_read_header: will broadcast, numParams==%d\n",
           comm->commRank(), *numParams);
#endif // DEBUG_OUTPUT

   status = MPI_Bcast(params, *numParams, MPI_INT, icRoot, comm->communicator());

#ifdef DEBUG_OUTPUT
   fprintf(stderr, "[%2d]: pvp_read_header: broadcast completed, numParams==%d\n",
           comm->commRank(), *numParams);
#endif // DEBUG_OUTPUT
#endif // PV_USE_MPI

   *filetype = params[INDEX_FILE_TYPE];
   *datatype = params[INDEX_DATA_TYPE];
   *time = timeFromParams(&params[INDEX_TIME]);

   return status;
}

int pvp_write_header(PV_Stream * pvstream, Communicator * comm, int * params, int numParams) {
   int status = PV_SUCCESS;
   int rootproc = 0;
   int rank = comm->commRank();
   if (rank == rootproc) {
      if ( (int) PV_fwrite(params, sizeof(int), numParams, pvstream) != numParams ) {
         status = -1;
      }
   }

   return status;
}



int pvp_write_header(PV_Stream * pvstream, Communicator * comm, double time, const PVLayerLoc * loc, int filetype,
                     int datatype, int numbands, bool extended, bool contiguous, unsigned int numParams, size_t localSize)
{
   int status = PV_SUCCESS;
   int nxBlocks, nyBlocks;
//   int numItems;
   int params[NUM_BIN_PARAMS];

   if (comm->commRank() != 0) return status;

   const int headerSize = numParams * sizeof(int);

   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();

//   const int nx = loc->nx;
//   const int ny = loc->ny;
//   const int nf = loc->nf;
//   const int nb = loc->nb;

   if (contiguous) {
      nxBlocks = 1;
      nyBlocks = 1;
   }
   else {
      nxBlocks = nxProcs;
      nyBlocks = nyProcs;
   }

   // const size_t globalSize = (size_t) localSize * nxBlocks * nyBlocks;

   // make sure we don't blow out size of int for record size
   // assert(globalSize < 0xffffffff); // should have checked this before calling pvp_write_header().

   int numRecords;
   switch(filetype) {
   case PVP_WGT_FILE_TYPE:
      numRecords = numbands * nxBlocks * nyBlocks; // Each process writes a record for each arbor
      break;
   case PVP_KERNEL_FILE_TYPE:
      numRecords = numbands; // Each arbor writes its own record; all processes have the same weights
      break;
   default:
      numRecords = nxBlocks * nyBlocks; // For activity files, each process writes its own record
      break;
   }

   params[INDEX_HEADER_SIZE] = headerSize;
   params[INDEX_NUM_PARAMS]  = numParams;
   params[INDEX_FILE_TYPE]   = filetype;
   params[INDEX_NX]          = contiguous ? loc->nxGlobal : loc->nx;
   params[INDEX_NY]          = contiguous ? loc->nyGlobal : loc->ny;
   params[INDEX_NF]          = loc->nf;
   params[INDEX_NUM_RECORDS] = numRecords;
   params[INDEX_RECORD_SIZE] = localSize;
   params[INDEX_DATA_SIZE]   = pv_sizeof(datatype);
   params[INDEX_DATA_TYPE]   = datatype;
   params[INDEX_NX_PROCS]    = nxBlocks;
   params[INDEX_NY_PROCS]    = nyBlocks;
   params[INDEX_NX_GLOBAL]   = loc->nxGlobal;
   params[INDEX_NY_GLOBAL]   = loc->nyGlobal;
   params[INDEX_KX0]         = loc->kx0;
   params[INDEX_KY0]         = loc->ky0;
   params[INDEX_NB]          = loc->nb;
   params[INDEX_NBANDS]      = numbands;

   timeToParams(time, &params[INDEX_TIME]);

   numParams = NUM_BIN_PARAMS;  // there may be more to come
   if ( PV_fwrite(params, sizeof(int), numParams, pvstream) != numParams ) {
      status = -1;
   }

   return status;
}

int * pvp_set_file_params(Communicator * comm, double timed, const PVLayerLoc * loc, int datatype, int numbands) {
   int numParams = NUM_BIN_PARAMS;
   int * params = alloc_params(numParams);
   assert(params!=NULL);
   params[INDEX_FILE_TYPE]   = PVP_FILE_TYPE;
   params[INDEX_NX]          = loc->nxGlobal;
   params[INDEX_NY]          = loc->nyGlobal;
   params[INDEX_NF]          = loc->nf;
   params[INDEX_NUM_RECORDS] = 1;
   int datasize = pv_sizeof(datatype);
   params[INDEX_RECORD_SIZE] = loc->nxGlobal * loc->nyGlobal * loc->nf * datasize;
   params[INDEX_DATA_SIZE]   = datasize;
   params[INDEX_DATA_TYPE]   = datatype;
   params[INDEX_NX_PROCS]    = 1;
   params[INDEX_NY_PROCS]    = 1;
   params[INDEX_NX_GLOBAL]   = loc->nxGlobal;
   params[INDEX_NY_GLOBAL]   = loc->nyGlobal;
   params[INDEX_KX0]         = 0;
   params[INDEX_KY0]         = 0;
   params[INDEX_NB]          = loc->nb;
   params[INDEX_NBANDS]      = numbands;
   timeToParams(timed, &params[INDEX_TIME]);
   return params;
}

int * pvp_set_activity_params(Communicator * comm, double timed, const PVLayerLoc * loc, int datatype, int numbands) {
   int numParams = NUM_BIN_PARAMS;
   int * params              = alloc_params(numParams);
   assert(params!=NULL);
   params[INDEX_FILE_TYPE]   = PVP_ACT_FILE_TYPE;
   params[INDEX_NX]          = loc->nxGlobal;
   params[INDEX_NY]          = loc->nyGlobal;
   params[INDEX_NF]          = loc->nf;
   params[INDEX_NUM_RECORDS] = 1;
   int datasize = pv_sizeof(datatype);
   params[INDEX_RECORD_SIZE] = loc->nxGlobal * loc->nyGlobal * loc->nf * datasize; // does not represent the size of the record in the file, but the size of the buffer
   params[INDEX_DATA_SIZE]   = datasize;
   params[INDEX_DATA_TYPE]   = datatype;
   params[INDEX_NX_PROCS]    = 1;
   params[INDEX_NY_PROCS]    = 1;
   params[INDEX_NX_GLOBAL]   = loc->nxGlobal;
   params[INDEX_NY_GLOBAL]   = loc->nyGlobal;
   params[INDEX_KX0]         = 0;
   params[INDEX_KY0]         = 0;
   params[INDEX_NBANDS]      = numbands;
   params[INDEX_NB]          = loc->nb;
   timeToParams(timed, &params[INDEX_TIME]);
   return params;
}

int * pvp_set_weight_params(Communicator * comm, double timed, const PVLayerLoc * loc, int datatype, int numbands, int nxp, int nyp, int nfp, float min, float max, int numPatches) {
   // numPatches in argument list is the number of patches per process, but what's saved in numPatches to the file is number of patches across all processes.
   int numParams = NUM_BIN_PARAMS + NUM_WGT_EXTRA_PARAMS;
   int * params = alloc_params(numParams);
   assert(params!=NULL);
   params[INDEX_FILE_TYPE]   = PVP_WGT_FILE_TYPE;
   params[INDEX_NX]          = loc->nx; // not yet contiguous
   params[INDEX_NY]          = loc->ny;
   params[INDEX_NF]          = loc->nf;
   int nxProcs               = comm->numCommColumns();
   int nyProcs               = comm->numCommRows();
   int datasize              = pv_sizeof(datatype);
   params[INDEX_NUM_RECORDS] = numbands * nxProcs * nyProcs;
   params[INDEX_RECORD_SIZE] = numPatches * (8 + datasize*nxp*nyp*nfp);
   params[INDEX_DATA_SIZE]   = datasize;
   params[INDEX_DATA_TYPE]   = datatype;
   params[INDEX_NX_PROCS]    = nxProcs;
   params[INDEX_NY_PROCS]    = nyProcs;
   params[INDEX_NX_GLOBAL]   = loc->nxGlobal;
   params[INDEX_NY_GLOBAL]   = loc->nyGlobal;
   params[INDEX_KX0]         = 0;
   params[INDEX_KY0]         = 0;
   params[INDEX_NB]          = loc->nb;
   params[INDEX_NBANDS]      = numbands;
   timeToParams(timed, &params[INDEX_TIME]);
   set_weight_params(params, nxp, nyp, nfp, min, max, numPatches);
   return params;
}

int * pvp_set_nonspiking_act_params(Communicator * comm, double timed, const PVLayerLoc * loc, int datatype, int numbands) {
   int numParams = NUM_BIN_PARAMS;
   int * params = alloc_params(numParams);
   assert(params!=NULL);
   params[INDEX_FILE_TYPE]   = PVP_NONSPIKING_ACT_FILE_TYPE;
   params[INDEX_NX]          = loc->nxGlobal;
   params[INDEX_NY]          = loc->nyGlobal;
   params[INDEX_NF]          = loc->nf;
   params[INDEX_NUM_RECORDS] = 1;
   int datasize = pv_sizeof(datatype);
   params[INDEX_RECORD_SIZE] = loc->nxGlobal * loc->nyGlobal * loc->nf;
   params[INDEX_DATA_SIZE]   = datasize;
   params[INDEX_DATA_TYPE]   = datatype;
   params[INDEX_NX_PROCS]    = 1;
   params[INDEX_NY_PROCS]    = 1;
   params[INDEX_NX_GLOBAL]   = loc->nxGlobal;
   params[INDEX_NY_GLOBAL]   = loc->nyGlobal;
   params[INDEX_KX0]         = 0;
   params[INDEX_KY0]         = 0;
   params[INDEX_NB]          = loc->nb;
   params[INDEX_NBANDS]      = numbands;
   timeToParams(timed, &params[INDEX_TIME]);
   return params;
}

int * pvp_set_kernel_params(Communicator * comm, double timed, const PVLayerLoc * loc, int datatype, int numbands, int nxp, int nyp, int nfp, float min, float max, int numPatches) {
   int numParams = NUM_BIN_PARAMS;
   int * params = alloc_params(numParams);
   assert(params!=NULL);
   params[INDEX_FILE_TYPE]   = PVP_KERNEL_FILE_TYPE;
   params[INDEX_NX]          = loc->nxGlobal;
   params[INDEX_NY]          = loc->nyGlobal;
   params[INDEX_NF]          = loc->nf;
   // int nxProcs               = 1;
   // int nyProcs               = 1;
   int datasize              = pv_sizeof(datatype);
   params[INDEX_NUM_RECORDS] = numbands;
   params[INDEX_RECORD_SIZE] = numPatches * (8 + datasize*nxp*nyp*nfp);
   params[INDEX_DATA_SIZE]   = datasize;
   params[INDEX_DATA_TYPE]   = datatype;
   params[INDEX_NX_PROCS]    = 1;
   params[INDEX_NY_PROCS]    = 1;
   params[INDEX_NX_GLOBAL]   = loc->nxGlobal;
   params[INDEX_NY_GLOBAL]   = loc->nyGlobal;
   params[INDEX_KX0]         = 0;
   params[INDEX_KY0]         = 0;
   params[INDEX_NB]          = loc->nb;
   timeToParams(timed, &params[INDEX_TIME]);
   set_weight_params(params, nxp, nyp, nfp, min, max, numPatches);
   return params;
}

int * alloc_params(int numParams) {
   int * params = NULL;
   if (numParams<2) {
      fprintf(stderr, "alloc_params must be called with at least two params (called with %d).\n", numParams);
      abort();
   }
   params = (int *) calloc((size_t) numParams, sizeof(int));
   if (params == NULL) {
      fprintf(stderr, "alloc_params unable to allocate %d params: %s\n", numParams, strerror(errno));
      abort();
   }
   params[INDEX_HEADER_SIZE] = sizeof(int)*numParams;
   params[INDEX_NUM_PARAMS] = numParams;
   return params;
}

int set_weight_params(int * params, int nxp, int nyp, int nfp, float min, float max, int numPatches) {
   int * wgtParams = &params[NUM_BIN_PARAMS];
   wgtParams[INDEX_WGT_NXP]  = nxp;
   wgtParams[INDEX_WGT_NYP]  = nyp;
   wgtParams[INDEX_WGT_NFP]  = nfp;
   assert(sizeof(int)==sizeof(float));
   union float_as_int {
      float f;
      int   i;
   };
   union float_as_int p;
   p.f = min;
   wgtParams[INDEX_WGT_MIN] = p.i;
   p.f = max;
   wgtParams[INDEX_WGT_MAX] = p.i;
   wgtParams[INDEX_WGT_NUMPATCHES]  = numPatches;
   return PV_SUCCESS;
}

int pvp_read_time(PV_Stream * pvstream, Communicator * comm, int root_process, double * timed) {
   // All processes call this routine simultaneously.
   // from the file at the current location, loaded into the variable timed, and
   // broadcast to all processes.  All processes have the same return value:
   // PV_SUCCESS if the read was successful, PV_FAILURE if not.
   int status = PV_SUCCESS;
   struct timeandstatus {
      int status;
      double time;
   };
   struct timeandstatus mpi_data;
   if (comm->commRank()==root_process) {
      if (pvstream==NULL) {
         fprintf(stderr, "pvp_read_time error: root process called with null stream argument.\n");
         abort();
      }
      int numread = fread(timed, sizeof(*timed), 1, pvstream->fp);
      mpi_data.status = (numread == 1) ? PV_SUCCESS : PV_FAILURE;
      mpi_data.time = *timed;
   }
   MPI_Bcast(&mpi_data, (int) sizeof(timeandstatus), MPI_CHAR, root_process, comm->communicator());
   status = mpi_data.status;
   *timed = mpi_data.time;
   return status;
}

int writeActivity(PV_Stream * pvstream, Communicator * comm, double timed, PVLayer * l)
{
   int status = PV_SUCCESS;
   // write header, but only at the beginning
#ifdef PV_USE_MPI
   int rank = comm->commRank();
#else // PV_USE_MPI
   int rank = 0;
#endif // PV_USE_MPI
   if( rank == 0 ) {
      long fpos = PV_ftell(pvstream);
      if (fpos == 0L) {
         int * params = pvp_set_nonspiking_act_params(comm, timed, &l->loc, PV_FLOAT_TYPE, 1/*numbands*/);
         assert(params && params[1]==NUM_BIN_PARAMS);
         int numParams = params[1];
         status = pvp_write_header(pvstream, comm, params, numParams);
      }
      // HyPerLayer::writeActivity calls HyPerLayer::incrementNBands, which maintains the value of numbands in the header.

      // write time
      //
      if ( PV_fwrite(&timed, sizeof(double), 1, pvstream) != 1 ) {
         fprintf(stderr,"fwrite of timestamp in PV::writeActivity failed for layer %d at time %f\n", l->layerId, timed);
         abort();
         return -1;
      }
   }

   if (gatherActivity(pvstream, comm, 0/*root process*/, l->activity->data, &l->loc, true/*extended*/)!=PV_SUCCESS) {
      status = PV_FAILURE;
   }
   return status;
}

int writeActivitySparse(PV_Stream * pvstream, Communicator * comm, double time, PVLayer * l)
{
   int status = PV_SUCCESS;

   const int icRoot = 0;
   const int icRank = comm->commRank();
   int localActive  = l->numActive;
   unsigned int * indices = l->activeIndices;

#ifdef PV_USE_MPI
   const int tag = PVP_ACT_FILE_TYPE;
   const MPI_Comm mpi_comm = comm->communicator();
#endif // PV_USE_MPI

   if (icRank != icRoot) {

#ifdef PV_USE_MPI
      const int dest = icRoot;
      MPI_Send(&localActive, 1, MPI_INT, dest, tag, mpi_comm);
      MPI_Send(indices, localActive, MPI_INT, dest, tag, mpi_comm);
#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: writeActivitySparse: sent to %d, localActive==%d\n",
              comm->commRank(), dest, localActive);
      fflush(stderr);
#endif // DEBUG_OUTPUT
#endif // PV_USE_MPI

      // leaving not root-process section
      //
   }
   else {
      // we are io root process
      //
      unsigned int totalActive = localActive;

#ifdef PV_USE_MPI
      // get the number active from each process
      // TODO - use collective?
      //
      unsigned int * numActive = NULL;
      const int icSize = comm->commSize();

      if (icSize > 1) {
         // otherwise numActive is not used
         numActive = (unsigned int *) malloc(icSize*sizeof(int));
         assert(numActive != NULL);
      }

      for (int p = 1; p < icSize; p++) {
         MPI_Recv(&numActive[p], 1, MPI_INT, p, tag, mpi_comm, MPI_STATUS_IGNORE);
         totalActive += numActive[p];
      }
#endif // PV_USE_MPI

      bool extended   = false;
      bool contiguous = true;

      const int datatype = PV_INT_TYPE;

      // write activity header
      //
      long fpos = PV_ftell(pvstream);
      if (fpos == 0L) {
         int numParams = NUM_BIN_PARAMS;
         status = pvp_write_header(pvstream, comm, time, &l->loc, PVP_ACT_FILE_TYPE,
                                   datatype, 1, extended, contiguous, numParams, (size_t) localActive);
         if (status != 0) {
            fprintf(stderr, "[%2d]: writeActivitySparse: failed in pvp_write_header, numParams==%d, localActive==%d\n",
                    comm->commRank(), numParams, localActive);
            return status;
         }
      }

      // write time, total active count, and local activity
      //
      status = (PV_fwrite(&time, sizeof(double), 1, pvstream) != 1 );
      if (status != 0) {
         fprintf(stderr, "[%2d]: writeActivitySparse: failed in fwrite(&time), time==%f\n",
                 comm->commRank(), time);
         return status;
      }
      status = ( PV_fwrite(&totalActive, sizeof(unsigned int), 1, pvstream) != 1 );
      if (status != 0) {
         fprintf(stderr, "[%2d]: writeActivitySparse: failed in fwrite(&totalActive), totalActive==%d\n",
                 comm->commRank(), totalActive);
         return status;
      }
     if (localActive > 0) {
         status = (PV_fwrite(indices, sizeof(unsigned int), localActive, pvstream) != (size_t) localActive );
         if (status != 0) {
            fprintf(stderr, "[%2d]: writeActivitySparse: failed in fwrite(indices), localActive==%d\n",
                    comm->commRank(), localActive);
            return status;
         }
      }

      // recv and write non-local activity
      //
#ifdef PV_USE_MPI
      for (int p = 1; p < icSize; p++) {
#ifdef DEBUG_OUTPUT
            fprintf(stderr, "[%2d]: writeActivitySparse: receiving from %d numActive==%d\n",
                    comm->commRank(), p, numActive[p]);
            fflush(stderr);
#endif // DEBUG_OUTPUT
            MPI_Recv(indices, numActive[p], MPI_INT, p, tag, mpi_comm, MPI_STATUS_IGNORE);
            status = (PV_fwrite(indices, sizeof(unsigned int), numActive[p], pvstream) != numActive[p] );
            if (status != 0) {
               fprintf(stderr, "[%2d]: writeActivitySparse: failed in fwrite(indices), numActive[p]==%d, p=%d\n",
                       comm->commRank(), numActive[p], p);
               return status;
            }
      }
      if (numActive != NULL) free(numActive);
#endif // PV_USE_MPI

      // leaving root-process section
      //
   }

   return status;
}

int readWeights(PVPatch *** patches, pvdata_t ** dataStart, int numArbors, int numPatches,
      const char * filename, Communicator * comm, double * timed, const PVLayerLoc * loc,
      bool * shmget_owner, bool shmget_flag)
{
   int status = PV_SUCCESS;
   int header_data_type;
   int header_file_type;

   int numParams = NUM_WGT_PARAMS;
   int params[NUM_WGT_PARAMS];

   int nxBlocks, nyBlocks;

   bool contiguous = false;   // for now

   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();

   const int icRank = comm->commRank();

   status = pvp_read_header(filename, comm, timed, &header_file_type, &header_data_type, params, &numParams);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "[%2d]: readWeights: failed in pvp_read_header, numParams==%d\n",
              comm->commRank(), numParams);
      return status;
   }

   status = pvp_check_file_header(comm, loc, params, numParams);
   if (status != 0) {
      fprintf(stderr, "[%2d]: readWeights: failed in pvp_check_file_header, numParams==%d\n",
              comm->commRank(), numParams);
      return status;
   }

   const int nxFileBlocks = params[INDEX_NX_PROCS];
   const int nyFileBlocks = params[INDEX_NY_PROCS];

   int * wgtParams = &params[NUM_BIN_PARAMS];

   const int nxp = wgtParams[INDEX_WGT_NXP];
   const int nyp = wgtParams[INDEX_WGT_NYP];
   const int nfp = wgtParams[INDEX_WGT_NFP];

   // Have to use memcpy instead of casting floats because of strict aliasing rules, since some are int and some are float
   float minVal = 0.0f;
   memcpy(&minVal, &wgtParams[INDEX_WGT_MIN], sizeof(float));
   float maxVal = 0.0f;
   memcpy(&maxVal, &wgtParams[INDEX_WGT_MAX], sizeof(float));
   // const float minVal = * ((float*) &wgtFloatParams[INDEX_WGT_MIN]);
   // const float maxVal = * ((float*) &wgtFloatParams[INDEX_WGT_MAX]);

   if (contiguous) {
      nxBlocks = 1;
      nyBlocks = 1;
   }
   else {
      nxBlocks = nxProcs;
      nyBlocks = nyProcs;
   }

   // make sure file is consistent with expectations
   //
   status = ( header_data_type != PV_BYTE_TYPE && header_data_type != PV_FLOAT_TYPE );
   if (status != 0) {
      fprintf(stderr, "[%2d]: readWeights: failed in pvp_check_file_header, datatype==%d\n",
              comm->commRank(), header_data_type);
      return status;
   }
   if (header_file_type != PVP_KERNEL_FILE_TYPE){
      status = (nxBlocks != nxFileBlocks || nyBlocks != nyFileBlocks);
      if (status != 0) {
         fprintf(stderr, "[%2d]: readWeights: failed in pvp_check_file_header, "
               "nxFileBlocks==%d, nyFileBlocks==%d\n, nxBlocks==%d, nyBlocks==%d\n",
               comm->commRank(), nxFileBlocks, nyFileBlocks, nxBlocks, nyBlocks);
         return status;
      }
   }
   if (header_file_type != PVP_KERNEL_FILE_TYPE){
      status = (numPatches*nxProcs*nyProcs != wgtParams[INDEX_WGT_NUMPATCHES]);
   }
   else{
      status = ((numPatches != wgtParams[INDEX_WGT_NUMPATCHES]));
   }
   if (status != 0) {
      fprintf(stderr, "[%2d]: readWeights: failed in pvp_check_file_header, "
            "numPatches==%d, nxProcs==%d\n, nyProcs==%d, wgtParams[INDEX_WGT_NUMPATCHES]==%d\n",
            comm->commRank(), numPatches, nxProcs, nyProcs, wgtParams[INDEX_WGT_NUMPATCHES]);
      return status;
   }


   const int numPatchItems = nxp * nyp * nfp;
   const size_t patchSize = pv_sizeof_patch(numPatchItems, header_data_type);
   const size_t localSize = numPatches * patchSize;

   unsigned char * cbuf = (unsigned char *) malloc(localSize);
   assert(cbuf != NULL);

#ifdef PV_USE_MPI
   const int tag = header_file_type;
   const MPI_Comm mpi_comm = comm->communicator();
#endif // PV_USE_MPI

   // read weights and send using MPI
   if( params[INDEX_NBANDS] > numArbors ) {
      fprintf(stderr, "PV::readWeights: file \"%s\" has %d arbors, but readWeights was called with only %d arbors", filename, params[INDEX_NBANDS], numArbors);
      return -1;
   }
   PV_Stream * pvstream = pvp_open_read_file(filename, comm);
   for(int arborId=0; arborId<params[INDEX_NBANDS]; arborId++) {
      if (icRank > 0) {

#ifdef PV_USE_MPI
         const int src = 0;

#ifdef DEBUG_OUTPUT
         fprintf(stderr, "[%2d]: readWeights: recv from %d, nxBlocks==%d nyBlocks==%d numPatches==%d\n",
                 comm->commRank(), src, nxBlocks, nyBlocks, numPatches);
#endif // DEBUG_OUTPUT
         MPI_Recv(cbuf, localSize, MPI_BYTE, src, tag, mpi_comm, MPI_STATUS_IGNORE);
#ifdef DEBUG_OUTPUT
         fprintf(stderr, "[%2d]: readWeights: recv from %d completed\n",
                 comm->commRank(), src);
#endif // DEBUG_OUTPUT
#endif // PV_USE_MPI

      }
      else {
         const int headerSize = numParams * sizeof(int);

         if (pvstream == NULL) {
            fprintf(stderr, "PV::readWeights: ERROR opening file %s\n", filename);
            return -1;
         }

         long arborStart;
#ifdef PV_USE_MPI
         int dest = -1;
#endif // PV_USE_MPI
         if( header_file_type == PVP_KERNEL_FILE_TYPE ) {
            arborStart = headerSize + localSize*arborId;
            long offset = arborStart;
            PV_fseek(pvstream, offset, SEEK_SET);
            int numRead = fread(cbuf, localSize, 1, pvstream->fp);
            if( numRead != 1 ) return -1;
#ifdef PV_USE_MPI
            for( int py=0; py<nyProcs; py++ ) {
               for( int px=0; px<nxProcs; px++ ) {
                  if( ++dest == 0 ) continue;
                  MPI_Send(cbuf, localSize, MPI_BYTE, dest, tag, mpi_comm);
               }
            }
#endif // PV_USE_MPI
         }
         else {
            arborStart = headerSize + localSize*nxBlocks*nyBlocks*arborId;
#ifdef PV_USE_MPI
            for( int py=0; py<nyProcs; py++ ) {
               for( int px=0; px<nxProcs; px++ ) {
                  if( ++dest == 0 ) continue;
                  long offset = arborStart + dest*localSize;
                  PV_fseek(pvstream, offset, SEEK_SET);
                  int numRead = fread(cbuf, localSize, 1, pvstream->fp);
                  if( numRead != 1 ) return -1;
                  MPI_Send(cbuf, localSize, MPI_BYTE, dest, tag, mpi_comm);
               }
            }
#endif // PV_USE_MPI
         }

         // read local portion
         // numPatches - each neuron has a patch; pre-synaptic neurons live in extended layer
         //
#ifdef PV_USE_MPI
         bool readLocalPortion = header_file_type != PVP_KERNEL_FILE_TYPE;
#else
         bool readLocalPortion = true;
#endif // PV_USE_MPI
         if( readLocalPortion ) {
            long offset = arborStart + 0*localSize;
            PV_fseek(pvstream, offset, SEEK_SET);
            int numRead = fread(cbuf, localSize, 1, pvstream->fp);
            if  (numRead != 1) {
               fprintf(stderr, "[%2d]: readWeights: failed in fread, offset==%ld\n",
                     comm->commRank(), offset);
            }
         }
      }  // if rank == 0

      // set the contents of the weights patches from the unsigned character buffer, cbuf
      //
#ifdef USE_SHMGET
      //only owner should write if connection uses shared memeory
      if (shmget_flag){
    	  assert(shmget_owner != NULL);
    	  if(!shmget_owner[arborId]){
    		  continue;
    	  }
      }
#endif
      bool compress = header_data_type == PV_BYTE_TYPE;
      status = pvp_set_patches(cbuf, patches ? patches[arborId] : NULL, dataStart[arborId], numPatches, nxp, nyp, nfp, minVal, maxVal, compress);
      if (status != PV_SUCCESS) {
         fprintf(stderr, "[%2d]: readWeights: failed in pvp_set_patches, numPatches==%d\n",
                 comm->commRank(), numPatches);
      }
   } // loop over arborId
   free(cbuf);
   status = pvp_close_file(pvstream, comm)==PV_SUCCESS ? status : PV_FAILURE;
   return status;
}

int writeWeights(const char * filename, Communicator * comm, double timed, bool append,
                 const PVLayerLoc * loc, int nxp, int nyp, int nfp, float minVal, float maxVal,
                 PVPatch *** patches, pvdata_t ** dataStart, int numPatches, int numArbors, bool compress, int file_type)
// compress has default of true, file_type has default value of PVP_WGT_FILE_TYPE
// If file_type is PVP_WGT_FILE_TYPE (HyPerConn), the patches variable is consulted for the shrunken patch information.
// If file_type is PVP_KERNEL_FILE_TYPE, patches is ignored and all patches are written with nx=nxp and ny=nyp
{
   int status = PV_SUCCESS;
   int nxBlocks, nyBlocks;

   bool extended = true;
   bool contiguous = false;   // TODO implement contiguous = true case

   int datatype = compress ? PV_BYTE_TYPE : PV_FLOAT_TYPE;

   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();

   const int icRank = comm->commRank();

   const int numPatchItems = nxp * nyp * nfp;
   const size_t patchSize = pv_sizeof_patch(numPatchItems, datatype);
   const size_t localSize = numPatches * patchSize;

   if (contiguous) {
      nxBlocks = 1;
      nyBlocks = 1;
   }
   else {
      nxBlocks = nxProcs;
      nyBlocks = nyProcs;
   }

   unsigned char * cbuf = (unsigned char *) malloc(localSize);
   if(cbuf == NULL) {
      fprintf(stderr, "Rank %d: writeWeights unable to allocate memory\n", icRank);
      abort();
   }


#ifdef PV_USE_MPI
   const int tag = file_type; // PVP_WGT_FILE_TYPE;
   const MPI_Comm mpi_comm = comm->communicator();
#endif // PV_USE_MPI
   if( icRank > 0 ) {
#ifdef PV_USE_MPI
      if( file_type != PVP_KERNEL_FILE_TYPE ) {
         const int dest = 0;
         for( int arbor=0; arbor<numArbors; arbor++ ) {
            pvp_copy_patches(cbuf, patches[arbor], dataStart[arbor], numPatches, nxp, nyp, nfp, minVal, maxVal, compress);
            MPI_Send(cbuf, localSize, MPI_BYTE, dest, tag, mpi_comm);
#ifdef DEBUG_OUTPUT
            fprintf(stderr, "[%2d]: writeWeights: sent to 0, nxBlocks==%d nyBlocks==%d numPatches==%d\n",
                    comm->commRank(), nxBlocks, nyBlocks, numPatches);
#endif // DEBUG_OUTPUT
         }
      }
#endif // PV_USE_MPI
   } // icRank > 0
   else /* icRank==0 */ {
      void * wgtExtraParams = calloc(NUM_WGT_EXTRA_PARAMS, sizeof(int));
      if (wgtExtraParams == NULL) {
         fprintf(stderr, "calloc error in writeWeights for \"%s\": %s\n", filename, strerror(errno));
         abort();
      }
      int * wgtExtraIntParams = (int *) wgtExtraParams;
      float * wgtExtraFloatParams = (float *) wgtExtraParams;
      // int wgtExtraParams[NUM_WGT_EXTRA_PARAMS];

      int numParams = NUM_WGT_PARAMS;

      PV_Stream * pvstream = pvp_open_write_file(filename, comm, append);

      if (pvstream == NULL) {
         fprintf(stderr, "PV::writeWeights: ERROR opening file %s\n", filename);
         return -1;
      }
      if (append) PV_fseek(pvstream, 0L, SEEK_END); // If append is true we open in "r+" mode so we need to move to the end of the file.

      // use file_type passed as argument to enable different behavior
      status = pvp_write_header(pvstream, comm, timed, loc, file_type,
                                datatype, numArbors, extended, contiguous, numParams, localSize);

      // write extra weight parameters
      //
      wgtExtraIntParams[INDEX_WGT_NXP] = nxp;
      wgtExtraIntParams[INDEX_WGT_NYP] = nyp;
      wgtExtraIntParams[INDEX_WGT_NFP] = nfp;

      wgtExtraFloatParams[INDEX_WGT_MIN] = minVal;
      wgtExtraFloatParams[INDEX_WGT_MAX] = maxVal;

      if (file_type == PVP_KERNEL_FILE_TYPE){
         wgtExtraIntParams[INDEX_WGT_NUMPATCHES] = numPatches; // KernelConn has same weights in all processes
      }
      else {
         wgtExtraIntParams[INDEX_WGT_NUMPATCHES] = numPatches * nxBlocks * nyBlocks;
      }

      numParams = NUM_WGT_EXTRA_PARAMS;
      unsigned int num_written = PV_fwrite(wgtExtraParams, sizeof(int), numParams, pvstream);
      free(wgtExtraParams); wgtExtraParams=NULL; wgtExtraIntParams=NULL; wgtExtraFloatParams=NULL;
      if ( num_written != (unsigned int) numParams ) {
         fprintf(stderr, "PV::writeWeights: error writing weight header to file %s\n", filename);
         return -1;
      }

      for( int arbor=0; arbor<numArbors; arbor++ ) {
         // write local portion
         // numPatches - each neuron has a patch; pre-synaptic neurons live in extended layer
         PVPatch ** arborPatches = file_type == PVP_KERNEL_FILE_TYPE ? NULL : patches[arbor];
         pvp_copy_patches(cbuf, arborPatches, dataStart[arbor], numPatches, nxp, nyp, nfp, minVal, maxVal, compress);
         size_t numfwritten = PV_fwrite(cbuf, localSize, 1, pvstream);
         if ( numfwritten != 1 ) {
            fprintf(stderr, "PV::writeWeights: error writing weight data to file %s\n", filename);
            return -1;
         }

         if (file_type == PVP_KERNEL_FILE_TYPE) continue;

         // gather portions from other processes
#ifdef PV_USE_MPI
         int src = -1;
         for (int py = 0; py < nyProcs; py++) {
            for (int px = 0; px < nxProcs; px++) {
               if (++src == 0) continue;
#ifdef DEBUG_OUTPUT
               fprintf(stderr, "[%2d]: writeWeights: receiving from %d nxProcs==%d nyProcs==%d localSize==%ld\n",
                     comm->commRank(), src, nxProcs, nyProcs, localSize);
#endif // DEBUG_OUTPUT
               MPI_Recv(cbuf, localSize, MPI_BYTE, src, tag, mpi_comm, MPI_STATUS_IGNORE);

               // const int headerSize = numParams * sizeof(int);
               // long offset = headerSize + src * localSize;
               // fseek(fp, offset, SEEK_SET);
               if ( PV_fwrite(cbuf, localSize, 1, pvstream) != 1 ) return -1;
            }
         }
#endif // PV_USE_MPI
      } // end loop over arbors
      pvp_close_file(pvstream, comm);
   } // icRank == 0
   free(cbuf); cbuf = NULL;

   return PV_SUCCESS; // TODO error handling
}  // end writeWeights (all arbors)

int writeRandState(const char * filename, Communicator * comm, uint4 * randState, const PVLayerLoc * loc) {
   int status = PV_SUCCESS;
   int rootproc = 0;
   int rank = comm->commRank();

   PV_Stream * pvstream = NULL;
   if (rank == rootproc) {
      pvstream = PV_fopen(filename, "w");
      if (pvstream==NULL) {
         fprintf(stderr, "writeRandState error: unable to open path %s for writing.\n", filename);
         abort();
      }
   }
   status = gatherActivity(pvstream, comm, rootproc, randState, loc, false/*extended*/);
   if (rank==rootproc) {
      PV_fclose(pvstream); pvstream = NULL;
   }
   return status;
}

int readRandState(const char * filename, Communicator * comm, uint4 * randState, const PVLayerLoc * loc) {
   int status = PV_SUCCESS;
   int rootproc = 0;
   int rank = comm->commRank();

   PV_Stream * pvstream = NULL;
   if (rank == rootproc) {
      pvstream = PV_fopen(filename, "r");
      if (pvstream==NULL) {
         fprintf(stderr, "readRandState error: unable to open path %s for reading.\n", filename);
         abort();
      }
   }
   status = scatterActivity(pvstream, comm, rootproc, randState, loc, false/*extended*/);
   if (rank==rootproc) {
      PV_fclose(pvstream); pvstream = NULL;
   }
   return status;
}

} // namespace PV
