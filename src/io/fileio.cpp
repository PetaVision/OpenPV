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
static void timeToParams(double time, void * params)
{
   memcpy(params, &time, sizeof(double));
}

static double timeFromParams(void * params)
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

FILE * pvp_open_read_file(const char * filename, Communicator * comm)
{
   FILE * fp = NULL;
   if (comm->commRank() == 0) {
      fp = fopen(filename, "rb");
   }
   return fp;
}

FILE * pvp_open_write_file(const char * filename, Communicator * comm, bool append)
{
   FILE * fp = NULL;
   if (comm->commRank() == 0) {
      if (append) fp = fopen(filename, "ab");
      else        fp = fopen(filename, "wb");
      if( !fp ) {
         fprintf(stderr, "pvp_open_write_file error opening \"%s\" for writing: %s\n", filename, strerror(errno));
         exit(EXIT_FAILURE);
      }
   }
   return fp;
}

int pvp_close_file(FILE * fp, Communicator * comm)
{
   int status = PV_SUCCESS;
   if (fp != NULL) {
      status = fclose(fp);
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

static
int pvp_read_header(FILE * fp, double * time, int * filetype,
                    int * datatype, int params[], int * numParams)
{
   int status = PV_SUCCESS;

   if (*numParams < 2) {
      *numParams = 0;
      return -1;
   }

   // find out how many parameters there are
   //
   if ( fread(params, sizeof(int), 2, fp) != 2 ) return -1;

   int nParams = params[INDEX_NUM_PARAMS];
   assert(params[INDEX_HEADER_SIZE] == (int) (nParams * sizeof(int)));
   if (nParams > *numParams) {
      *numParams = 2;
      return -1;
   }

   // read the rest
   //
   if (fread(&params[2], sizeof(int), nParams - 2, fp) != (unsigned int) nParams - 2) return -1;

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
       FILE * fp = pvp_open_read_file(filename, comm);
       if (fp == NULL) {
          fprintf(stderr, "[%2d]: pvp_read_header: pvp_open_read_file failed to open file \"%s\"\n",
                  comm->commRank(), filename);
          return -1;
       }

       status = pvp_read_header(fp, time, filetype, datatype, params, numParams);
       pvp_close_file(fp, comm);
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

int pvp_write_header(FILE * fp, Communicator * comm, double time, const PVLayerLoc * loc, int filetype,
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
   if ( fwrite(params, sizeof(int), numParams, fp) != numParams ) {
      status = -1;
   }

   return status;
}

int read_pvdata(const char * filename, Communicator * comm, double * timed, void * data,
         const PVLayerLoc * loc, int datatype, bool extended, bool contiguous)
{
   int status = PV_SUCCESS;

   // TODO - everything isn't implemented yet so make sure we are using it correctly
   assert(datatype == PV_FLOAT_TYPE || datatype == PV_INT_TYPE);
   assert(sizeof(float) == 4 && sizeof(int) == 4);

   // scale factor for floating point conversion
   float scale = 1.0f;

   const int icRank = comm->commRank();

// Only the interior, non-restricted part of the buffer gets written, even if the buffer is extended.
   int numItemsInMem = loc->nx * loc->ny * loc->nf;

   const size_t sizeBufInMem = numItemsInMem * pv_sizeof(datatype);

   unsigned char * cbuf = (unsigned char *) malloc(sizeBufInMem);
   if(cbuf == NULL) {
      fprintf(stderr, "read_pvdata: unable to allocate memory for file %s\n", filename);
      abort();
   }

#ifdef PV_USE_MPI
   const int tag = PVP_FILE_TYPE;
   const MPI_Comm mpi_comm = comm->communicator();
#endif // PV_USE_MPI
   int fileexists;

   if (icRank > 0) {

#ifdef PV_USE_MPI
      const int src = 0;
      MPI_Bcast(&fileexists, 1, MPI_INT, src, mpi_comm);
#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: read: received from 0, filename \"%s\"",filename);
      if( fileexists ) fprintf(stderr, "exists\n");
      else fprintf(stderr, "does not exist\n");
#endif // DEBUG_OUTPUT
      if( !fileexists ) return PV_ERR_FILE_NOT_FOUND;

      MPI_Recv(cbuf, sizeBufInMem, MPI_BYTE, src, tag, mpi_comm, MPI_STATUS_IGNORE);

#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: read: received from 0, nx==%d ny==%d numItems==%d\n",
              comm->commRank(), loc->nx, loc->ny, numItems);
#endif // DEBUG_OUTPUT
#endif // PV_USE_MPI
   }
   else {
      int params[NUM_PAR_BYTE_PARAMS];
      int numParams, numRead, type, nxIn, nyIn, nfIn;

      FILE * fp = pv_open_binary(filename, &numParams, &type, &nxIn, &nyIn, &nfIn);
      fileexists = fp != NULL;
#ifdef PV_USE_MPI
      MPI_Bcast(&fileexists, 1, MPI_INT, 0, mpi_comm);
#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: read: broadcasting from 0, filename \"%s\"",filename);
      if( fileexists ) fprintf(stderr, "exists\n");
      else fprintf(stderr, "does not exist\n");
#endif // DEBUG_OUTPUT
#endif // PV_USE_MPI
      if (!fileexists) return PV_ERR_FILE_NOT_FOUND;
      
      if(numParams != NUM_PAR_BYTE_PARAMS) {
         fprintf(stderr, "read_pvdata: expecting %lu params in file %s, but NUM_PAR_BYTE_PARAMS is %d.  Exiting.\n", NUM_PAR_BYTE_PARAMS, filename, numParams);
         abort();
      }
      assert(type      == PVP_FILE_TYPE);

      status = pv_read_binary_params(fp, numParams, params);
      assert(status == numParams);

      const int dataSize = params[INDEX_DATA_SIZE];
      const int dataType = params[INDEX_DATA_TYPE];

      *timed = timeFromParams(&params[INDEX_TIME]);

      assert(dataSize == (int) pv_sizeof(datatype));
      assert(dataType == PV_FLOAT_TYPE || dataType == PV_INT_TYPE);

#ifdef PV_USE_MPI
      const int nxProcsInMem = comm->numCommColumns();
      const int nyProcsInMem = comm->numCommRows();

      int dest = -1;
      for (int py = 0; py < nyProcsInMem; py++) {
         for (int px = 0; px < nxProcsInMem; px++) {
            if (++dest == 0) continue;

#ifdef DEBUG_OUTPUT
            fprintf(stderr, "[%2d]: read: sending to %d nx==%d ny==%d numItems==%d\n",
                    comm->commRank(), dest, loc->nx, loc->ny, numItems);
#endif // DEBUG_OUTPUT
            numRead = read_pvdata_oneproc(fp, px, py, loc, cbuf, sizeBufInMem, params, numParams);
            //long offset = headerSize + dest * sizeBufInMem;
            //fseek(fp, offset, SEEK_SET);
            //numRead = fread(cbuf, sizeof(unsigned char), sizeBufInMem, fp);
            assert(numRead == (int) sizeBufInMem);
            MPI_Send(cbuf, sizeBufInMem, MPI_BYTE, dest, tag, mpi_comm);
         }
      }
#endif // PV_USE_MPI

      // get local image portion
      numRead = read_pvdata_oneproc(fp, 0, 0, loc, cbuf, sizeBufInMem, params, numParams);
      //fseek(fp, (long) headerSize, SEEK_SET);
      //numRead = fread(cbuf, sizeof(unsigned char), sizeBufInMem, fp);
      assert(numRead == (int) sizeBufInMem);

      status = pvp_close_file(fp, comm);
   }

   // copy from communication buffer to data array
   //
   if (datatype == PV_FLOAT_TYPE) {
      float * fbuf = (float *) cbuf;
      status = HyPerLayer::copyFromBuffer(fbuf, (float*) data, loc, extended, scale);
   }
   else if (datatype == PV_INT_TYPE) {
      int * fbuf = (int *) cbuf;
      status = HyPerLayer::copyFromBuffer(fbuf, (int*) data, loc, extended, 1);
   }
   free(cbuf);
   return status;
}

size_t read_pvdata_oneproc(FILE * fp, int px, int py, const PVLayerLoc * loc, unsigned char * cbuf, const size_t localSizeInMem, const int * params, int numParams) {
   assert(loc->nf == params[INDEX_NF]);
   unsigned char * ptrintocbuf = cbuf;
   bool startset = false;
   size_t numread = 0;
   long offset;
   long blockstart = -1;
   long blockstop = -1;
   for( int y=py*loc->ny; y<(py+1)*loc->ny; y++ ) {
      for( int x=px*loc->nx; x<(px+1)*loc->nx; x++ ) {
         int xProcInFile = x/params[INDEX_NX];
         int yProcInFile = y/params[INDEX_NY];
         int kProcInFile = rankFromRowAndColumn(yProcInFile, xProcInFile, params[INDEX_NY_PROCS], params[INDEX_NX_PROCS]);
         int xInProc = x % params[INDEX_NX];
         int yInProc = y % params[INDEX_NY];
         int idxInProc = kIndex(xInProc, yInProc, 0, params[INDEX_NX], params[INDEX_NY], params[INDEX_NF]);
         offset = params[INDEX_HEADER_SIZE] + kProcInFile * params[INDEX_RECORD_SIZE] + idxInProc*params[INDEX_DATA_SIZE];
         if( startset ) {
            assert(blockstart>=0 && blockstop>=0);
            if( offset == blockstop ) {
               blockstop += params[INDEX_NF]*params[INDEX_DATA_SIZE];
            }
            else {
               fseek(fp, blockstart, SEEK_SET);
               numread += fread(ptrintocbuf, 1, blockstop-blockstart, fp);
               blockstart = offset;
               blockstop = offset + params[INDEX_NF]*params[INDEX_DATA_SIZE];
               ptrintocbuf = &cbuf[numread];
            }
         }
         else {
            blockstart = offset;
            blockstop = offset + params[INDEX_NF]*params[INDEX_DATA_SIZE];
            startset = true;
         }
      }
   }
   fseek(fp, blockstart, SEEK_SET);
   numread += fread(ptrintocbuf, 1, blockstop-blockstart, fp);
   return numread;
}

int readNonspikingActFile(const char * filename, Communicator * comm, double * time, void * data,
         int level, const PVLayerLoc * loc, int datatype, bool extended, bool contiguous)
{
   int status = PV_SUCCESS;
   int nxBlocks, nyBlocks, numItems;

   // TODO - everything isn't implemented yet so make sure we are using it correctly
   assert(contiguous == false);
   assert(datatype == PV_FLOAT_TYPE || datatype == PV_INT_TYPE);
   assert(sizeof(float) == sizeof(int));

   // scale factor for floating point conversion
   float scale = 1.0f;

   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();

   const int icRank = comm->commRank();

// Only the interior, non-restricted part of the buffer gets written, even if the buffer is extended.
   numItems = loc->nx * loc->ny * loc->nf;

   const size_t localSize = numItems*pv_sizeof(datatype);
   // mult by pv_sizof is peculiar to PVP_NONSPIKING_ACT_FILE_TYPE and should probably be changed
   // addition of sizeof(double) handles the timestamp

   const size_t mpiBufferSize = sizeof(int) + sizeof(double) + localSize;
   // mpiBufferSize consists of:
   // one integer to transmit information on whether the file could be opened (PV_SUCCESS or PV_FAILURE)
   // one double-precision float for the time
   // loc->nx*loc->ny*loc->nf objects of type datatype for the layer values

   if (contiguous) {
      nxBlocks = 1;
      nyBlocks = 1;
   }
   else {
      nxBlocks = nxProcs;
      nyBlocks = nyProcs;
   }

   unsigned char * cbuf = (unsigned char *) malloc(mpiBufferSize);
   assert(cbuf != NULL);

#ifdef PV_USE_MPI
   const int tag = PVP_FILE_TYPE;
   const MPI_Comm mpi_comm = comm->communicator();
#endif // PV_USE_MPI

   if (icRank > 0) {

#ifdef PV_USE_MPI
      const int src = 0;
      MPI_Recv(cbuf, mpiBufferSize, MPI_BYTE, src, tag, mpi_comm, MPI_STATUS_IGNORE);

#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: read: received from 0, nx==%d ny==%d numItems==%d\n",
              comm->commRank(), loc->nx, loc->ny, numItems);
#endif // DEBUG_OUTPUT
#endif // PV_USE_MPI
   }
   else {
      int params[NUM_PAR_BYTE_PARAMS];
      int numParams, numRead, type, nxIn, nyIn, nfIn;

      FILE * fp = pv_open_binary(filename, &numParams, &type, &nxIn, &nyIn, &nfIn);
      status = fp != NULL ? PV_SUCCESS : PV_FAILURE;
      *((int *) cbuf) = status;
#ifdef PV_USE_MPI
#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: read: broadcasting from 0, filename \"%s\"",filename);
      if( fileexists ) fprintf(stderr, "exists\n");
      else fprintf(stderr, "does not exist\n");
#endif // DEBUG_OUTPUT
#endif // PV_USE_MPI

      assert(numParams == NUM_PAR_BYTE_PARAMS);
      assert(type      == PVP_NONSPIKING_ACT_FILE_TYPE);

      assert( pv_read_binary_params(fp, numParams, params) == numParams);

      const size_t headerSize = (size_t) params[INDEX_HEADER_SIZE];
      const size_t recordSize = (size_t) params[INDEX_RECORD_SIZE];
      assert(numItems >= 0 && recordSize == (size_t) numItems);

      const int numRecords = params[INDEX_NUM_RECORDS];
      const int dataSize = params[INDEX_DATA_SIZE];
      const int dataType = params[INDEX_DATA_TYPE];
      const int nxBlocks = params[INDEX_NX_PROCS];
      const int nyBlocks = params[INDEX_NY_PROCS];

      assert(dataSize == (int) pv_sizeof(datatype));
      assert(dataType == PV_FLOAT_TYPE || dataType == PV_INT_TYPE);
      assert(nxBlocks == comm->numCommColumns());
      assert(nyBlocks == comm->numCommRows());
      assert(numRecords == comm->commSize());

      // Position at start of specified level
      long startoflevel = headerSize + level*(nxProcs*nyProcs*localSize+sizeof(double));
      fseek(fp, startoflevel, SEEK_SET);
      // Read time into cbuf where it can be Bcast to all processes.
      numRead = fread(cbuf+sizeof(int), sizeof(double), 1, fp);
      assert(numRead == 1);

#ifdef PV_USE_MPI
      int dest = -1;
      for (int py = 0; py < nyProcs; py++) {
         for (int px = 0; px < nxProcs; px++) {
            if (++dest == 0) continue;

#ifdef DEBUG_OUTPUT
            fprintf(stderr, "[%2d]: read: sending to %d nx==%d ny==%d numItems==%d\n",
                    comm->commRank(), dest, loc->nx, loc->ny, numItems);
#endif // DEBUG_OUTPUT
            long offset = startoflevel + sizeof(double) + dest * localSize;
            fseek(fp, offset, SEEK_SET);
            numRead = fread(cbuf+sizeof(int)+sizeof(double), localSize, 1, fp);
            assert(numRead == 1);
            MPI_Send(cbuf, mpiBufferSize, MPI_BYTE, dest, tag, mpi_comm);
         }
      }
#endif // PV_USE_MPI

      // get local image portion
      fseek(fp, startoflevel, SEEK_SET);
      numRead = fread(cbuf+sizeof(int), sizeof(unsigned char), localSize+sizeof(double), fp);
      assert(numRead == (int) (localSize+sizeof(double)));

      pvp_close_file(fp, comm);
   }

   // copy from buffer communication buffer
   //
   status = *((int *) cbuf);
   if( status == PV_SUCCESS ) {
      if (datatype == PV_FLOAT_TYPE) {
         float * fbuf = (float *) (cbuf+sizeof(int)+sizeof(double));
         status = HyPerLayer::copyFromBuffer(fbuf, (float*) data, loc, extended, scale);
      }
      else if (datatype == PV_INT_TYPE) {
         int * fbuf = (int *) (cbuf+sizeof(int)+sizeof(double));
         status = HyPerLayer::copyFromBuffer(fbuf, (int*) data, loc, extended, 1);
      }
      *time = *((double *) (cbuf+sizeof(int)));
   }
   free(cbuf);
   return status;
}

int write_pvdata(const char * filename, Communicator * comm, double time, const pvdata_t * data,
          const PVLayerLoc * loc, int datatype, bool extended, bool contiguous)
{
   int status = PV_SUCCESS;
   FILE * fp = NULL;

   if (comm->commRank() == 0) {
      // int numItems;
      // if (extended) {
      //    numItems = (loc->nx + 2*loc->nb) * (loc->ny + 2*loc->nb) * loc->nf;
      // }
      // else {
      //    numItems = loc->nx * loc->ny * loc->nf;
      // }
      int numItems = loc->nx * loc->ny * loc->nf;
      const size_t localSize = numItems * pv_sizeof(datatype);

      const bool append = false;
      fp = pvp_open_write_file(filename, comm, append);

      const int numParams = NUM_PAR_BYTE_PARAMS;
      status = pvp_write_header(fp, comm, time, loc, PVP_FILE_TYPE, datatype,
                                1, extended, contiguous, numParams, localSize);
      if (status != PV_SUCCESS) return status;
   }
   status |= write_pvdata(fp, comm, time, data, loc, datatype, extended, contiguous, PVP_FILE_TYPE);
   status |= pvp_close_file(fp, comm);
   
   return status;
}

int write_pvdata(FILE *fp, Communicator * comm, double time, const pvdata_t * data,
          const PVLayerLoc * loc, int datatype, bool extended, bool contiguous, int tag)
{
   int status = PV_SUCCESS;
   int nxBlocks, nyBlocks, numItems;

   // TODO - everything isn't implemented yet so make sure we are using it correctly
   assert(contiguous == false);
   assert(datatype == PV_FLOAT_TYPE);

   // scale factor for floating point conversion
   float scale = 1.0f;

   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();

   const int icRank = comm->commRank();

   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nf;
#ifdef OBSOLETE // Marked obsolete Aug 31, 2011.  Border of extended region doesn't get written, so don't allocate space for it.
   const int nb = loc->nb;

   if (extended) {
      numItems = (nx + 2*nb) * (ny + 2*nb) * nf;
   }
   else {
      numItems = nx * ny * nf;
   }
#endif // OBSOLETE
   numItems = nx * ny * nf;

   const size_t localSize = numItems * pv_sizeof(datatype);

   if (contiguous) {
      nxBlocks = 1;
      nyBlocks = 1;
   }
   else {
      nxBlocks = nxProcs;
      nyBlocks = nyProcs;
   }

   unsigned char * cbuf = (unsigned char *) malloc(localSize);
   assert(cbuf != NULL);

   if (datatype == PV_FLOAT_TYPE) {
      float * fbuf = (float *) cbuf;
      status = HyPerLayer::copyToBuffer(fbuf, data, loc, extended, scale);
   }

#ifdef PV_USE_MPI
   // const int tag = PVP_FILE_TYPE;
   const MPI_Comm mpi_comm = comm->communicator();
#endif // PV_USE_MPI

   if (icRank > 0) {

#ifdef PV_USE_MPI
      const int dest = 0;
      MPI_Send(cbuf, localSize, MPI_BYTE, dest, tag, mpi_comm);
#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: write_pvdata: sent to 0, nx==%d ny==%d numItems==%d\n",
              comm->commRank(), nx, ny, numItems);
#endif // DEBUG_OUTPUT
#endif // PV_USE_MPI

   }
   else {
      assert(fp != NULL);

      // write local image portion
      size_t numWrite = fwrite(cbuf, sizeof(unsigned char), localSize, fp);
      assert(numWrite == localSize);

#ifdef PV_USE_MPI
      int src = -1;
      for (int py = 0; py < nyProcs; py++) {
         for (int px = 0; px < nxProcs; px++) {
            if (++src == 0) continue;  // rank 0 already written
#ifdef DEBUG_OUTPUT
            fprintf(stderr, "[%2d]: write: receiving from %d nx==%d ny==%d numItems==%d\n",
                    comm->commRank(), src, nx, ny, numItems);
#endif // DEBUG_OUTPUT
            MPI_Recv(cbuf, localSize, MPI_BYTE, src, tag, mpi_comm, MPI_STATUS_IGNORE);

            //const int numParams = NUM_PAR_BYTE_PARAMS;
            // !!! do not overwrite previous time steps !!!
            //const int headerSize = numParams * sizeof(int);
            //long offset = headerSize + src * localSize;
            //fseek(fp, offset, SEEK_SET);
            numWrite = fwrite(cbuf, sizeof(unsigned char), localSize, fp);
            fflush(fp); // for debugging
            assert(numWrite == localSize);
         }
      }
#endif // PV_USE_MPI

   }
   free(cbuf);

   return status;
}

int writeActivity(FILE * fp, Communicator * comm, double time, PVLayer * l)
{
   int status;
   bool extended = true; // activity is an extended layer
   bool contiguous = false; // TODO implement contiguous=true case

   // write header, but only at the beginning
#ifdef PV_USE_MPI
   int rank = comm->commRank();
#else // PV_USE_MPI
   int rank = 0;
#endif // PV_USE_MPI
   if( rank == 0 ) {
      long fpos = ftell(fp);
      if (fpos == 0L) {
         int numParams = NUM_BIN_PARAMS;
         status = pvp_write_header(fp, comm, time, &l->loc, PVP_NONSPIKING_ACT_FILE_TYPE,
                                   PV_FLOAT_TYPE, 1/*numbands*/, extended, contiguous, numParams, (size_t) l->numNeurons);
         if (status != PV_SUCCESS) return status;
      }
      // HyPerLayer::writeActivity calls HyPerLayer::incrementNBands, which maintains the value of numbands in the header.

      // write time and V-buffer
      //
      if ( fwrite(&time, sizeof(double), 1, fp) != 1 )              return -1;
   }

   return write_pvdata(fp, comm, time, l->activity->data, &(l->loc), PV_FLOAT_TYPE,
                       extended, contiguous, PVP_NONSPIKING_ACT_FILE_TYPE);
}

int writeActivitySparse(FILE * fp, Communicator * comm, double time, PVLayer * l)
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
      long fpos = ftell(fp);
      if (fpos == 0L) {
         int numParams = NUM_BIN_PARAMS;
         status = pvp_write_header(fp, comm, time, &l->loc, PVP_ACT_FILE_TYPE,
                                   datatype, 1, extended, contiguous, numParams, (size_t) localActive);
         if (status != 0) {
            fprintf(stderr, "[%2d]: writeActivitySparse: failed in pvp_write_header, numParams==%d, localActive==%d\n",
                    comm->commRank(), numParams, localActive);
            return status;
         }
      }

      // write time, total active count, and local activity
      //
      status = (fwrite(&time, sizeof(double), 1, fp) != 1 );
      if (status != 0) {
         fprintf(stderr, "[%2d]: writeActivitySparse: failed in fwrite(&time), time==%f\n",
                 comm->commRank(), time);
         return status;
      }
      status = ( fwrite(&totalActive, sizeof(unsigned int), 1, fp) != 1 );
      if (status != 0) {
         fprintf(stderr, "[%2d]: writeActivitySparse: failed in fwrite(&totalActive), totalActive==%d\n",
                 comm->commRank(), totalActive);
         return status;
      }
     if (localActive > 0) {
         status = (fwrite(indices, sizeof(unsigned int), localActive, fp) != (size_t) localActive );
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
            status = (fwrite(indices, sizeof(unsigned int), numActive[p], fp) != numActive[p] );
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
      bool shmget_owner, bool shmget_flag)
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
      fprintf(stderr, "[%2d]: readWeights: failed in pvp_read_head, numParams==%d\n",
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

   // extra weight parameters, done as a void pointer, since some are int and some are float
   //
   void * wgtParams = &params[NUM_BIN_PARAMS];
   int * wgtIntParams = (int *) wgtParams;
   float * wgtFloatParams = (float *) wgtParams;

   const int nxp = wgtIntParams[INDEX_WGT_NXP];
   const int nyp = wgtIntParams[INDEX_WGT_NYP];
   const int nfp = wgtIntParams[INDEX_WGT_NFP];
   const float minVal = * ((float*) &wgtFloatParams[INDEX_WGT_MIN]);
   const float maxVal = * ((float*) &wgtFloatParams[INDEX_WGT_MAX]);

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
      status = (numPatches*nxProcs*nyProcs != wgtIntParams[INDEX_WGT_NUMPATCHES]);
   }
   else{
      status = ((numPatches != wgtIntParams[INDEX_WGT_NUMPATCHES]));
   }
   if (status != 0) {
      fprintf(stderr, "[%2d]: readWeights: failed in pvp_check_file_header, "
            "numPatches==%d, nxProcs==%d\n, nyProcs==%d, wgtParams[INDEX_WGT_NUMPATCHES]==%d\n",
            comm->commRank(), numPatches, nxProcs, nyProcs, wgtIntParams[INDEX_WGT_NUMPATCHES]);
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
   FILE * fp = pvp_open_read_file(filename, comm);
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

         if (fp == NULL) {
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
            fseek(fp, offset, SEEK_SET);
            int numRead = fread(cbuf, localSize, 1, fp);
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
                  fseek(fp, offset, SEEK_SET);
                  int numRead = fread(cbuf, localSize, 1, fp);
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
            fseek(fp, offset, SEEK_SET);
            int numRead = fread(cbuf, localSize, 1, fp);
            if  (numRead != 1) {
               fprintf(stderr, "[%2d]: readWeights: failed in fread, offset==%ld\n",
                     comm->commRank(), offset);
            }
         }
      }  // if rank == 0

      // set the contents of the weights patches from the unsigned character buffer, cbuf
      //
#ifdef USE_SHMGET
   if (shmget_flag && !shmget_owner){
      continue;
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
   status = pvp_close_file(fp, comm)==PV_SUCCESS ? status : PV_FAILURE;
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

      FILE * fp = pvp_open_write_file(filename, comm, append);

      if (fp == NULL) {
         fprintf(stderr, "PV::writeWeights: ERROR opening file %s\n", filename);
         return -1;
      }

      // use file_type passed as argument to enable different behavior
      status = pvp_write_header(fp, comm, timed, loc, file_type,
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
      unsigned int num_written = fwrite(wgtExtraParams, sizeof(int), numParams, fp);
      free(wgtExtraParams); wgtExtraParams=NULL; wgtExtraIntParams=NULL; wgtExtraFloatParams=NULL;
      if ( num_written != (unsigned int) numParams ) return -1;

      for( int arbor=0; arbor<numArbors; arbor++ ) {
         // write local portion
         // numPatches - each neuron has a patch; pre-synaptic neurons live in extended layer
         PVPatch ** arborPatches = file_type == PVP_KERNEL_FILE_TYPE ? NULL : patches[arbor];
         pvp_copy_patches(cbuf, arborPatches, dataStart[arbor], numPatches, nxp, nyp, nfp, minVal, maxVal, compress);
         size_t numfwritten = fwrite(cbuf, localSize, 1, fp);
         if ( numfwritten != 1 ) return -1;

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
               if ( fwrite(cbuf, localSize, 1, fp) != 1 ) return -1;
            }
         }
#endif // PV_USE_MPI
      } // end loop over arbors
      free(cbuf);
      pvp_close_file(fp, comm);
   } // icRank == 0

   return PV_SUCCESS; // TODO error handling
}  // end writeWeights (all arbors)

} // namespace PV
