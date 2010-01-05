/*
 * fileio.cpp
 *
 *  Created on: Oct 21, 2009
 *      Author: rasmussn
 */

#include "fileio.hpp"
#include "../layers/HyPerLayer.hpp"

#include <assert.h>

#undef DEBUG_OUTPUT

namespace PV {

static void timeToParams(double time, void * params)
{
   double * dptr = (double *) params;
   *dptr = time;
}

static double timeFromParams(void * params)
{
   return * ( (double *) params );
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
   return ( 2*sizeof(unsigned short) + count*pv_sizeof(datatype) );
}

/**
 * Copy patches into an unsigned char buffer
 */
int pvp_copy_patches(unsigned char * buf, size_t bufSize, PVPatch ** patches, int numPatches,
                     int numPatchItems, float minVal, float maxVal)
{
   unsigned char * cptr = buf;

   for (int k = 0; k < numPatches; k++) {
      PVPatch * p = patches[k];
      const pvdata_t * data = p->data;

      unsigned short * nxny = (unsigned short *) cptr;

      nxny[0] = (unsigned short) p->nx;
      nxny[1] = (unsigned short) p->ny;

      cptr += 2 * sizeof(unsigned short);

      int nItems = (int) nxny[0] * (int) nxny[1] * (int) p->nf;

      for (int i = 0; i < nItems; i++) {
         float val = 255.0 * (data[i] - minVal) / (maxVal - minVal);
         *cptr++ = (unsigned char) val;
      }

      // write leftover null characters
      int nExtra = numPatchItems - nItems;

      for (int i = 0; i < nExtra; i++) {
         *cptr++ = (unsigned char) 0;
      }
   }

   return 0;
}

/**
 * Set patches given an unsigned char input buffer
 */
int pvp_set_patches(unsigned char * buf, size_t bufSize, PVPatch ** patches, int numPatches,
                    int numPatchItems, int nfp, float minVal, float maxVal)
{
   unsigned char * cptr = buf;

   for (int k = 0; k < numPatches; k++) {
      PVPatch * p = patches[k];
      pvdata_t * data = p->data;

      unsigned short * nxny = (unsigned short *) cptr;

      p->nx = nxny[0];
      p->ny = nxny[1];
      p->nf = nfp;

      p->sf = 1;
      p->sx = p->nf;
      p->sy = p->sx * p->nx;

      cptr += 2 * sizeof(unsigned short);

      int nItems = (int) p->nx * (int) p->ny * (int) p->nf;

      for (int i = 0; i < nItems; i++) {
         // data are packed into chars
         float val = (float) *cptr++;
         data[i] = minVal + (maxVal - minVal) * (val / 255.0);
      }

      // skip leftover null characters
      int nExtra = numPatchItems - nItems;
      cptr += nExtra;
   }

   return 0;
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
   }
   return fp;
}

int pvp_close_file(FILE * fp, Communicator * comm)
{
   int status = 0;
   if (fp != NULL) {
      status = fclose(fp);
   }
   return status;
}

int pvp_check_file_header(const LayerLoc * loc, int params[], int numParams)
{
   int status = 0;

   if (loc->nx       != params[INDEX_NX])        status = -1;
   if (loc->ny       != params[INDEX_NY])        status = -1;
   if (loc->nBands   != params[INDEX_NF])        status = -1;
   if (loc->nxGlobal != params[INDEX_NX_GLOBAL]) status = -1;
   if (loc->nyGlobal != params[INDEX_NY_GLOBAL]) status = -1;
//   if (loc->kx0      != params[INDEX_KX0])       status = -1;
//   if (loc->ky0      != params[INDEX_KY0])       status = -1;
   if (loc->nPad     != params[INDEX_NPAD])      status = -1;
   if (loc->nBands   != params[INDEX_NBANDS])    status = -1;

   if (status != 0) {
      for (int i = 0; i < numParams; i++) {
         fprintf(stderr, "params[%d]==%d ", i, params[i]);
      }
      fprintf(stderr, "\n");
   }

   return status;
}

static
int pvp_read_header(FILE * fp, double * time, int * filetype,
                    int * datatype, int params[], int * numParams)
{
   int status = 0;

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
   int status = 0;
   const int icRank = comm->commRank();

   if (icRank == 0) {
       FILE * fp = pvp_open_read_file(filename, comm);
       if (fp == NULL) return -1;

       status = pvp_read_header(fp, time, filetype, datatype, params, numParams);
       pvp_close_file(fp, comm);
       if (status != 0) return status;
   }

#ifdef PV_USE_MPI
   const int icRoot = 0;
#ifdef DEBUG_OUTPUT
   fprintf(stderr, "[%2d]: pvp_read_header: will broadcast, numParams==%d\n",
           comm->commRank(), *numParams);
#endif

   status = MPI_Bcast(params, *numParams, MPI_INT, icRoot, comm->communicator());

#ifdef DEBUG_OUTPUT
   fprintf(stderr, "[%2d]: pvp_read_header: broadcast completed, numParams==%d\n",
           comm->commRank(), *numParams);
#endif
#endif // PV_USE_MPI

   *filetype = params[INDEX_FILE_TYPE];
   *datatype = params[INDEX_DATA_TYPE];
   *time = timeFromParams(&params[INDEX_TIME]);

   return status;
}

int pvp_write_header(FILE * fp, Communicator * comm, double time, const LayerLoc * loc, int filetype,
                     int datatype, int subRecordSize, bool extended, bool contiguous, unsigned int numParams)
{
   int status = 0;
   int nxBlocks, nyBlocks;
   int params[NUM_BIN_PARAMS];

   if (comm->commRank() != 0) return status;

   const int headerSize = numParams * sizeof(int);

   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();

   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nBands;

   const int numItems = nx * ny * nf;

   if (contiguous) {
      nxBlocks = 1;
      nyBlocks = 1;
   }
   else {
      nxBlocks = nxProcs;
      nyBlocks = nyProcs;
   }

   const int localSize = numItems * subRecordSize;
   const size_t globalSize = localSize * nxBlocks * nyBlocks;

   // make sure we don't blow out size of int for record size
   assert(globalSize < 0xffffffff);

   params[INDEX_HEADER_SIZE] = headerSize;
   params[INDEX_NUM_PARAMS]  = numParams;
   params[INDEX_FILE_TYPE]   = filetype;
   params[INDEX_NX]          = loc->nx;
   params[INDEX_NY]          = loc->ny;
   params[INDEX_NF]          = loc->nBands;
   params[INDEX_NUM_RECORDS] = nxBlocks * nyBlocks;  // one record could be one node or all nodes
   params[INDEX_RECORD_SIZE] = localSize;
   params[INDEX_DATA_SIZE]   = pv_sizeof(datatype);
   params[INDEX_DATA_TYPE]   = datatype;
   params[INDEX_NX_PROCS]    = nxBlocks;
   params[INDEX_NY_PROCS]    = nyBlocks;
   params[INDEX_NX_GLOBAL]   = loc->nxGlobal;
   params[INDEX_NY_GLOBAL]   = loc->nyGlobal;
   params[INDEX_KX0]         = loc->kx0;
   params[INDEX_KY0]         = loc->ky0;
   params[INDEX_NPAD]        = loc->nPad;
   params[INDEX_NBANDS]      = loc->nBands;

   timeToParams(time, &params[INDEX_TIME]);

   numParams = NUM_BIN_PARAMS;  // there may be more to come
   if ( fwrite(params, sizeof(int), numParams, fp) != numParams ) {
      status = -1;
   }

   return status;
}

int read(const char * filename, Communicator * comm, double * time, pvdata_t * data,
         const LayerLoc * loc, int datatype, bool extended, bool contiguous)
{
   int status = 0;
   int nxBlocks, nyBlocks;

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

   const int numBands = loc->nBands;
   const int numItems = nx * ny * numBands;

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

#ifdef PV_USE_MPI
   const int tag = PVP_FILE_TYPE;
   const MPI_Comm mpi_comm = comm->communicator();
#endif // PV_USE_MPI

   if (icRank > 0) {

#ifdef PV_USE_MPI
      const int src = 0;
      MPI_Recv(cbuf, localSize, MPI_BYTE, src, tag, mpi_comm, MPI_STATUS_IGNORE);

#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: read: received from 0, nx==%d ny==%d numItems==%d\n",
              comm->commRank(), nx, ny, numItems);
#endif
#endif // PV_USE_MPI
   }
   else {
      int params[NUM_PAR_BYTE_PARAMS];
      int numParams, numRead, type, nxIn, nyIn, nfIn;

      FILE * fp = pv_open_binary(filename, &numParams, &type, &nxIn, &nyIn, &nfIn);
      assert(fp != NULL);
      assert(numParams == NUM_PAR_BYTE_PARAMS);
      assert(type      == PVP_FILE_TYPE);

      status = pv_read_binary_params(fp, numParams, params);
      assert(status == numParams);

      const size_t headerSize = (size_t) params[INDEX_HEADER_SIZE];
      const size_t recordSize = (size_t) params[INDEX_RECORD_SIZE];

      const int numRecords = params[INDEX_NUM_RECORDS];
      const int dataSize = params[INDEX_DATA_SIZE];
      const int dataType = params[INDEX_DATA_TYPE];
      const int nxBlocks = params[INDEX_NX_PROCS];
      const int nyBlocks = params[INDEX_NY_PROCS];

//      loc->nx       = params[INDEX_NX];
//      loc->ny       = params[INDEX_NY];
//      loc->nxGlobal = params[INDEX_NX_GLOBAL];
//      loc->nyGlobal = params[INDEX_NY_GLOBAL];
//      loc->kx0      = params[INDEX_KX0];
//      loc->ky0      = params[INDEX_KY0];
//      loc->nPad     = params[INDEX_NPAD];
//      loc->nBands   = params[INDEX_NBANDS];

      *time = timeFromParams(&params[INDEX_TIME]);

      assert(dataSize == (int) pv_sizeof(datatype));
      assert(dataType == PV_FLOAT_TYPE);
      assert(nxBlocks == comm->numCommColumns());
      assert(nyBlocks == comm->numCommRows());
      assert(numRecords == comm->commSize());

#ifdef PV_USE_MPI
      int dest = -1;
      for (int py = 0; py < nyProcs; py++) {
         for (int px = 0; px < nxProcs; px++) {
            if (++dest == 0) continue;

#ifdef DEBUG_OUTPUT
            fprintf(stderr, "[%2d]: read: sending to %d nx==%d ny==%d numItems==%d\n",
                    comm->commRank(), dest, nx, ny, numItems);
#endif
            long offset = headerSize + dest * recordSize;
            fseek(fp, offset, SEEK_SET);
            numRead = fread(cbuf, sizeof(unsigned char), localSize, fp);
            assert(numRead == (int) localSize);
            MPI_Send(cbuf, localSize, MPI_BYTE, dest, tag, mpi_comm);
         }
      }
#endif // PV_USE_MPI

      // get local image portion
      fseek(fp, (long) headerSize, SEEK_SET);
      numRead = fread(cbuf, sizeof(unsigned char), localSize, fp);
      assert(numRead == (int) localSize);

      // copy from buffer communication buffer
      //
      if (datatype == PV_FLOAT_TYPE) {
         float * fbuf = (float *) cbuf;
         status = HyPerLayer::copyFromBuffer(fbuf, data, loc, extended, scale);
      }

      free(cbuf);
      status = pvp_close_file(fp, comm);
   }

   return status;
}

int write(const char * filename, Communicator * comm, double time, pvdata_t * data,
          const LayerLoc * loc, int datatype, bool extended, bool contiguous)
{
   int status = 0;
   int nxBlocks, nyBlocks;

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
   const int nf = loc->nBands;
   const int numItems = nx * ny * nf;

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
   const int tag = PVP_FILE_TYPE;
   const MPI_Comm mpi_comm = comm->communicator();
#endif // PV_USE_MPI

   if (icRank > 0) {

#ifdef PV_USE_MPI
      const int dest = 0;
      MPI_Send(cbuf, localSize, MPI_BYTE, dest, tag, mpi_comm);
#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: write: sent to 0, nx==%d ny==%d numItems==%d\n",
              comm->commRank(), nx, ny, numItems);
#endif
#endif // PV_USE_MPI

   }
   else {
      const bool append = false;
      const int numParams = NUM_PAR_BYTE_PARAMS;
      const int headerSize = numParams * sizeof(int);

      FILE * fp = pvp_open_write_file(filename, comm, append);

      status = pvp_write_header(fp, comm, time, loc, PVP_FILE_TYPE,
                                datatype, pv_sizeof(datatype), extended, contiguous, numParams);
      if (status != 0) return status;

      // write local image portion
      size_t numWrite = fwrite(cbuf, sizeof(unsigned char), localSize, fp);
      assert(numWrite == localSize);

#ifdef PV_USE_MPI
      int src = -1;
      for (int py = 0; py < nyProcs; py++) {
         for (int px = 0; px < nxProcs; px++) {
            if (++src == 0) continue;
#ifdef DEBUG_OUTPUT
            fprintf(stderr, "[%2d]: write: receiving from %d nx==%d ny==%d numItems==%d\n",
                    comm->commRank(), src, nx, ny, numItems);
#endif
            MPI_Recv(cbuf, localSize, MPI_BYTE, src, tag, mpi_comm, MPI_STATUS_IGNORE);

            long offset = headerSize + src * localSize;
            fseek(fp, offset, SEEK_SET);
            numWrite = fwrite(cbuf, sizeof(unsigned char), localSize, fp);
            assert(numWrite == localSize);
         }
      }
#endif // PV_USE_MPI

      free(cbuf);
      status = pvp_close_file(fp, comm);
   }

   return status;
}

int writeActivitySparse(FILE * fp, Communicator * comm, double time, PVLayer * l)
{
   int status = 0;

   const int icRoot = 0;
   const int icRank = comm->commRank();
   int localActive = l->numActive;
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
      fprintf(stderr, "[%2d]: writeActivitySparse: sent to 0, localActive==%d\n",
              comm->commRank(), localActive);
#endif
#endif // PV_USE_MPI

   }
   else {
      // we are io root process
      //
      unsigned int totalActive = localActive;

#ifdef PV_USE_MPI
      // get the number active from each process
      // TODO - use collective?
      //
      const int icSize = comm->commSize();
      unsigned int * numActive = (unsigned int *) malloc(icSize*sizeof(int));
      for (int p = 1; p < icSize; p++) {
         MPI_Recv(&numActive[p], 1, MPI_INT, p, tag, mpi_comm, MPI_STATUS_IGNORE);
         totalActive += numActive[p];
      }
#else
      unsigned int numActive[1];
      numActive[0] = totalActive;
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
                                   datatype, sizeof(int), extended, contiguous, numParams);
         if (status != 0) return status;
      }

      // write time, total active count, and local activity
      //
      if ( fwrite(&time, sizeof(double), 1, fp) != 1 )              return -1;
      if ( fwrite(&totalActive, sizeof(unsigned int), 1, fp) != 1 ) return -1;
      if (totalActive > 0) {
         if ( fwrite(indices, sizeof(unsigned int), numActive[0], fp) != numActive[0] ) {
            return -1;
         }
      }

#ifdef PV_USE_MPI
      for (int p = 1; p < icSize; p++) {
#ifdef DEBUG_OUTPUT
            fprintf(stderr, "[%2d]: writeActivitySparse: receiving from %d numActive==%d\n",
                    comm->commRank(), p, numActive[p]);
#endif
            MPI_Recv(indices, numActive[p], MPI_INT, p, tag, mpi_comm, MPI_STATUS_IGNORE);
            if ( fwrite(indices, sizeof(unsigned int), numActive[p], fp) != numActive[p] ) return -1;
      }
      free(numActive);
#endif // PV_USE_MPI
   }

   return status;
}

int readWeights(PVPatch ** patches, int numPatches, const char * filename,
                Communicator * comm, double * time, const LayerLoc * loc, bool extended)
{
   int status = 0;
   int filetype, datatype;

   int numParams = NUM_WGT_PARAMS;
   int params[NUM_WGT_PARAMS];

   int nxBlocks, nyBlocks;

   bool contiguous = false;   // for now

   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();

   const int icRank = comm->commRank();

   // read file header (uses MPI to broadcast the results)
   //
   status = pvp_read_header(filename, comm, time, &filetype, &datatype, params, &numParams);
   if (status != 0) return status;

   status = pvp_check_file_header(loc, params, numParams);
   if (status < 0) {
      fprintf(stderr, "[%2d]: readWeights: failed in pvp_check_file_header, numParams==%d\n",
              comm->commRank(), numParams);
      return status;
   }

   // extra weight parameters
   //
   int * wgtParams = &params[NUM_BIN_PARAMS];

   const int nxp = wgtParams[INDEX_WGT_NXP];
   const int nyp = wgtParams[INDEX_WGT_NYP];
   const int nfp = wgtParams[INDEX_WGT_NFP];
   const float minVal = wgtParams[INDEX_WGT_MIN];
   const float maxVal = wgtParams[INDEX_WGT_MAX];

   if (numPatches != wgtParams[INDEX_WGT_NUMPATCHES]) return -1;
   if (datatype != PV_BYTE_TYPE)                      return -1;

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
   assert(cbuf != NULL);

#ifdef PV_USE_MPI
   const int tag = PVP_WGT_FILE_TYPE;
   const MPI_Comm mpi_comm = comm->communicator();
#endif // PV_USE_MPI

   // read weights and send using MPI
   //
   if (icRank > 0) {

#ifdef PV_USE_MPI
      const int src = 0;
#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: readWeights: recv from %d, nxBlocks==%d nyBlocks==%d numPatches==%d\n",
              comm->commRank(), src, nxBlocks, nyBlocks, numPatches);
#endif
      MPI_Recv(cbuf, localSize, MPI_BYTE, src, tag, mpi_comm, MPI_STATUS_IGNORE);
#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: readWeights: recv from %d completed\n",
              comm->commRank(), src);
#endif
#endif // PV_USE_MPI

   }
   else {
      FILE * fp = pvp_open_read_file(filename, comm);
      const int headerSize = numParams * sizeof(int);

      if (fp == NULL) {
         fprintf(stderr, "PV::readWeights: ERROR opening file %s\n", filename);
         return -1;
      }

#ifdef PV_USE_MPI
      int dest = -1;
      for (int py = 0; py < nyProcs; py++) {
         for (int px = 0; px < nxProcs; px++) {
            if (++dest == 0) continue;
#ifdef DEBUG_OUTPUT
            fprintf(stderr, "[%2d]: readWeights: sending to %d nxProcs==%d nyProcs==%d localSize==%ld\n",
                    comm->commRank(), dest, nxProcs, nyProcs, localSize);
#endif
            long offset = headerSize + dest * localSize;
            fseek(fp, offset, SEEK_SET);
            if ( fread(cbuf, localSize, 1, fp) != 1 ) return -1;

            MPI_Send(cbuf, localSize, MPI_BYTE, dest, tag, mpi_comm);
#ifdef DEBUG_OUTPUT
            fprintf(stderr, "[%2d]: readWeights: sending to %d completed\n",
                    comm->commRank(), dest);
#endif
         }
      }
#endif // PV_USE_MPI

      // read local portion
      // numPatches - each neuron has a patch; pre-synaptic neurons live in extended layer
      //
      long offset = headerSize + 0 * localSize;
      fseek(fp, offset, SEEK_SET);
      if ( fread(cbuf, localSize, 1, fp) != 1 ) return -1;

      status = pvp_close_file(fp, comm);
   }

   // set the contents of the weights patches from the unsigned character buffer, cbuf
   //
   status = pvp_set_patches(cbuf, localSize, patches,
                            numPatches, numPatchItems, nfp, minVal, maxVal);

   free(cbuf);

   return status;
}

/*!
 *
 * numPatches is NX x NY x NF
 * patchSize includes these records:
 * - nx
 * - ny
 * - nxp x nyp weights
 * .
 *
 */
int writeWeights(const char * filename, Communicator * comm, double time, bool append,
                 const LayerLoc * loc, int nxp, int nyp, int nfp, float minVal, float maxVal,
                 PVPatch ** patches, int numPatches)
{
   int status = 0;
   int nxBlocks, nyBlocks;

   bool extended = true;      // this shouldn't matter (TODO - get rid of it)
   bool contiguous = false;   // for now

   int datatype = PV_BYTE_TYPE;

   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();

   const int icRank = comm->commRank();

// TODO - do I need to check this??
   //   assert(numPatches == nx*ny*nf);

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
   assert(cbuf != NULL);

   pvp_copy_patches(cbuf, localSize, patches, numPatches, numPatchItems, minVal, maxVal);

#ifdef PV_USE_MPI
   const int tag = PVP_WGT_FILE_TYPE;
   const MPI_Comm mpi_comm = comm->communicator();
#endif // PV_USE_MPI

   if (icRank > 0) {

#ifdef PV_USE_MPI
      const int dest = 0;
      MPI_Send(cbuf, localSize, MPI_BYTE, dest, tag, mpi_comm);
#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: writeWeights: sent to 0, nxBlocks==%d nyBlocks==%d numPatches==%d\n",
              comm->commRank(), nxBlocks, nyBlocks, numPatches);
#endif
#endif // PV_USE_MPI

   }
   else {
      int params[NUM_WGT_EXTRA_PARAMS];

      int numParams = NUM_WGT_PARAMS;
      const int headerSize = numParams * sizeof(int);

      FILE * fp = pvp_open_write_file(filename, comm, append);

      if (fp == NULL) {
         fprintf(stderr, "PV::writeWeights: ERROR opening file %s\n", filename);
         return -1;
      }

      status = pvp_write_header(fp, comm, time, loc, PVP_WGT_FILE_TYPE,
                                datatype, patchSize, extended, contiguous, numParams);
      if (status != 0) return status;

      // write extra weight parameters
      //
      params[INDEX_WGT_NXP] = nxp;
      params[INDEX_WGT_NYP] = nyp;
      params[INDEX_WGT_NFP] = nfp;
      params[INDEX_WGT_MIN] = (int) minVal;
      params[INDEX_WGT_MAX] = (int) maxVal;
      params[INDEX_WGT_NUMPATCHES] = numPatches;

      numParams = NUM_WGT_EXTRA_PARAMS;
      if ( fwrite(params, sizeof(int), numParams, fp) != (unsigned int) numParams ) return -1;

      // write local portion
      // numPatches - each neuron has a patch; pre-synaptic neurons live in extended layer
      //
      if ( fwrite(cbuf, localSize, 1, fp) != 1 ) return -1;

#ifdef PV_USE_MPI
      int src = -1;
      for (int py = 0; py < nyProcs; py++) {
         for (int px = 0; px < nxProcs; px++) {
            if (++src == 0) continue;
#ifdef DEBUG_OUTPUT
            fprintf(stderr, "[%2d]: writeWeights: receiving from %d nxProcs==%d nyProcs==%d localSize==%ld\n",
                    comm->commRank(), src, nxProcs, nyProcs, localSize);
#endif
            MPI_Recv(cbuf, localSize, MPI_BYTE, src, tag, mpi_comm, MPI_STATUS_IGNORE);

            long offset = headerSize + src * localSize;
            fseek(fp, offset, SEEK_SET);
            if ( fwrite(cbuf, localSize, 1, fp) != 1 ) return -1;
         }
      }
#endif // PV_USE_MPI

      free(cbuf);
      status = pvp_close_file(fp, comm);
   }

   return status;
}

} // namespace PV
