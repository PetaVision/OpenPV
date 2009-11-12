/*
 * fileio.cpp
 *
 *  Created on: Oct 21, 2009
 *      Author: rasmussn
 */

#include "fileio.hpp"
#include "../layers/HyPerLayer.hpp"

#include <assert.h>

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

int copy_patches(unsigned char * buf, size_t bufSize, PVPatch ** patches, int numPatches,
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

FILE * pvp_open_read_file(const char * filename)
{
   return fopen(filename, "rb");
}

FILE * pvp_open_write_file(const char * filename, bool append)
{
   FILE * fp;
   if (append) fp = fopen(filename, "ab");
   else        fp = fopen(filename, "wb");

   return fp;
}

int pvp_read_header(FILE * fp, double * time, const LayerLoc * loc, int * filetype,
                    int * datatype, int params[], int * numParams)
{
   int status = 0;

   // find out how many parameters there are
   //
   if ( fread(params, sizeof(int), 2, fp) != 2 ) return -1;

   int nParams = params[INDEX_NUM_PARAMS];
   assert(params[INDEX_HEADER_SIZE] == (int) (nParams * sizeof(int)));
   assert(nParams <= *numParams);

   // read the rest
   //
   if ( fread(&params[2], sizeof(int), nParams - 2, fp) != (unsigned int) nParams - 2 ) return -1;

   *numParams  = params[INDEX_NUM_PARAMS];
   *filetype   = params[INDEX_FILE_TYPE];
   *datatype   = params[INDEX_DATA_TYPE];

   // make sure the parameters are what we are expecting
   //

   assert(params[INDEX_DATA_SIZE] == (int) pv_sizeof(*datatype));

   assert(loc->nx       == params[INDEX_NX]);
   assert(loc->ny       == params[INDEX_NY]);
   assert(loc->nBands   == params[INDEX_NF]);
   assert(loc->nxGlobal == params[INDEX_NX_GLOBAL]);
   assert(loc->nyGlobal == params[INDEX_NY_GLOBAL]);
   assert(loc->kx0      == params[INDEX_KX0]);
   assert(loc->ky0      == params[INDEX_KY0]);
   assert(loc->nPad     == params[INDEX_NPAD]);
   assert(loc->nBands   == params[INDEX_NBANDS]);

   *time = timeFromParams(&params[INDEX_TIME]);

   return status;
}

int pvp_write_header(FILE * fp, Communicator * comm, double time, const LayerLoc * loc, int filetype,
                     int datatype, int subRecordSize, bool extended, bool contiguous, unsigned int numParams)
{
   int status = 0;
   int nxBlocks, nyBlocks;
   int params[NUM_BIN_PARAMS];

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

   const size_t localSize  = numItems * subRecordSize;
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
   params[INDEX_RECORD_SIZE] = (unsigned int) globalSize;
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
      const int tag = PVP_FILE_TYPE;
      const MPI_Comm mpi_comm = comm->communicator();

      MPI_Recv(cbuf, localSize, MPI_BYTE, src, tag, mpi_comm, MPI_STATUS_IGNORE);

#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: read: received from 0, nx==%d ny==%d numItems==%d\n",
              comm->commRank(), nx, ny, numItems);
#endif
#endif // PV_USE_MPI

#ifdef PV_USE_MPI
      const int dest = 0;
      MPI_Send(cbuf, size, MPI_BYTE, dest, tag, mpi_comm);
#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: write: sent to 0, nx==%d ny==%d numItems==%d\n",
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

      assert(dataSize == pv_sizeof(datatype));
      assert(dataType == PV_FLOAT_TYPE);
      assert(nxBlocks == comm->numCommColumns());
      assert(nyBlocks == comm->numCommRows());
      assert(numRecords == comm->commSize());

#ifdef PV_USE_MPI
      int dest = -1;
      const int tag = PVP_FILE_TYPE;
      const MPI_Comm mpi_comm = comm->communicator();

      for (int py = 0; py < nyProcs; py++) {
         for (int px = 0; px < nxProcs; px++) {
            if (++dest == 0) continue;

#ifdef DEBUG_OUTPUT
            fprintf(stderr, "[%2d]: read: sending to %d xSize==%d"
                    " ySize==%d size==%d total==%d\n",
                    comm->commRank(), dest, nx, ny, nx*ny,
                    nx*ny*comm->commSize());
#endif
            long offset = headerSize + dest * recordSize;
            fseek(fp, offset, SEEK_SET);
            numRead = fread(buf, sizeof(unsigned char), localSize, fp);
            assert(numRead == localSize);
            MPI_Send(sbuf, localSize, MPI_BYTE, dest, tag, mpi_comm);
         }
      }
#endif // PV_USE_MPI

      // get local image portion
      fseek(fp, (long) headerSize, SEEK_SET);
      numRead = fread(cbuf, sizeof(unsigned char), localSize, fp);
      assert(numRead == localSize);

      // copy from buffer communication buffer
      //
      if (datatype == PV_FLOAT_TYPE) {
         float * fbuf = (float *) cbuf;
         status = HyPerLayer::copyFromBuffer(fbuf, data, loc, extended, scale);
      }

      free(cbuf);
      status = pv_close_binary(fp);
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
      bool append = false;
      FILE * fp = pvp_open_write_file(filename, append);

//      int params[NUM_PAR_BYTE_PARAMS];
      int numParams = NUM_PAR_BYTE_PARAMS;

      status = pvp_write_header(fp, comm, time, loc, PVP_FILE_TYPE,
                                datatype, pv_sizeof(datatype), extended, contiguous, numParams);
      if (status != 0) return status;

      // write local image portion
//      fseek(fp, (long) headerSize, SEEK_SET);
      size_t numWrite = fwrite(cbuf, sizeof(unsigned char), localSize, fp);
      assert(numWrite == localSize);

#ifdef PV_USE_MPI
      int src = -1;
      for (int py = 0; py < nyProcs; py++) {
         for (int px = 0; px < nxProcs; px++) {
            if (++src == 0) continue;
#ifdef DEBUG_OUTPUT
            fprintf(stderr, "[%2d]: gather: receiving from %d xSize==%d"
                    " ySize==%d size==%d total==%d\n",
                    comm->commRank(), src, nx, ny, numTotal,
                    numTotal*comm->commSize());
#endif
            MPI_Recv(buf, numItems, MPI_BYTE, src, tag, mpi_comm, MPI_STATUS_IGNORE);

            long offset = headerSize + src * localSize;
            fseek(fp, offset, SEEK_SET);
            numWrite = fwrite(buf, sizeof(unsigned char), numItems, fp);
            assert(numWrite == numItems);
         }
      }
#endif // PV_USE_MPI

      free(cbuf);
      status = fclose(fp);
   }

   return status;
}

int writeActivitySparse(FILE * fp, Communicator * comm, double time, PVLayer * l)
{
   int status = 0;

   const int icRoot = 0;
   const int icRank = comm->commRank();
   const int localActive = l->numActive;
   const unsigned int * indices = l->activeIndices;

#ifdef PV_USE_MPI
   const int tag = PVP_ACT_FILE_TYPE;
   const MPI_Comm mpi_comm = comm->communicator();
#endif // PV_USE_MPI

   if (icRank != icRoot) {

#ifdef PV_USE_MPI
      const int dest = icRoot;
      MPI_Send(&localActive, 1, MPI_INTEGER, dest, tag, mpi_comm);
      MPI_Send(indices, numActive, MPI_INTEGER, dest, tag, mpi_comm);
#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: writeActivity: sent to 0, nx==%d ny==%d numItems==%d\n",
              comm->commRank(), nx, ny, numActive);
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
         MPI_Recv(&numActive[p], 1, MPI_INTEGER, p, tag, mpi_comm);
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
            fprintf(stderr, "[%2d]: writeActivitySparse: receiving from %d xSize==%d"
                    " ySize==%d numActive==%d total==%d\n",
                    comm->commRank(), p, nx, ny, numActive[p], numTotal*comm->commSize());
#endif
            MPI_Recv(indices, numActive[p], MPI_INTEGER, p, tag, mpi_comm);
            if ( fwrite(indices, sizeof(unsigned int), numActive[p], fp) != numActive[p] ) return -1;
            assert(numWrite == numActive);
         }
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
   int filetype, datatype, numParams;
   int params[NUM_WGT_PARAMS];

   FILE * fp = pvp_open_read_file(filename);
   if (fp == NULL) {
      fprintf(stderr, "PV::readWeights: ERROR opening file %s\n", filename);
      return -1;
   }

   numParams = NUM_WGT_PARAMS;
   status = pvp_read_header(fp, time, loc, &filetype, &datatype, params, &numParams);

   // extra weight parameters
   //
   int * wgtParams = &params[NUM_BIN_PARAMS];

   const int nxp = wgtParams[INDEX_WGT_NXP];
   const int nyp = wgtParams[INDEX_WGT_NYP];
   const int nfp = wgtParams[INDEX_WGT_NFP];
   const float minVal = wgtParams[INDEX_WGT_MIN];
   const float maxVal = wgtParams[INDEX_WGT_MAX];

   assert(numPatches = wgtParams[INDEX_WGT_NUMPATCHES]);

   status = pv_read_patches(fp, nxp, nyp, nfp, minVal, maxVal, numPatches, patches);
   fclose(fp);

   return status;
}

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

   copy_patches(cbuf, localSize, patches, numPatches, numPatchItems, minVal, maxVal);

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
      FILE * fp = pvp_open_write_file(filename, append);

      int params[NUM_WGT_EXTRA_PARAMS];

      int numParams = NUM_WGT_PARAMS;

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

      // write local image portion
      // numPatches - each neuron has a patch; pre-synaptic neurons live in extended layer
      //
      if ( fwrite(cbuf, localSize, 1, fp) != 1 ) return -1;

#ifdef PV_USE_MPI
      int src = -1;
      for (int py = 0; py < nyProcs; py++) {
         for (int px = 0; px < nxProcs; px++) {
            if (++src == 0) continue;
#ifdef DEBUG_OUTPUT
            fprintf(stderr, "[%2d]: writeWeights: receiving from %d xSize==%d"
                    " ySize==%d size==%d total==%d\n",
                    comm->commRank(), src, nx, ny, localSize,
                    numTotal*comm->commSize());
#endif
            MPI_Recv(cbuf, localSize, MPI_BYTE, src, tag, mpi_comm, MPI_STATUS_IGNORE);

            long offset = headerSize + src * localSize;
            fseek(fp, offset, SEEK_SET);
            numWrite = fwrite(buf, sizeof(unsigned char), numItems, fp);
            assert(numWrite == numItems);
         }
      }
#endif // PV_USE_MPI

      free(cbuf);
      status = fclose(fp);
   }

   return status;
}

} // namespace PV
