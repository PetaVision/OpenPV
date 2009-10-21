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

static void copyTimeToParams(double time, void * buf)
{
   double * dptr = (double *) buf;
   *dptr = time;
}

static double timeFromParams(void * buf)
{
   return * ( (double *) buf );
}


size_t pv_sizeof(int datatype)
{
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

   const int nxny     = nx * ny;
   const int numItems = nxny * numBands;

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

   const int numBands = loc->nBands;

   const int nxny     = nx * ny;
   const int numItems = nxny * numBands;

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
      int params[NUM_PAR_BYTE_PARAMS];

      const int numParams  = NUM_PAR_BYTE_PARAMS;
      const int headerSize = numParams * sizeof(int);

      FILE * fp = fopen(filename, "wb");

      params[INDEX_HEADER_SIZE] = headerSize;
      params[INDEX_NUM_PARAMS]  = numParams;
      params[INDEX_FILE_TYPE]   = PVP_FILE_TYPE;
      params[INDEX_NX]          = loc->nx;
      params[INDEX_NY]          = loc->ny;
      params[INDEX_NF]          = loc->nBands;
      params[INDEX_NUM_RECORDS] = nxBlocks * nyBlocks;
      params[INDEX_RECORD_SIZE] = localSize * nxBlocks * nyBlocks;
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

      copyTimeToParams(time, &params[INDEX_TIME]);

      int numWrite = fwrite(params, sizeof(int), numParams, fp);
      assert(numWrite == numParams);

      // write local image portion
      fseek(fp, (long) headerSize, SEEK_SET);
      numWrite = fwrite(cbuf, sizeof(unsigned char), localSize, fp);
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

} // namespace PV
