/*
 * test_read_header.cpp
 *
 *  Created on: Dec 30, 2009
 *      Author: Craig Rasmussen
 */

#include <columns/Communicator.hpp>
#include <io/fileio.hpp>

#include <stdio.h>
#include <stdlib.h>

#undef DEBUG_OUTPUT

#include <arch/mpi/mpi.h>
#include <columns/PV_Init.hpp>

// function declarations that are not public in fileio.hpp
//
namespace PV {
   size_t pv_sizeof(int datatype);
}

const char filename[]  = "output/test_read_header.pvp";

using namespace PV;

int main(int argc, char* argv[])
{
   PV_Init* initObj = new PV_Init(&argc, &argv, false/*allowUnrecognizedArguments*/);
   int status = 0;
   PV_Stream * stream = NULL;
   double time, write_time = 33.3;

   int params[NUM_BIN_PARAMS];
   int filetype, write_filetype = PVP_FILE_TYPE;
   int datatype, write_datatype = PV_FLOAT_TYPE;
   int numParams = NUM_BIN_PARAMS, write_num_params = NUM_BIN_PARAMS;

   const int subRecordSize = pv_sizeof(write_datatype);

   const bool extended = false;
   const bool contiguous = false;

   // create the Communicator object for MPI
   //
   Communicator * comm = new Communicator(initObj->getArguments());
   int numRows = comm->numCommRows();
   int numCols = comm->numCommColumns();
   PVLayerLoc loc;
   loc.nx = 16;
   loc.ny = 8;
   loc.nf = 3;
   loc.nxGlobal = loc.nx*numCols;
   loc.nyGlobal = loc.ny*numRows;
   loc.kx0 = 16*comm->commRow();
   loc.ky0 = 8*comm->commColumn();

   loc.halo.lt = 0;
   loc.halo.rt = 0;
   loc.halo.dn = 0;
   loc.halo.up = 0;

   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();

   // NULL return value normal for rank > 0
   stream = pvp_open_write_file(filename, comm, false);

   status = pvp_write_header(stream, comm, write_time, &loc,
                             write_filetype,
                             write_datatype, subRecordSize,
                             extended, contiguous, write_num_params,
                             (size_t) loc.nx * loc.ny * loc.nf);

   PV::pvp_close_file(stream, comm);
   if (status != 0) return status;

   MPI_Barrier(comm->communicator());

   status = pvp_read_header(filename, comm, &time, &filetype, &datatype,
                            params, &numParams);

   if (numParams != write_num_params) {
      fprintf(stderr, "numParams != write_num_params, %d %d\n",
              numParams, write_num_params);
      status = -1;
   }

#ifdef DEBUG_OUTPUT
   for (int i = 0; i < numParams - 2; i++) {
      printf("params[%d] = %d\n", i, params[i]);
   }
#endif

   if (time != write_time) status = -1;
   if (filetype != write_filetype) status = -1;
   if (datatype != write_datatype) status = -1;

   const int size_float = (int) sizeof(float);
   const int numRecords = nxProcs * nyProcs;
   const int localSize  = loc.nx * loc.ny * loc.nf * subRecordSize;

   if (params[INDEX_HEADER_SIZE] != numParams*size_float) {
      status = -INDEX_HEADER_SIZE;
   }
   if (params[INDEX_NUM_PARAMS]  != numParams)      status = -INDEX_NUM_PARAMS;
   if (params[INDEX_FILE_TYPE]   != filetype)       status = -INDEX_FILE_TYPE;
   if (params[INDEX_NX]          != loc.nx)         status = -INDEX_NX;
   if (params[INDEX_NY]          != loc.ny)         status = -INDEX_NY;
   if (params[INDEX_NF]          != loc.nf)     status = -INDEX_NF;
   if (params[INDEX_NUM_RECORDS] != numRecords)     status = -INDEX_NUM_RECORDS;
   if (params[INDEX_RECORD_SIZE] != localSize)      status = -INDEX_RECORD_SIZE;

   if (params[INDEX_DATA_SIZE]   != size_float)     status = -INDEX_DATA_SIZE;

   if (params[INDEX_NX_PROCS]    != nxProcs)        status = -INDEX_NX_PROCS;
   if (params[INDEX_NY_PROCS]    != nyProcs)        status = -INDEX_NY_PROCS;

   if (params[INDEX_NX_GLOBAL]   != nxProcs*loc.nx) status = -INDEX_NX_GLOBAL;
   if (params[INDEX_NY_GLOBAL]   != nyProcs*loc.ny) status = -INDEX_NY_GLOBAL;

   if (params[INDEX_KX0]         != 0)              status = -INDEX_KX0;
   if (params[INDEX_KY0]         != 0)              status = -INDEX_KY0;

   // INDEX_NB is no longer used

   if (params[INDEX_NBANDS]      != loc.nf)     status = -INDEX_NBANDS;;

   if (status != 0) {
      fprintf(stderr, "ERROR in test_read_header, err==%d\n", status);
   }

   delete initObj;

   return status;
}
