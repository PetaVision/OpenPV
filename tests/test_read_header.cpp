/*
 * test_read_header.cpp
 *
 *  Created on: Dec 30, 2009
 *      Author: Craig Rasmussen
 */

#include "columns/HyPerCol.hpp"
#include "columns/Communicator.hpp"
#include "io/fileio.hpp"

#include <stdio.h>
#include <stdlib.h>

#undef DEBUG_OUTPUT

#ifdef PV_USE_MPI
#  include <mpi.h>
#else
#  include "../include/mpi_stubs.h"
#endif

// function declarations that are not public in fileio.hpp
//
namespace PV {
   size_t pv_sizeof(int datatype);
}

const char filename[]  = "output/test_read_header.pvp";

using namespace PV;

int main(int argc, char* argv[])
{
   int status = 0;
   FILE * fp = NULL;
   double time, write_time = 33.3;

   int params[NUM_BIN_PARAMS];
   int filetype, write_filetype = PVP_FILE_TYPE;
   int datatype, write_datatype = PV_FLOAT_TYPE;
   int numParams = NUM_BIN_PARAMS, write_num_params = NUM_BIN_PARAMS;

   const int subRecordSize = pv_sizeof(write_datatype);

   const bool extended = false;
   const bool contiguous = false;

   // create the managing hypercolumn
   //
   HyPerCol * hc = new HyPerCol("column", argc, argv);

   PVLayerLoc loc = hc->getImageLoc();

   Communicator * comm = hc->icCommunicator();
   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();

   // NULL return value normal for rank > 0
   fp = pvp_open_write_file(filename, comm, false);

   status = pvp_write_header(fp, comm, write_time, &loc,
                             write_filetype, write_datatype, subRecordSize,
                             extended, contiguous, write_num_params);

   PV::pvp_close_file(fp, comm);
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
   const int localSize  = loc.nx * loc.ny * loc.nBands * subRecordSize;

   if (params[INDEX_HEADER_SIZE] != numParams*size_float) {
      status = -INDEX_HEADER_SIZE;
   }
   if (params[INDEX_NUM_PARAMS]  != numParams)      status = -INDEX_NUM_PARAMS;
   if (params[INDEX_FILE_TYPE]   != filetype)       status = -INDEX_FILE_TYPE;
   if (params[INDEX_NX]          != loc.nx)         status = -INDEX_NX;
   if (params[INDEX_NY]          != loc.ny)         status = -INDEX_NY;
   if (params[INDEX_NF]          != loc.nBands)     status = -INDEX_NF;
   if (params[INDEX_NUM_RECORDS] != numRecords)     status = -INDEX_NUM_RECORDS;
   if (params[INDEX_RECORD_SIZE] != localSize)      status = -INDEX_RECORD_SIZE;

   if (params[INDEX_DATA_SIZE]   != size_float)     status = -INDEX_DATA_SIZE;

   if (params[INDEX_NX_PROCS]    != nxProcs)        status = -INDEX_NX_PROCS;
   if (params[INDEX_NY_PROCS]    != nyProcs)        status = -INDEX_NY_PROCS;

   if (params[INDEX_NX_GLOBAL]   != nxProcs*loc.nx) status = -INDEX_NX_GLOBAL;
   if (params[INDEX_NY_GLOBAL]   != nyProcs*loc.ny) status = -INDEX_NY_GLOBAL;

   if (params[INDEX_KX0]         != 0)              status = -INDEX_KX0;
   if (params[INDEX_KY0]         != 0)              status = -INDEX_KY0;

   if (params[INDEX_NPAD]        != loc.nPad)       status = -INDEX_NPAD;
   if (params[INDEX_NBANDS]      != loc.nBands)     status = -INDEX_NBANDS;;

   if (status != 0) {
      fprintf(stderr, "ERROR in test_read_header, err==%d\n", status);
   }

   delete hc;
   return status;
}
