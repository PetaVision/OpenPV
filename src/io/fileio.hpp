/*
 * fileio.hpp
 *
 *  Created on: Oct 21, 2009
 *      Author: rasmussn
 */

#ifndef FILEIO_HPP_
#define FILEIO_HPP_

#include "io.h"
#ifdef PV_USE_MPI
#  include <mpi.h>
#else
#  include "../include/mpi_stubs.h"
#endif // PV_USE_MPI
#include "../include/PVLayerLoc.h"
#include "../columns/Communicator.hpp"

namespace PV {

size_t pv_sizeof(int datatype);

FILE * pvp_open_read_file(const char * filename, Communicator * comm);

FILE * pvp_open_write_file(const char * filename, Communicator * comm, bool append);

int pvp_close_file(FILE * fp, Communicator * comm);

int pvp_read_header(const char * filename, Communicator * comm, double * time,
                    int * filetype, int * datatype, int params[], int * numParams);
int pvp_write_header(FILE * fp, Communicator * comm, double time, const PVLayerLoc * loc,
                     int filetype, int datatype, int numbands,
                     bool extended, bool contiguous, unsigned int numParams, size_t localSize);

int readNonspikingActFile(const char * filename, Communicator * comm, double * time, void * data,
         int level, const PVLayerLoc * loc, int datatype, bool extended, bool contiguous);
int read_pvdata(const char * filename, Communicator * comm, double * time, void * data,
         const PVLayerLoc * loc, int datatype, bool extended, bool contiguous);

int write_pvdata(const char * filename, Communicator * comm, double time, const pvdata_t * data,
          const PVLayerLoc * loc, int datatype, bool extended, bool contiguous);

int write_pvdata(FILE * fp, Communicator * comm, double time, const pvdata_t * data,
          const PVLayerLoc * loc, int datatype, bool extended, bool contiguous, int tag);

int writeActivity(FILE * fp, Communicator * comm, double time, PVLayer * l);

int writeActivitySparse(FILE * fp, Communicator * comm, double time, PVLayer * l);

int readWeights(PVPatch *** patches, pvdata_t ** dataStart, int numArbors, int numPatches, const char * filename,
                Communicator * comm, double * timed, const PVLayerLoc * loc);

int writeWeights(const char * filename, Communicator * comm, double timed, bool append,
                 const PVLayerLoc * loc, int nxp, int nyp, int nfp, float minVal, float maxVal,
                 PVPatch *** patches, pvdata_t ** dataStart, int numPatches, int numArbors, bool compress=true, int file_type=PVP_WGT_FILE_TYPE);

int writeWeights(const char * filename, Communicator * comm, double time, bool append,
                 const PVLayerLoc * loc, int nxp, int nyp, int nfp, float minVal, float maxVal,
                 PVPatch ** patches, int numPatches, bool compressed=true, int fileType=PVP_WGT_FILE_TYPE);

int pvp_check_file_header(Communicator * comm, const PVLayerLoc * loc, int params[], int numParams);

} // namespace PV

#endif /* FILEIO_HPP_ */
