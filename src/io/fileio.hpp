/*
 * fileio.hpp
 *
 *  Created on: Oct 21, 2009
 *      Author: rasmussn
 */

#ifndef FILEIO_HPP_
#define FILEIO_HPP_

#include "io.h"
#include <mpi.h>
#include "../include/PVLayerLoc.h"
#include "../columns/Communicator.hpp"

namespace PV {

FILE * pvp_open_write_file(const char * filename, Communicator * comm, bool append);

int pvp_close_file(FILE * fp, Communicator * comm);

int pvp_read_header(const char * filename, Communicator * comm, double * time,
                    int * filetype, int * datatype, int params[], int * numParams);
int pvp_write_header(FILE * fp, Communicator * comm, double time, const PVLayerLoc * loc,
                     int filetype, int datatype, int subRecordSize,
                     bool extended, bool contiguous, unsigned int numParams, size_t localSize);

int read(const char * filename, Communicator * comm, double * time, void * data,
         const PVLayerLoc * loc, int datatype, bool extended, bool contiguous);

int write(const char * filename, Communicator * comm, double time, const pvdata_t * data,
          const PVLayerLoc * loc, int datatype, bool extended, bool contiguous);

int write(FILE * fp, Communicator * comm, double time, const pvdata_t * data,
          const PVLayerLoc * loc, int datatype, bool extended, bool contiguous);

int writeActivity(FILE * fp, Communicator * comm, double time, PVLayer * l);

int writeActivitySparse(FILE * fp, Communicator * comm, double time, PVLayer * l);

int readWeights(PVPatch ** patches, int numPatches, const char * filename,
                Communicator * comm, double * time, const PVLayerLoc * loc, bool extended);

int writeWeights(const char * filename, Communicator * comm, double time, bool append,
                 const PVLayerLoc * loc, int nxp, int nyp, int nfp, float minVal, float maxVal,
                 PVPatch ** patches, int numPatches, bool compressed=true);

int pvp_check_file_header(Communicator * comm, const PVLayerLoc * loc, int params[], int numParams);

} // namespace PV

#endif /* FILEIO_HPP_ */
