/*
 * fileio.hpp
 *
 *  Created on: Oct 21, 2009
 *      Author: rasmussn
 */

#ifndef FILEIO_HPP_
#define FILEIO_HPP_

#include "io.h"
#include "../include/LayerLoc.h"
#include "../columns/Communicator.hpp"

namespace PV {

FILE * pvp_open_write_file(const char * filename, bool append);

int pvp_write_header(FILE * fp, Communicator * comm, double time, const LayerLoc * loc,
                     int filetype, int datatype, int subRecordSize,
                     bool extended, bool contiguous, unsigned int numParams);

int read(const char * filename, Communicator * comm, double * time, pvdata_t * data,
         const LayerLoc * loc, int datatype, bool extended, bool contiguous);

int write(const char * filename, Communicator * comm, double time, pvdata_t * data,
          const LayerLoc * loc, int datatype, bool extended, bool contiguous);

int writeActivitySparse(FILE * fp, Communicator * comm, double time, PVLayer * l);

int readWeights(PVPatch ** patches, int numPatches, const char * filename,
                Communicator * comm, double * time, const LayerLoc * loc, bool extended);

int writeWeights(const char * filename, Communicator * comm, double time, bool append,
                 const LayerLoc * loc, int nxp, int nyp, int nfp, float minVal, float maxVal,
                 PVPatch ** patches, int numPatches);

} // namespace PV

#endif /* FILEIO_HPP_ */
