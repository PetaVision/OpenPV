/*
 * fileio.hpp
 *
 *  Created on: Oct 21, 2009
 *      Author: rasmussn
 */

#ifndef FILEIO_HPP_
#define FILEIO_HPP_

#include "io.hpp"
#include "arch/mpi/mpi.h"
#include "include/pv_types.h"
#include "include/PVLayerLoc.h"
#include "columns/Communicator.hpp"
#include "columns/DataStore.hpp"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace PV {

// index/value pairs used by writeActivitySparseNonspiking()
typedef struct indexvaluepair_ {
    unsigned int index;
    pvdata_t value;
} indexvaluepair;

void timeToParams(double time, void * params);
double timeFromParams(void * params);

size_t pv_sizeof(int datatype);

PV_Stream * PV_fopen(const char * path, const char * mode, bool verifyWrites);
int PV_stat(const char * path, struct stat * buf);
long int getPV_StreamFilepos(PV_Stream * pvstream);
long int updatePV_StreamFilepos(PV_Stream * pvstream);
long int PV_ftell(PV_Stream * pvstream);
int PV_fseek(PV_Stream * pvstream, long int offset, int whence);
size_t PV_fwrite(const void * RESTRICT ptr, size_t size, size_t nitems, PV_Stream * RESTRICT pvstream);
size_t PV_fread(void * RESTRICT ptr, size_t size, size_t nitems, PV_Stream * RESTRICT pvstream);
int PV_fclose(PV_Stream * pvstream);
PV_Stream * PV_stdout();

PV_Stream * pvp_open_read_file(const char * filename, Communicator * comm);

PV_Stream * pvp_open_write_file(const char * filename, Communicator * comm, bool append);

int pvp_close_file(PV_Stream * pvstream, Communicator * comm);

int pvp_read_header(PV_Stream * pvstream, Communicator * comm, int * params, int * numParams);
int pvp_read_header(const char * filename, Communicator * comm, double * time,
                    int * filetype, int * datatype, int params[], int * numParams);
void read_header_err(const char * filename, Communicator * comm, int returned_num_params, int * params);
int pvp_write_header(PV_Stream * pvstream, Communicator * comm, int * params, int numParams);

// The pvp_write_header below will go away in favor of the pvp_write_header above.
int pvp_write_header(PV_Stream * pvstream, Communicator * comm, double time, const PVLayerLoc * loc,
                     int filetype, int datatype, int numbands,
                     bool extended, bool contiguous, unsigned int numParams, size_t recordSize);

int * pvp_set_file_params(Communicator * comm, double timed, const PVLayerLoc * loc, int datatype, int numbands);
int * pvp_set_activity_params(Communicator * comm, double timed, const PVLayerLoc * loc, int datatype, int numbands);
int * pvp_set_weight_params(Communicator * comm, double timed, const PVLayerLoc * loc, int datatype, int numbands, int nxp, int nyp, int nfp, float min, float max, int numPatches);
int * pvp_set_nonspiking_act_params(Communicator * comm, double timed, const PVLayerLoc * loc, int datatype, int numbands);
int * pvp_set_kernel_params(Communicator * comm, double timed, const PVLayerLoc * loc, int datatype, int numbands, int nxp, int nyp, int nfp, float min, float max, int numPatches);
int * alloc_params(int numParams);
int set_weight_params(int * params, int nxp, int nyp, int nfp, float min, float max, int numPatches);

int pvp_read_time(PV_Stream * pvstream, Communicator * comm, int root_process, double * timed);

int writeActivity(PV_Stream * pvstream, Communicator * comm, double timed, DataStore * store, const PVLayerLoc* loc);

int writeActivitySparse(PV_Stream * pvstream, Communicator * comm, double timed, DataStore * store, const PVLayerLoc* loc, bool includeValues);

//This function is not defined anywhere?
//int writeActivitySparseValues(PV_Stream * pvstream, PV_Stream * posstream, Communicator * comm, double time, PVLayer * l);

int readWeights(PVPatch *** patches, pvwdata_t ** dataStart, int numArbors, int numPatches, int nxp, int nyp, int nfp, const char * filename,
                Communicator * comm, double * timed, const PVLayerLoc * loc);
// The old readWeights, now readWeightsDeprecated, was deprecated Nov 20, 2014.
// readWeights() reads weights that were saved in an MPI-independent manner (the current writeWeights)
// readWeightsDeprecated() reads weights saved in the old MPI-dependent manner.
int readWeightsDeprecated(PVPatch *** patches, pvwdata_t ** dataStart, int numArbors, int numPatches, int nxp, int nyp, int nfp, const char * filename,
                Communicator * comm, double * timed, const PVLayerLoc * loc);

int writeWeights(const char * filename, Communicator * comm, double timed, bool append,
                 const PVLayerLoc * preLoc, const PVLayerLoc * postLoc, int nxp, int nyp, int nfp, float minVal, float maxVal,
                 PVPatch *** patches, pvwdata_t ** dataStart, int numPatches, int numArbors, bool compress=true, int file_type=PVP_WGT_FILE_TYPE);

int pvp_check_file_header(Communicator * comm, const PVLayerLoc * loc, int params[], int numParams);

int writeRandState(const char * filename, Communicator * comm, taus_uint4 * randState, const PVLayerLoc * loc, bool isExtended, bool verifyWrites);

int readRandState(const char * filename, Communicator * comm, taus_uint4 * randState, const PVLayerLoc * loc, bool isExtended);

template <typename T> int gatherActivity(PV_Stream * pvstream, Communicator * comm, int rootproc, T * buffer, const PVLayerLoc * layerLoc, bool extended);
template <typename T> int scatterActivity(PV_Stream * pvstream, Communicator * comm, int rootproc, T * buffer, const PVLayerLoc * layerLoc, bool extended, const PVLayerLoc * fileLoc=NULL, int offsetX=0, int offsetY=0, int filetype=PVP_NONSPIKING_ACT_FILE_TYPE, int numActive=0);
} // namespace PV

#endif /* FILEIO_HPP_ */
