/*
 * fileio.hpp
 *
 *  Created on: Oct 21, 2009
 *      Author: rasmussn
 */

#ifndef FILEIO_HPP_
#define FILEIO_HPP_

#include "arch/mpi/mpi.h"
#include "components/Patch.hpp"
#include "include/PVLayerLoc.h"
#include "include/pv_types.h"
#include "structures/MPIBlock.hpp"

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

namespace PV {

PV_Stream *PV_fopen(const char *path, const char *mode, bool verifyWrites);
int PV_stat(const char *path, struct stat *buf);
long int PV_ftell(PV_Stream *pvstream);
int PV_fseek(PV_Stream *pvstream, long int offset, int whence);
size_t
PV_fwrite(const void *RESTRICT ptr, size_t size, size_t nitems, PV_Stream *RESTRICT pvstream);
size_t PV_fread(void *RESTRICT ptr, size_t size, size_t nitems, PV_Stream *RESTRICT pvstream);
int PV_fclose(PV_Stream *pvstream);
void ensureDirExists(MPIBlock const *mpiBlock, char const *dirname);

} // namespace PV

#endif /* FILEIO_HPP_ */
