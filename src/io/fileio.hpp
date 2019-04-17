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

int PV_stat(const char *path, struct stat *buf);
void ensureDirExists(MPIBlock const *mpiBlock, char const *dirname);

} // namespace PV

#endif /* FILEIO_HPP_ */
