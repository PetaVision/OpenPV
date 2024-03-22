/*
 * fileio.hpp
 *
 *  Created on: Oct 21, 2009
 *      Author: rasmussn
 */

#ifndef FILEIO_HPP_
#define FILEIO_HPP_

#include "structures/MPIBlock.hpp"
#include <memory>
#include <sys/stat.h>

namespace PV {

int PV_stat(const char *path, struct stat *buf);
void ensureDirExists(MPIBlock const *mpiBlock, char const *dirname);
void ensureDirExists(std::shared_ptr<MPIBlock const> mpiBlock, char const *dirname);

} // namespace PV

#endif /* FILEIO_HPP_ */
