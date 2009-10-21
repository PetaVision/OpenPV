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

int read(const char * filename, Communicator * comm, double * time, pvdata_t * data,
         const LayerLoc * loc, int datatype, bool extended, bool contiguous);

int write(const char * filename, Communicator * comm, double time, pvdata_t * data,
          const LayerLoc * loc, int datatype, bool extended, bool contiguous);

} // namespace PV

#endif /* FILEIO_HPP_ */
