/*
 * imageio.hpp
 *
 *  Created on: Aug 25, 2009
 *      Author: rasmussn
 */

#ifndef IMAGEIO_HPP_
#define IMAGEIO_HPP_

#include "../columns/Communicator.hpp"
#include "../include/pv_types.h"
#include "fileio.hpp"

int getFileType(const char * filename);

int getImageInfoPVP( const char * filename, PV::Communicator * comm, PVLayerLoc * loc);

int gatherImageFile(    const char * filename,
                        PV::Communicator * comm, const PVLayerLoc * loc, unsigned char * buf, bool verifyWrites);
int gatherImageFile(    const char * filename,
                        PV::Communicator * comm, const PVLayerLoc * loc, pvdata_t * buf, bool verifyWrites);
int gatherImageFilePVP( const char * filename,
                        PV::Communicator * comm, const PVLayerLoc * loc, unsigned char * buf, bool verifyWrites);

#endif /* IMAGEIO_HPP_ */
