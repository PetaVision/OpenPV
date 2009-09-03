/*
 * imageio.hpp
 *
 *  Created on: Aug 25, 2009
 *      Author: rasmussn
 */

#ifndef IMAGEIO_HPP_
#define IMAGEIO_HPP_

#include "Communicator.hpp"
#include "../pann_types.h"

int getImageInfo(const char* filename, PV::Communicator * comm, LayerLoc * loc);
int scatterImageFile(const char* filename,
                     PV::Communicator * comm, LayerLoc * loc, float * buf);
int gatherImageFile( const char* filename,
                     PV::Communicator * comm, LayerLoc * loc, float * buf);

int scatter(PV::Communicator * comm, LayerLoc * loc, float * buf);
int gather (PV::Communicator * comm, LayerLoc * loc, float * buf);

int writeWithBorders(const char * filename, LayerLoc * loc, float * buf);

#endif /* IMAGEIO_HPP_ */
