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

int getImageInfo    (const char * filename, PV::Communicator * comm, LayerLoc * loc);
int getImageInfoPVP (const char * filename, PV::Communicator * comm, LayerLoc * loc);
int getImageInfoGDAL(const char * filename, PV::Communicator * comm, LayerLoc * loc);

int gatherImageFile (const char * filename,
                     PV::Communicator * comm, LayerLoc * loc, unsigned char * buf);
int scatterImageFile(const char * filename,
                     PV::Communicator * comm, LayerLoc * loc, unsigned char * buf);

int gatherParByteFile (const char * filename,
                       PV::Communicator * comm, LayerLoc * loc, unsigned char * buf);
int scatterParByteFile(const char * filename,
                       PV::Communicator * comm, LayerLoc * loc, unsigned char * buf);

int scatter(PV::Communicator * comm, LayerLoc * loc, float * buf);
int gather (PV::Communicator * comm, LayerLoc * loc, float * buf);

int writeWithBorders(const char * filename, LayerLoc * loc, float * buf);

#endif /* IMAGEIO_HPP_ */
