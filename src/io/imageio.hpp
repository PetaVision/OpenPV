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

int getFileType(const char * filename);
int getImageInfo(    const char * filename, PV::Communicator * comm, PVLayerLoc * loc);
int getImageInfoPVP( const char * filename, PV::Communicator * comm, PVLayerLoc * loc);
int getImageInfoGDAL(const char * filename, PV::Communicator * comm, PVLayerLoc * loc);

int gatherImageFile(    const char * filename,
                        PV::Communicator * comm, const PVLayerLoc * loc, unsigned char * buf);
int gatherImageFile(    const char * filename,
                        PV::Communicator * comm, const PVLayerLoc * loc, pvdata_t * buf);
int gatherImageFilePVP( const char * filename,
                        PV::Communicator * comm, const PVLayerLoc * loc, unsigned char * buf);
int gatherImageFileGDAL(const char * filename,
                        PV::Communicator * comm, const PVLayerLoc * loc, unsigned char * buf);

int scatterImageFile(    const char * filename, int xOffset, int yOffset,
                         PV::Communicator * comm, PVLayerLoc * loc, unsigned char * buf);
int scatterImageFilePVP( const char * filename,
                         PV::Communicator * comm, PVLayerLoc * loc, unsigned char * buf);
int scatterImageFileGDAL(const char * filename, int xOffset, int yOffset,
                         PV::Communicator * comm, PVLayerLoc * loc, unsigned char * buf);

int scatter(PV::Communicator * comm, PVLayerLoc * loc, unsigned char * buf);
int gather (PV::Communicator * comm, const PVLayerLoc * loc,
            unsigned char * dstBuf, unsigned char * srcBuf);

int scatter(PV::Communicator * comm, PVLayerLoc * loc, float * buf);
int gather (PV::Communicator * comm, const PVLayerLoc * loc, float * buf);

int writeWithBorders(const char * filename, PVLayerLoc * loc, float * buf);

#endif /* IMAGEIO_HPP_ */
