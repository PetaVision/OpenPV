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
#include <gdal.h>

int getFileType(const char * filename);
int getImageInfo(    const char * filename, PV::Communicator * comm, PVLayerLoc * loc, GDALColorInterp ** colorbandtypes);
int getImageInfoPVP( const char * filename, PV::Communicator * comm, PVLayerLoc * loc, GDALColorInterp ** colorbandtypes);
int getImageInfoGDAL(const char * filename, PV::Communicator * comm, PVLayerLoc * loc, GDALColorInterp ** colorbandtypes);

int gatherImageFile(    const char * filename,
                        PV::Communicator * comm, const PVLayerLoc * loc, unsigned char * buf);
int gatherImageFile(    const char * filename,
                        PV::Communicator * comm, const PVLayerLoc * loc, pvdata_t * buf);
int gatherImageFilePVP( const char * filename,
                        PV::Communicator * comm, const PVLayerLoc * loc, unsigned char * buf);
int gatherImageFileGDAL(const char * filename,
                        PV::Communicator * comm, const PVLayerLoc * loc, unsigned char * buf);

int scatterImageFile(    const char * filename, int xOffset, int yOffset,
                         PV::Communicator * comm, const PVLayerLoc * loc, float * buf);
int scatterImageFilePVP( const char * filename, int xOffset, int yOffset,
                         PV::Communicator * comm, const PVLayerLoc * loc, float * buf);
#ifdef OBSOLETE // Marked obsolete Dec 10, 2012, during reworking of PVP files to be MPI-independent.
int windowFromPVPBuffer(int startx, int starty, int nx, int ny, int * params, float * destbuf, char * pvpbuffer, const char * filename);
#endif // OBSOLETE
int scatterImageFileGDAL(const char * filename, int xOffset, int yOffset,
                         PV::Communicator * comm, const PVLayerLoc * loc, float * buf);

#ifdef OBSOLETE // Marked obsolete Dec 10, 2012.  No one calls either gather or writeWithBorders and they have TODO's that indicate they're broken.
int gather (PV::Communicator * comm, const PVLayerLoc * loc,
            unsigned char * dstBuf, unsigned char * srcBuf);

int writeWithBorders(const char * filename, PVLayerLoc * loc, float * buf);
#endif // OBSOLETE

#endif /* IMAGEIO_HPP_ */
