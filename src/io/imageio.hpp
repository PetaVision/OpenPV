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
#ifdef PV_USE_GDAL
#  include <gdal.h>
#  include <gdal_priv.h>
#  include <ogr_spatialref.h>
#else
#  define GDAL_CONFIG_ERR_STR "PetaVision must be compiled with GDAL to use this file type\n"
#endif // PV_USE_GDAL

int getFileType(const char * filename);
#ifdef OBSOLETE // Marked obsolete Jan 29, 2016.  getImageInfo was commented out in the .cpp file some time ago.
int getImageInfo(    const char * filename, PV::Communicator * comm, PVLayerLoc * loc, GDALColorInterp ** colorbandtypes);
#endif // OBSOLETE // Marked obsolete Jan 29, 2016.  getImageInfo was commented out in the .cpp file some time ago.
int getImageInfoPVP( const char * filename, PV::Communicator * comm, PVLayerLoc * loc);

int gatherImageFile(    const char * filename,
                        PV::Communicator * comm, const PVLayerLoc * loc, unsigned char * buf, bool verifyWrites);
int gatherImageFile(    const char * filename,
                        PV::Communicator * comm, const PVLayerLoc * loc, pvdata_t * buf, bool verifyWrites);
int gatherImageFilePVP( const char * filename,
                        PV::Communicator * comm, const PVLayerLoc * loc, unsigned char * buf, bool verifyWrites);
#ifdef PV_USE_GDAL
int getImageInfoGDAL(const char * filename, PV::Communicator * comm, PVLayerLoc * loc, GDALColorInterp ** colorbandtypes);
int gatherImageFileGDAL(const char * filename,
                       PV::Communicator * comm, const PVLayerLoc * loc, unsigned char * buf, bool verifyWrites);
GDALDataset * PV_GDALOpen(const char * filename);
#endif // PV_USE_GDAL

#endif /* IMAGEIO_HPP_ */
