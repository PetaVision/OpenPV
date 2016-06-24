/*
 * Image.hpp
 *
 *  Created on: Sep 8, 2009
 *      Author: rasmussn
 */


#ifndef IMAGE_HPP_
#define IMAGE_HPP_

#include "BaseInput.hpp"
#include <cMakeHeader.h>

namespace PV {

class Image : public BaseInput{

#ifdef PV_USE_GDAL

protected:
   /** 
    * List of parameters needed from the Image class
    * @name Image Parameters
    * @{
    */

   /**
    * @brief writeStep: The Image class changes the default of writeStep to -1 (i.e. never write to the output pvp file).
    */
   virtual void ioParam_writeStep(enum ParamsIOFlag ioFlag);

   /** @} */
   
protected:
   Image();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

public:
   Image(const char * name, HyPerCol * hc);
   virtual ~Image();
   virtual int communicateInitInfo();

   // primary layer interface
   //
   virtual double getDeltaUpdateTime();
   virtual int updateState(double time, double dt);

private:
   int initialize_base();

protected:
   virtual int readImageFileGDAL(char const * filename, PVLayerLoc const * loc);
#ifdef INACTIVE // Commented out April 19, 2016.  Might prove useful to restore the option to resize using GDAL.
   virtual int scatterImageFileGDAL(const char * filename, int xOffset, int yOffset, PV::Communicator * comm, const PVLayerLoc * loc, float * buf, bool autoResizeFlag);
#endif // INACTIVE // Commented out April 19, 2016.  Might prove useful to restore the option to resize using GDAL.

   virtual int retrieveData(double timef, double dt, int batchIndex);

   //Virtual function to define how readImage specifies batches
   virtual int readImage(const char * filename);
   int calcColorType(int numBands, GDALColorInterp * colorbandtypes);

#else // PV_USE_GDAL
public:
   Image(char const * name, HyPerCol * hc);
protected:
   Image();
   int retrieveData(double timef, double dt);
#endif // PV_USE_GDAL

}; // class Image

BaseObject * createImage(char const * name, HyPerCol * hc);

}  // namespace PV

#endif /* IMAGE_HPP_ */
