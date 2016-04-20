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

   //TODO this functionality should be in both pvp and image. Set here for now, as pvp does not support imageBC
   /**
    * @brief autoResizeFlag: If set to true, the image will be resized to the layer
    * @details 
    *  For the auto resize flag, PV checks which side (x or y) is the shortest, relative to the
    *  hypercolumn size specified.  Then it determines the largest chunk it can possibly take
    *  from the image with the correct aspect ratio determined by hypercolumn.  It then
    *  determines the offset needed in the long dimension to center the cropped image,
    *  and reads in that portion of the image.  The offset can optionally be translated by
    *  offset{X,Y} specified in the params file (values can be positive or negative).
    *  If the specified offset takes the cropped image outside the image file, it uses the
    *  largest-magnitude offset that stays within the image file's borders.
    */
   virtual void ioParam_autoResizeFlag(enum ParamsIOFlag ioFlag);

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
   //virtual int allocateDataStructures();

   // primary layer interface
   //
   virtual double getDeltaUpdateTime();
   //virtual bool needUpdate(double time, double dt);
   virtual int updateState(double time, double dt);
   //virtual int outputState(double time, bool last=false);

   //const char * getFilename() { return imageFilename; }

private:
   int initialize_base();

protected:
   virtual int scatterImageFileGDAL(const char * filename, int xOffset, int yOffset, PV::Communicator * comm, const PVLayerLoc * loc, float * buf, bool autoResizeFlag);
   virtual int retrieveData(double timef, double dt);

   //Virtual function to define how readImage specifies batches
   //virtual int readImage(const char * filename, int offsetX, int offsetY, const char* anchor);

   //virtual int readImage(const char * filename, int batchIdx);
   //virtual int readImage(const char * filename, int batchIdx, int offsetX, int offsetY);
   virtual int readImage(const char * filename, int batchIdx, int offsetX, int offsetY, const char* anchor);

   //char * imageFilename;       // path to file if a file exists
   static float * convertToGrayScale(float * buf, int nx, int ny, int numBands, GDALColorInterp * colorbandtypes);
   static float* copyGrayScaletoMultiBands(float * buf, int nx, int ny, int numBands, GDALColorInterp * colorbandtypes);
   static inline int calcBandWeights(int numBands, float * bandweights, GDALColorInterp * colorbandtypes);
   static inline void equalBandWeights(int numBands, float * bandweights);


   bool autoResizeFlag; // if true, PetaVision will automatically resize your images to the size specified by hypercolumn
#else // PV_USE_GDAL
public:
   Image(char const * name, HyPerCol * hc);
protected:
   Image();
#endif // PV_USE_GDAL

}; // class Image

BaseObject * createImage(char const * name, HyPerCol * hc);

}  // namespace PV

#endif /* IMAGE_HPP_ */
