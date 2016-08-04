/*
 * Image.cpp
 *
 *  Created on: Sep 8, 2009
 *      Author: rasmussn
 */

#include "Image.hpp"

#include "utils/PVImg.hpp"
#include "../arch/mpi/mpi.h"
#include <assert.h>
#include <string.h>
#include <iostream>

namespace PV {

   Image::Image() {
      initialize_base();
   }

   Image::Image(const char * name, HyPerCol * hc) {
      initialize_base();
      initialize(name, hc);
   }

   Image::~Image() {
   }

   int Image::initialize_base() {
         return PV_SUCCESS;
   }

   int Image::initialize(const char * name, HyPerCol * hc) {
      int status = BaseInput::initialize(name, hc);
      return status;
   }

   int Image::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
      int status = BaseInput::ioParamsFillGroup(ioFlag);
      return status;
   }

   //Image readImage reads the same thing to every batch
   //This call is here since this is the entry point called from allocate
   //Movie overwrites this function to define how it wants to load into batches
   int Image::retrieveData(double timef, double dt, int batchIndex)
   {
      readImage(mInputPath);
      mInputData.resize(mImage->getHeight(), mImage->getWidth(), getLayerLoc()->nf);
      mInputData.set(mImage->serialize(getLayerLoc()->nf));
      return PV_SUCCESS;
   }

   double Image::getDeltaUpdateTime(){
         return -1; //Never update
   }

   int Image::communicateInitInfo() {
      int status = BaseInput::communicateInitInfo();
      int fileType = getFileType(mInputPath.c_str());
      if(fileType == PVP_FILE_TYPE){
         pvError() << "Image/Movie no longer reads PVP files. Use ImagePvp/MoviePvp layer instead.\n";
      }
      return status;
   }

   /**
    * update the image buffers
    */
   int Image::updateState(double time, double dt)
   {
      return 0;
   }

   void Image::ioParam_writeStep(enum ParamsIOFlag ioFlag) {
      //Default to -1 in Image
      parent->ioParamValue(ioFlag, name, "writeStep", &writeStep, -1.0);
   }

   void Image::readImage(std::string filename)
   {
      assert(parent->columnId()==0); // readImage is called by retrieveData, which only the root process calls.  BaseInput::scatterInput does the scattering.
      const PVLayerLoc *loc = getLayerLoc();
      bool usingTempFile = false;

      // Attempt to download our input file if we've been passed a URL or AWS path
      if(filename.find("://") != std::string::npos) {
         usingTempFile = true;
         std::string extension = filename.substr(filename.find_last_of("."));
         std::string pathstring = parent->getOutputPath() + std::string("/temp.XXXXXX") + extension;
         char tempStr[256];
         strcpy(tempStr, pathstring.c_str());
         int tempFileID = mkstemps(tempStr, extension.size());
         pathstring = std::string(tempStr);
         if(tempFileID < 0) {
            pvError().printf("Cannot create temp image file.\n");
         }
         std::string systemstring;
         if (filename.find("s3://") != std::string::npos) {
            systemstring = std::string("aws s3 cp \'") + filename + std::string("\' ") + pathstring;
         }
         else { // URLs other than s3://
            systemstring = std::string("wget -O ") + pathstring + std::string(" \'") + filename + std::string("\'");
         }
         filename = pathstring;
         const int numAttempts = 5;
         for(int attemptNum = 0; attemptNum < numAttempts; attemptNum++) {
            int status = system(systemstring.c_str());
            if(status != 0) {
               if(attemptNum == numAttempts - 1) {
                  pvError().printf("download command \"%s\" failed: %s.  Exiting\n", systemstring.c_str(), strerror(errno));
               }
               sleep(1);
            }
            else {
               break;
            }
         }
      }

      mImage = std::unique_ptr<PVImg>(new PVImg(std::string(filename)));

      if (usingTempFile) {
         int rmstatus = remove(filename.c_str());
         if(rmstatus) {
            pvError().printf("remove(\"%s\") failed.  Exiting.\n", filename.c_str());
         }
      }
   }

   int Image::postProcess(double timef, double dt) {

      //TODO: Grayscale conversion stuff here
      return BaseInput::postProcess(timef, dt);
   }

/*
   int Image::convertToGrayScale(float ** buffer, int nx, int ny, int numBands, InputColorType colorType)
   {
      // even though the numBands argument goes last, the routine assumes that
      // the organization of buf is, bands vary fastest, then x, then y.
      if (numBands < 2) {return PV_SUCCESS;}

      const int sxcolor = numBands;
      const int sycolor = numBands*nx;
      const int sb = 1;

      const int sxgray = 1;
      const int sygray = nx;

      float * graybuf = new float[nx*ny];
      float * colorbuf = *buffer;

      float bandweight[numBands];
      calcBandWeights(numBands, bandweight, colorType);

      for (int j = 0; j < ny; j++) {
         for (int i = 0; i < nx; i++) {
            float val = 0;
            for (int b = 0; b < numBands; b++) {
               float d = colorbuf[i*sxcolor + j*sycolor + b*sb];
               val += d*bandweight[b];
            }
            graybuf[i*sxgray + j*sygray] = val;
         }
      }
      delete[] *buffer;
      *buffer = graybuf;
      return PV_SUCCESS;
   }

   int Image::convertGrayScaleToMultiBand(float ** buffer, int nx, int ny, int numBands)
   {
      const int sxcolor = numBands;
      const int sycolor = numBands*nx;
      const int sb = 1;

      const int sxgray = 1;
      const int sygray = nx;

      float * multiBandsBuf = new float[nx*ny*numBands];
      float * graybuf = *buffer;

      for (int j = 0; j < ny; j++)
      {
         for (int i = 0; i < nx; i++)
         {
            for (int b = 0; b < numBands; b++)
            {
               multiBandsBuf[i*sxcolor + j*sycolor + b*sb] = graybuf[i*sxgray + j*sygray];
            }

         }
      }
      delete[] *buffer;
      *buffer = multiBandsBuf;
      return PV_SUCCESS;
   }

   int Image::calcBandWeights(int numBands, float * bandweight, InputColorType colorType) {
      const float grayalphaweights[2] = {1.0, 0.0};
      const float rgbaweights[4] = {0.30f, 0.59f, 0.11f, 0.0f}; // RGB weights from <https://en.wikipedia.org/wiki/Grayscale>, citing Pratt, Digital Image Processing
      switch (colorType) {
      case COLORTYPE_UNRECOGNIZED:
         equalBandWeights(numBands, bandweight);
         break;
      case COLORTYPE_GRAYSCALE:
         if (numBands==1 || numBands==2) {
            memcpy(bandweight, grayalphaweights, numBands*sizeof(*bandweight));
         }
         else {
            pvAssert(0);
         }
         break;
      case COLORTYPE_RGB:
         if (numBands==3 || numBands==4) {
            memcpy(bandweight, rgbaweights, numBands*sizeof(*bandweight));
         }
         else {
            pvAssert(0);
         }
         break;
      default:
         pvAssert(0);
      }
      return PV_SUCCESS;
   }
*/

   BaseObject * createImage(char const * name, HyPerCol * hc) {
      return hc ? new Image(name, hc) : NULL;
   }

} // namespace PV
