#include "ImageLayer.hpp"
#include "utils/Image.hpp"
#include "../arch/mpi/mpi.h"

#include <assert.h>
#include <string.h>
#include <iostream>

namespace PV {

   ImageLayer::ImageLayer() {
      initialize_base();
   }

   ImageLayer::ImageLayer(const char * name, HyPerCol * hc) {
      initialize_base();
      initialize(name, hc);
   }

   ImageLayer::~ImageLayer() {
   }   

   int ImageLayer::initialize_base() {
         return PV_SUCCESS;
   }

   int ImageLayer::initialize(const char * name, HyPerCol * hc) {
      int status = InputLayer::initialize(name, hc);
      return status;
   }

   Buffer ImageLayer::retrieveData(std::string filename)
   {
      readImage(filename);
      Buffer result(mImage->getHeight(), mImage->getWidth(), getLayerLoc()->nf);
      result.set(mImage->serialize(getLayerLoc()->nf));
      result.rescale(getLayerLoc()->ny, getLayerLoc()->nx, mRescaleMethod, mInterpolationMethod);
      return result;
   }

   bool ImageLayer::readyForNextFile() {
      return true;
   }

   void ImageLayer::readImage(std::string filename)
   {
      pvAssert(parent->columnId() == 0); // readImage is called by retrieveData, which only the root process calls.
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

      mImage = std::unique_ptr<Image>(new Image(std::string(filename)));

      if (usingTempFile) {
         int rmstatus = remove(filename.c_str());
         if(rmstatus) {
            pvError().printf("remove(\"%s\") failed.  Exiting.\n", filename.c_str());
         }
      }
   }

   int ImageLayer::postProcess(double timef, double dt) {

      //TODO: Grayscale conversion stuff here
      return InputLayer::postProcess(timef, dt);
   }

/*
   int ImageLayer::convertToGrayScale(float ** buffer, int nx, int ny, int numBands, InputColorType colorType)
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

   int ImageLayer::convertGrayScaleToMultiBand(float ** buffer, int nx, int ny, int numBands)
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

   int ImageLayer::calcBandWeights(int numBands, float * bandweight, InputColorType colorType) {
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
} // namespace PV
