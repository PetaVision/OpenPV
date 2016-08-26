#include "ImageLayer.hpp"
#include "utils/Image.hpp"
#include "../arch/mpi/mpi.h"

#include <assert.h>
#include <string.h>
#include <iostream>

namespace PV {

   ImageLayer::ImageLayer(const char * name, HyPerCol * hc) {
      initialize(name, hc);
   }

   Buffer ImageLayer::retrieveData(std::string filename, int batchIndex)  {
      readImage(filename);
      if (mImage->getFeatures() != getLayerLoc()->nf) {
         switch(getLayerLoc()->nf) {
            case 1: // Grayscale
               mImage->convertToGray(false);
               break;
            case 2: // Grayscale + Alpha
               mImage->convertToGray(true);
               break;
            case 3: // RGB
               mImage->convertToColor(false);
               break;
            case 4: // RGBA
               mImage->convertToColor(true);
               break;
            default:
               pvError() << "Failed to read " << filename
                  << ": Could not convert " << mImage->getFeatures()
                  << " channels to " << getLayerLoc()->nf
                  << std::endl;
               break;

         }
      }

      Buffer result(mImage->asVector(), mImage->getWidth(), mImage->getHeight(), getLayerLoc()->nf);
      return result;
   }

   void ImageLayer::readImage(std::string filename) {
      const PVLayerLoc *loc = getLayerLoc();
      bool usingTempFile = false;

      // Attempt to download our input file if we've been passed a URL or AWS path
      if (filename.find("://") != std::string::npos) {
         usingTempFile = true;
         std::string extension = filename.substr(filename.find_last_of("."));
         std::string pathstring = parent->getOutputPath() + std::string("/temp.XXXXXX") + extension;
         char tempStr[256];
         strcpy(tempStr, pathstring.c_str());
         int tempFileID = mkstemps(tempStr, extension.size());
         pathstring = std::string(tempStr);
         pvErrorIf(tempFileID < 0, "Cannot create temp image file.\n");
         std::string systemstring;

         if (filename.find("s3://") != std::string::npos) {
            systemstring = std::string("aws s3 cp \'") + filename + std::string("\' ") + pathstring;
         }
         else { // URLs other than s3://
            systemstring = std::string("wget -O ") + pathstring + std::string(" \'") + filename + std::string("\'");
         }

         filename = pathstring;
         const int numAttempts = 5;
         for (int attemptNum = 0; attemptNum < numAttempts; attemptNum++) {
            if (system(systemstring.c_str()) == 0) {
               break;
            }
            sleep(1);
            pvErrorIf(attemptNum == numAttempts - 1,
                  "download command \"%s\" failed: %s.  Exiting\n", systemstring.c_str(), strerror(errno));
         }
      }

      mImage = std::unique_ptr<Image>(new Image(std::string(filename)));

      pvErrorIf(usingTempFile && remove(filename.c_str()), "remove(\"%s\") failed.  Exiting.\n", filename.c_str());
   }
}
