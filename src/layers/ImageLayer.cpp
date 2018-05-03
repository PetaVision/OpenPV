#include "ImageLayer.hpp"
#include "../arch/mpi/mpi.h"
#include "structures/Image.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>

namespace PV {

ImageLayer::ImageLayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

int ImageLayer::countInputImages() {
   // Check if the input path ends in ".txt" and enable the file list if so
   std::string txt = ".txt";
   if (getInputPath().size() > txt.size()
       && getInputPath().compare(getInputPath().size() - txt.size(), txt.size(), txt) == 0) {
      mUsingFileList = true;

      // Calculate file positions for beginning of each frame
      populateFileList();
      InfoLog() << "File " << getInputPath() << " contains " << mFileList.size() << " frames\n";
      return mFileList.size();
   }
   else {
      mUsingFileList = false;
      return 1;
   }
}

std::string const &ImageLayer::getCurrentFilename(int localBatchElement, int mpiBatchIndex) const {
   if (mUsingFileList) {
      int blockBatchElement = localBatchElement + getLayerLoc()->nbatch * mpiBatchIndex;
      int inputIndex        = mBatchIndexer->getIndex(blockBatchElement);
      return mFileList.at(inputIndex);
   }
   else {
      return getInputPath();
   }
}

Response::Status ImageLayer::registerData(Checkpointer *checkpointer) {
   auto status = InputLayer::registerData(checkpointer);
   if (!Response::completed(status)) {
      return status;
   }
   mURLDownloadTemplate = checkpointer->getOutputPath() + "/temp.XXXXXX";
   return Response::SUCCESS;
}

void ImageLayer::populateFileList() {
   if (getMPIBlock()->getRank() == 0) {
      std::string line;
      mFileList.clear();
      InfoLog() << "Reading list: " << getInputPath() << "\n";
      std::ifstream infile(getInputPath(), std::ios_base::in);
      FatalIf(
            infile.fail(), "Unable to open \"%s\": %s\n", getInputPath().c_str(), strerror(errno));
      while (getline(infile, line, '\n')) {
         std::string noWhiteSpace = line;
         noWhiteSpace.erase(
               std::remove_if(noWhiteSpace.begin(), noWhiteSpace.end(), ::isspace),
               noWhiteSpace.end());
         if (!noWhiteSpace.empty()) {
            mFileList.push_back(noWhiteSpace);
         }
      }
      FatalIf(
            mFileList.empty(),
            "%s inputPath file list \"%s\" is empty.\n",
            getDescription_c(),
            getInputPath().c_str());
   }
}

Buffer<float> ImageLayer::retrieveData(int inputIndex) {
   std::string filename;
   if (mUsingFileList) {
      filename = mFileList.at(inputIndex);
   }
   else {
      filename = getInputPath();
   }
   readImage(filename);

   if (mImage->getFeatures() != getLayerLoc()->nf) {
      switch (getLayerLoc()->nf) {
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
            Fatal() << "Failed to read " << filename << ": Could not convert "
                    << mImage->getFeatures() << " channels to " << getLayerLoc()->nf << std::endl;
            break;
      }
   }

   Buffer<float> result(
         mImage->asVector(), mImage->getWidth(), mImage->getHeight(), getLayerLoc()->nf);
   return result;
}

void ImageLayer::readImage(std::string filename) {
   const PVLayerLoc *loc = getLayerLoc();
   bool usingTempFile    = false;

   // Attempt to download our input file if we've been passed a URL or AWS path
   if (filename.find("://") != std::string::npos) {
      usingTempFile          = true;
      std::string extension  = filename.substr(filename.find_last_of("."));
      std::string pathstring = mURLDownloadTemplate + extension;
      char tempStr[256];
      strcpy(tempStr, pathstring.c_str());
      int tempFileID = mkstemps(tempStr, extension.size());
      pathstring     = std::string(tempStr);
      FatalIf(tempFileID < 0, "Cannot create temp image file.\n");
      std::string systemstring;

      if (filename.find("s3://") != std::string::npos) {
         systemstring = std::string("aws s3 cp \'") + filename + std::string("\' ") + pathstring;
      }
      else { // URLs other than s3://
         systemstring = std::string("wget -O ") + pathstring + std::string(" \'") + filename
                        + std::string("\'");
      }

      filename              = pathstring;
      const int numAttempts = 5;
      for (int attemptNum = 0; attemptNum < numAttempts; attemptNum++) {
         if (system(systemstring.c_str()) == 0) {
            break;
         }
         sleep(1);
         FatalIf(
               attemptNum == numAttempts - 1,
               "download command \"%s\" failed: %s.  Exiting\n",
               systemstring.c_str(),
               strerror(errno));
      }
   }

   mImage = std::unique_ptr<Image>(new Image(std::string(filename)));

   FatalIf(
         usingTempFile && remove(filename.c_str()),
         "remove(\"%s\") failed.  Exiting.\n",
         filename.c_str());
}

std::string ImageLayer::describeInput(int index) {
   std::string description("");
   if (mUsingFileList) {
      description = mFileList.at(index);
   }
   return description;
}
}
