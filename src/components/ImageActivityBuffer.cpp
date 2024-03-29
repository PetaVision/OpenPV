/*
 * ImageActivityBuffer.cpp
 *
 *  Created on: Jul 22, 2015
 *      Author: Sheng Lundquist
 */

#include "ImageActivityBuffer.hpp"
#include <algorithm>

namespace PV {

ImageActivityBuffer::ImageActivityBuffer(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

ImageActivityBuffer::~ImageActivityBuffer() {}

void ImageActivityBuffer::initialize(char const *name, PVParams *params, Communicator const *comm) {
   InputActivityBuffer::initialize(name, params, comm);
}

void ImageActivityBuffer::setObjectType() { mObjectType = "ImageActivityBuffer"; }

Response::Status ImageActivityBuffer::registerData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = InputActivityBuffer::registerData(message);
   // This can be moved to AllocateDataStructures stage, or even CommunicateInitInfo,
   // since it only requires the OutputFileManager have the correct OutputPath string.
   if (!Response::completed(status)) {
      return status;
   }
   mURLDownloadTemplate =
         getCommunicator()->getOutputFileManager()->makeBlockFilename(std::string("temp.XXXXXX"));
   return Response::SUCCESS;
}

int ImageActivityBuffer::countInputImages() {
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

void ImageActivityBuffer::populateFileList() {
   if (getCommunicator()->getIOMPIBlock()->getRank() == 0) {
      std::string line;
      mFileList.clear();
      InfoLog() << "Reading list: " << getInputPath() << "\n";
      std::ifstream infile(getInputPath(), std::ios_base::in);
      FatalIf(
            infile.fail(), "Unable to open \"%s\": %s\n", getInputPath().c_str(), strerror(errno));
      while (getline(infile, line, '\n')) {
         auto firstNonWhitespace = (std::string::size_type)0;
         while (firstNonWhitespace < line.size() and isspace(line[firstNonWhitespace])) {
            firstNonWhitespace++;
         }
         auto firstTrailingWhitespace = line.size();
         while (firstTrailingWhitespace > firstNonWhitespace
                and isspace(line[firstTrailingWhitespace - 1])) {
            firstTrailingWhitespace--;
         }
         if (firstTrailingWhitespace > firstNonWhitespace) {
            auto trimmedLength      = firstTrailingWhitespace - firstNonWhitespace;
            std::string trimmedLine = line.substr(firstNonWhitespace, trimmedLength);
            mFileList.push_back(trimmedLine);
         }
      }
      FatalIf(
            mFileList.empty(),
            "%s inputPath file list \"%s\" is empty.\n",
            getDescription_c(),
            getInputPath().c_str());
   }
}

std::string const &
ImageActivityBuffer::getCurrentFilename(int localBatchIndex, int mpiBatchIndex) const {
   if (mUsingFileList) {
      int blockBatchIndex = localBatchIndex + getLayerLoc()->nbatch * mpiBatchIndex;
      int inputIndex      = mBatchIndexer->getIndex(blockBatchIndex);
      return mFileList.at(inputIndex);
   }
   else {
      return getInputPath();
   }
}

std::string ImageActivityBuffer::describeInput(int index) {
   std::string description("");
   if (mUsingFileList) {
      description = mFileList.at(index);
   }
   return description;
}

Buffer<float> ImageActivityBuffer::retrieveData(int inputIndex) {
   std::string filename;
   if (mUsingFileList) {
      filename = mFileList.at(inputIndex);
   }
   else {
      filename = getInputPath();
   }
   readImage(filename);

   int const numFeatures = getLayerLoc()->nf;
   if (mImage->getFeatures() != numFeatures) {
      switch (numFeatures) {
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
                    << mImage->getFeatures() << " channels to " << numFeatures << std::endl;
            break;
      }
   }

   Buffer<float> result(mImage->asVector(), mImage->getWidth(), mImage->getHeight(), numFeatures);
   return result;
}

void ImageActivityBuffer::readImage(std::string filename) {
   bool usingTempFile = false;

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

} // namespace PV
