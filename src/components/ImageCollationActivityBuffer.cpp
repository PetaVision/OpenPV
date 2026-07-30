/*
 * ImageCollationActivityBuffer.cpp
 */

#include "ImageCollationActivityBuffer.hpp"
#include "utils/PathComponents.hpp"

namespace PV {

ImageCollationActivityBuffer::ImageCollationActivityBuffer(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

ImageCollationActivityBuffer::~ImageCollationActivityBuffer() {}

void ImageCollationActivityBuffer::initialize(
      char const *name, PVParams *params, Communicator const *comm) {
   InputActivityBuffer::initialize(name, params, comm);
}

void ImageCollationActivityBuffer::setObjectType() { mObjectType = "ImageCollationActivityBuffer"; }

Response::Status ImageCollationActivityBuffer::allocateDataStructures() {
   auto status = InputActivityBuffer::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   auto outputFileManager = getCommunicator()->getOutputFileManager();
   mURLDownloadTemplate = outputFileManager->makeBlockFilename(std::string("temp.XXXXXX"));
   return Response::SUCCESS;
}

int ImageCollationActivityBuffer::countInputImages() {
   // Calculate file positions for beginning of each frame
   populateFileList();
   InfoLog() << "File " << getInputPath() << " contains " << mFileList.size() << " frames\n";
   int numInputImages = static_cast<int>(mFileList.size()) / getLayerLoc()->nf; //Integer arithmetic
   mFileList.resize(numInputImages * getLayerLoc()->nf); // Drop anything not in an nf-bundle
   return numInputImages;
}

void ImageCollationActivityBuffer::populateFileList() {
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
ImageCollationActivityBuffer::getCurrentFilename(int localBatchIndex, int mpiBatchIndex) const {
   int blockBatchIndex = localBatchIndex + getLayerLoc()->nbatch * mpiBatchIndex;
   int inputIndex      = mBatchIndexer->getIndex(blockBatchIndex);
   return mFileList.at(inputIndex * getLayerLoc()->nf);
}

std::string ImageCollationActivityBuffer::describeInput(int index) {
   // Format resembles "Lines 91-100, input/ListOfImages.txt"
   int nf = getLayerLoc()->nf;
   std::string fileLineStart = std::to_string(index * nf + 1);
   std::string fileLineStop  = std::to_string((index + 1) * nf);
   std::string description("Lines " + fileLineStart + "-" + fileLineStop + ", " + mInputPath);
   return description;
}

Buffer<float> ImageCollationActivityBuffer::retrieveData(int inputIndex) {
   Buffer<float> result;
   int imageWidth, imageHeight;
   std::string firstFilename;
   for (int f = 0; f < getLayerLoc()->nf; ++f) {
      std::string filename = mFileList.at(inputIndex * getLayerLoc()->nf + f);
      auto oneFeature = readImageChannel(filename);
      assert(oneFeature->getFeatures() == 1);
      if (f == 0) {
         firstFilename = filename;
         imageWidth = oneFeature->getWidth();
         imageHeight = oneFeature->getHeight();
         result.resize(imageWidth, imageHeight, getLayerLoc()->nf);
      }
      else {
         FatalIf(
               oneFeature->getWidth() != imageWidth or oneFeature->getHeight() != imageHeight,
               "ImageCollationLayer \"%s\": files \"%s\" and \"%s\" do not have the same "
               "dimensions (%d-by-%d versus %d-by-%d)\n",
               getName(), firstFilename.c_str(), filename.c_str(),
               imageWidth, imageHeight, oneFeature->getWidth(), oneFeature->getHeight());

      }
      result.insertFeatures(*oneFeature, f);
   }
   return result;
}

std::shared_ptr<Image> ImageCollationActivityBuffer::readImageChannel(std::string const &filename) {
   std::shared_ptr<Image> result;

   // Attempt to download our input file if we've been passed a URL or AWS path
   if (filename.find("://") != std::string::npos) {
      std::string tempFilename = downloadURL(filename);
      result = std::make_shared<Image>(tempFilename);
      FatalIf(
            remove(tempFilename.c_str()),
            "Removing temporary file \"%s\" failed.  Exiting.\n",
            tempFilename.c_str());
   }
   else {
      result = std::make_shared<Image>(filename);
   }

   result->convertToGray(false /*alphaChannelFlag*/);
   return result;
}

std::string ImageCollationActivityBuffer::downloadURL(std::string const &url) {
   std::string ext        = extension(url);
   std::string pathstring = mURLDownloadTemplate + ext;
   int tempFileID = ::mkstemps(&pathstring.at(0), static_cast<int>(ext.size()));
   FatalIf(
         tempFileID < 0,
         "Input layer \"%s\" cannot create temporary image file to download \"%s\".\n",
         getName(),
         url.c_str());

   std::string systemstring;
   if (url.find("s3://") != std::string::npos) {
      systemstring = "aws s3 cp \'" + url + "\'" + " " + "\'" + pathstring + "\'";
   }
   else { // URLs other than s3://
      systemstring = "wget -O \'" + pathstring + "\'" + " " + "\'" + url + "\'";
   }

   int const numAttempts = 5;
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
   return pathstring;
}

} // namespace PV
