/*
 * InputLayer.cpp
 * Formerly InputLayer.cpp
 */

#include "InputLayer.hpp"
#include "columns/RandomSeed.hpp"
#include "utils/BufferUtilsMPI.hpp"

#include <algorithm>
#include <cfloat>

namespace PV {

InputLayer::InputLayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

InputLayer::~InputLayer() { Communicator::freeDatatypes(mDatatypes); }

int InputLayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

int InputLayer::allocateDataStructures() {
   int status = HyPerLayer::allocateDataStructures();
   if (status != PV_SUCCESS) {
      return status;
   }

   if (mWriteFrameToTimestamp) {
      std::string timestampFilename =
            std::string(parent->getOutputPath()) + std::string("/timestamps/");
      ensureDirExists(parent->getCommunicator(), timestampFilename.c_str());
      timestampFilename += name + std::string(".txt");
      if (getParent()->getCommunicator()->commRank() == 0) {
         std::string cpFileStreamLabel(getName());
         cpFileStreamLabel.append("_TimestampState");
         // If checkpoint read is set, append, otherwise, clobber
         if (getParent()->getCheckpointReadFlag()) {
            struct stat statbuf;
            if (PV_stat(timestampFilename.c_str(), &statbuf) != 0) {
               WarnLog().printf(
                     "%s: timestamp file \"%s\" unable to be found.  Creating new file.\n",
                     getDescription_c(),
                     timestampFilename.c_str());
               mTimestampStream = new CheckpointableFileStream(
                     timestampFilename.c_str(),
                     std::ios_base::out,
                     cpFileStreamLabel,
                     parent->getVerifyWrites());
            }
            else {
               mTimestampStream = new CheckpointableFileStream(
                     timestampFilename.c_str(),
                     std::ios_base::in | std::ios_base::out,
                     cpFileStreamLabel,
                     false);
            }
         }
         else {
            mTimestampStream = new CheckpointableFileStream(
                  timestampFilename.c_str(),
                  std::ios_base::out,
                  cpFileStreamLabel,
                  parent->getVerifyWrites());
         }
         pvAssert(mTimestampStream);
      }
   }

   int numBatch = parent->getNBatch();

   if (parent->columnId() == 0) {

      // Calculate file positions for beginning of each frame
      if (mUsingFileList) {
         populateFileList();
         InfoLog() << "File " << mInputPath << " contains " << mFileList.size() << " frames\n";
      }

      mInputData.resize(numBatch);
      for (int b = 0; b < numBatch; ++b) {
         mInputData.at(b).resize(getLayerLoc()->ny, getLayerLoc()->nx, getLayerLoc()->nf);
      }
      initializeBatchIndexer(mFileList.size());
      mBatchIndexer->setWrapToStartIndex(mResetToStartOnLoop);

      // We want to fill the activity buffer with the initial data without actually advancing
      // our indices, so this is a quick hack to "rewind" after the initial nextInput()
      std::vector<int> tempIndices = mBatchIndexer->getIndices();
      nextInput(parent->simulationTime(), 0);
      mBatchIndexer->setIndices(tempIndices);
   }
   else {
      nextInput(parent->simulationTime(), 0);
   }

   // create mpi_datatypes for border transfer
   mDatatypes = Communicator::newDatatypes(getLayerLoc());
   exchange();

   return PV_SUCCESS;
}

void InputLayer::initializeBatchIndexer(int fileCount) {
   int localBatchCount   = parent->getNBatch();
   int mpiBatchIndex     = parent->commBatch();
   int globalBatchOffset = localBatchCount * mpiBatchIndex;
   mBatchIndexer         = std::unique_ptr<BatchIndexer>(
         new BatchIndexer(
               parent->getNBatchGlobal(),
               mpiBatchIndex,
               parent->numCommBatches(),
               fileCount,
               mBatchMethod));
   for (int b = 0; b < localBatchCount; ++b) {
      mBatchIndexer->specifyBatching(
            b,
            mStartFrameIndex.at(globalBatchOffset + b),
            mSkipFrameIndex.at(globalBatchOffset + b));
      mBatchIndexer->initializeBatch(b);
   }
   mBatchIndexer->setRandomSeed(RandomSeed::instance()->getInitialSeed() + mRandomSeed);
}

// Virtual method used to spend multiple display periods on one file.
// Can be used to implement lists of collections or modifications to
// the loaded file, such as streaming audio or video.
bool InputLayer::readyForNextFile() {
   // A display period <= 0 means we never change files
   return mDisplayPeriod > 0;
}

int InputLayer::updateState(double time, double dt) {
   Communicator *icComm = getParent()->getCommunicator();
   if (readyForNextFile()) {

      // Write file path to timestamp file
      if (icComm->commRank() == 0 && mTimestampStream) {
         std::ostringstream outStrStream;
         outStrStream.precision(15);
         int kb0 = getLayerLoc()->kb0;
         for (int b = 0; b < parent->getNBatch(); ++b) {
            if (mUsingFileList) {
               outStrStream << "[" << getName() << "] time: " << time << ", batch: " << b + kb0
                            << ", index: " << mBatchIndexer->getIndex(b) << ","
                            << mFileList.at(mBatchIndexer->getIndex(b)) << "\n";
            }
            else {
               outStrStream << "[" << getName() << "] time: " << time << ", batch: " << b + kb0
                            << ", index: " << mBatchIndexer->getIndex(b) << "\n";
            }
         }
         size_t len = outStrStream.str().length();
         mTimestampStream->write(outStrStream.str().c_str(), len);
         mTimestampStream->flush();
      }

      // Read in the next file
      nextInput(time, dt);
   }
   return PV_SUCCESS;
}

void InputLayer::nextInput(double timef, double dt) {
   for (int b = 0; b < parent->getNBatch(); b++) {
      if (parent->columnId() == 0) {
         std::string fileName = mInputPath;
         if (mUsingFileList) {
            fileName = mFileList.at(mBatchIndexer->nextIndex(b));
         }
         mInputData.at(b) = retrieveData(fileName, b);
         fitBufferToLayer(mInputData.at(b));
      }
      scatterInput(b);
   }
   postProcess(timef, dt);
}

int InputLayer::scatterInput(int batchIndex) {
   const int rank        = parent->columnId();
   MPI_Comm mpiComm      = parent->getCommunicator()->communicator();
   float *activityBuffer = getActivity() + batchIndex * getNumExtended();
   const PVLayerLoc *loc = getLayerLoc();
   const PVHalo *halo    = &loc->halo;
   int activityWidth     = loc->nx + (mUseInputBCflag ? halo->lt + halo->rt : 0);
   int activityHeight    = loc->ny + (mUseInputBCflag ? halo->up + halo->dn : 0);
   Buffer<float> croppedBuffer;

   if (rank == 0) {
      croppedBuffer = mInputData.at(batchIndex);
      BufferUtils::scatter<float>(parent->getCommunicator(), croppedBuffer, loc->nx, loc->ny);
   }
   else {
      croppedBuffer.resize(activityWidth, activityHeight, loc->nf);
      BufferUtils::scatter<float>(parent->getCommunicator(), croppedBuffer, loc->nx, loc->ny);
   }

   // At this point, croppedBuffer has the correct data for this
   // process, regardless of if we are root or not. Clear the current
   // activity buffer, then copy the input data over row by row.
   for (int n = 0; n < getNumExtended(); ++n) {
      activityBuffer[n] = mPadValue;
   }

   for (int y = 0; y < activityHeight; ++y) {
      for (int x = 0; x < activityWidth; ++x) {
         for (int f = 0; f < numFeatures; ++f) {
            int activityIndex = kIndex(
                  halo->lt + x,
                  halo->up + y,
                  f,
                  loc->nx + halo->lt + halo->rt,
                  loc->ny + halo->up + halo->dn,
                  numFeatures);
            activityBuffer[activityIndex] = croppedBuffer.at(x, y, f);
         }
      }
   }

   return PV_SUCCESS;
}

void InputLayer::fitBufferToLayer(Buffer<float> &buffer) {
   pvAssert(parent->columnId() == 0);
   const PVLayerLoc *loc  = getLayerLoc();
   const PVHalo *halo     = &loc->halo;
   const int targetWidth  = loc->nxGlobal + (mUseInputBCflag ? (halo->lt + halo->rt) : 0);
   const int targetHeight = loc->nyGlobal + (mUseInputBCflag ? (halo->dn + halo->up) : 0);

   FatalIf(
         buffer.getFeatures() != loc->nf,
         "ERROR: Input for layer %s has %d features, but layer has %d\n.",
         getName(),
         buffer.getFeatures(),
         loc->nf);

   if (mAutoResizeFlag) {
      BufferUtils::rescale(
            buffer, targetWidth, targetHeight, mRescaleMethod, mInterpolationMethod, mAnchor);
      buffer.translate(-mOffsetX, -mOffsetY);
   }
   else {
      buffer.grow(targetWidth, targetHeight, mAnchor);
      buffer.translate(-mOffsetX, -mOffsetY);
      buffer.crop(targetWidth, targetHeight, mAnchor);
   }
}

// Apply normalizeLuminanceFlag, normalizeStdDev, and inverseFlag, which can be done pixel-by-pixel
// after scattering.
int InputLayer::postProcess(double timef, double dt) {
   int numExtended = getNumExtended();

   // if normalizeLuminanceFlag == true:
   //     if normalizeStdDev is true, then scale so that average luminance to be 0 and std. dev. of
   //     luminance to be 1.
   //     if normalizeStdDev is false, then scale so that minimum is 0 and maximum is 1
   // if normalizeLuminanceFlag == true and the image in buffer is completely flat, force all values
   // to zero
   for (int b = 0; b < parent->getNBatch(); b++) {
      float *buf = getActivity() + b * numExtended;
      if (mNormalizeLuminanceFlag) {
         if (mNormalizeStdDev) {
            float image_sum  = 0.0;
            float image_sum2 = 0.0;
            for (int k = 0; k < numExtended; k++) {
               image_sum += buf[k];
               image_sum2 += buf[k] * buf[k];
            }
            float image_ave  = image_sum / numExtended;
            float image_ave2 = image_sum2 / numExtended;
            MPI_Allreduce(
                  MPI_IN_PLACE,
                  &image_ave,
                  1,
                  MPI_FLOAT,
                  MPI_SUM,
                  parent->getCommunicator()->communicator());
            image_ave /= parent->getCommunicator()->commSize();
            MPI_Allreduce(
                  MPI_IN_PLACE,
                  &image_ave2,
                  1,
                  MPI_FLOAT,
                  MPI_SUM,
                  parent->getCommunicator()->communicator());
            image_ave2 /= parent->getCommunicator()->commSize();

            // set mean to zero
            for (int k = 0; k < numExtended; k++) {
               buf[k] -= image_ave;
            }

            // set std dev to 1
            float image_std = sqrtf(image_ave2 - image_ave * image_ave);
            if (image_std == 0 || image_std != image_std) {
               for (int k = 0; k < numExtended; k++) {
                  buf[k] = 0.0f;
               }
            }
            else {
               for (int k = 0; k < numExtended; k++) {
                  buf[k] /= image_std;
               }
            }
         }
         else {
            float image_max = -FLT_MAX;
            float image_min = FLT_MAX;
            for (int k = 0; k < numExtended; k++) {
               image_max = buf[k] > image_max ? buf[k] : image_max;
               image_min = buf[k] < image_min ? buf[k] : image_min;
            }
            MPI_Allreduce(
                  MPI_IN_PLACE,
                  &image_max,
                  1,
                  MPI_FLOAT,
                  MPI_MAX,
                  parent->getCommunicator()->communicator());
            MPI_Allreduce(
                  MPI_IN_PLACE,
                  &image_min,
                  1,
                  MPI_FLOAT,
                  MPI_MIN,
                  parent->getCommunicator()->communicator());
            if (image_max > image_min) {
               float image_stretch = 1.0f / (image_max - image_min);
               for (int k = 0; k < numExtended; k++) {
                  buf[k] -= image_min;
                  buf[k] *= image_stretch;
               }
            }
            else {
               for (int k = 0; k < numExtended; k++) {
                  buf[k] = 0.0f;
               }
            }
         }
      }
      if (mInverseFlag) {
         for (int k = 0; k < numExtended; k++) {
            // If normalizeLuminanceFlag is true, should the effect of inverseFlag be buf[k] =
            // -buf[k]?
            buf[k] = 1.0f - buf[k];
         }
      }
   }
   return PV_SUCCESS;
}

void InputLayer::exchange() {
   std::vector<MPI_Request> req{};
   for (int b = 0; b < getLayerLoc()->nbatch; ++b) {
      parent->getCommunicator()->exchange(
            getActivity() + b * getNumExtended(), mDatatypes, getLayerLoc(), req);
      parent->getCommunicator()->wait(req);
      pvAssert(req.empty());
   }
}

double InputLayer::getDeltaUpdateTime() { return mDisplayPeriod > 0 ? mDisplayPeriod : DBL_MAX; }

int InputLayer::requireChannel(int channelNeeded, int *numChannelsResult) {
   if (parent->columnId() == 0) {
      ErrorLog().printf("%s cannot be a post-synaptic layer.\n", getDescription_c());
   }
   *numChannelsResult = 0;
   return PV_FAILURE;
}

int InputLayer::allocateV() {
   clayer->V = nullptr;
   return PV_SUCCESS;
}

int InputLayer::initializeV() {
   pvAssert(getV() == nullptr);
   return PV_SUCCESS;
}

int InputLayer::initializeActivity() { return PV_SUCCESS; }

void InputLayer::populateFileList() {
   if (mUsingFileList && parent->columnId() == 0) {
      std::string line;
      mFileList.clear();
      InfoLog() << "Reading list: " << mInputPath << "\n";
      std::ifstream infile(mInputPath, std::ios_base::in);
      FatalIf(infile.fail(), "Unable to open \"%s\": %s\n", mInputPath.c_str(), strerror(errno));
      while (getline(infile, line, '\n')) {
         std::string noWhiteSpace = line;
         noWhiteSpace.erase(
               std::remove_if(noWhiteSpace.begin(), noWhiteSpace.end(), ::isspace),
               noWhiteSpace.end());
         if (!noWhiteSpace.empty()) {
            mFileList.push_back(noWhiteSpace);
         }
      }
   }
}

int InputLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLayer::ioParamsFillGroup(ioFlag);
   ioParam_displayPeriod(ioFlag);
   ioParam_inputPath(ioFlag);
   ioParam_offsetAnchor(ioFlag);
   ioParam_offsets(ioFlag);
   ioParam_autoResizeFlag(ioFlag);
   ioParam_aspectRatioAdjustment(ioFlag);
   ioParam_interpolationMethod(ioFlag);
   ioParam_inverseFlag(ioFlag);
   ioParam_normalizeLuminanceFlag(ioFlag);
   ioParam_normalizeStdDev(ioFlag);
   ioParam_useInputBCflag(ioFlag);
   ioParam_padValue(ioFlag);
   ioParam_batchMethod(ioFlag);
   ioParam_randomSeed(ioFlag);
   ioParam_start_frame_index(ioFlag);
   ioParam_skip_frame_index(ioFlag);
   ioParam_writeFrameToTimestamp(ioFlag);
   ioParam_resetToStartOnLoop(ioFlag);
   return status;
}

int InputLayer::registerData(Checkpointer *checkpointer, std::string const &objName) {
   int status = HyPerLayer::registerData(checkpointer, objName);
   if (parent->getCommunicator()->commRank() == 0) {
      mBatchIndexer->registerData(checkpointer, objName);
   }
   if (mTimestampStream) {
      mTimestampStream->registerData(checkpointer, objName);
   }
   return status;
}

int InputLayer::readStateFromCheckpoint(Checkpointer *checkpointer) {
   int status = PV_SUCCESS;
   if (initializeFromCheckpointFlag) {
      int status = HyPerLayer::readStateFromCheckpoint(checkpointer);
      if (mBatchIndexer) {
         pvAssert(parent->getCommunicator()->commRank() == 0) {
            mBatchIndexer->readStateFromCheckpoint(checkpointer);
         }
      }
   }
   return status;
}

int InputLayer::checkValidAnchorString(const char *offsetAnchor) {
   int status = PV_SUCCESS;
   if (offsetAnchor == NULL || strlen(offsetAnchor) != (size_t)2) {
      status = PV_FAILURE;
   }
   else {
      char xOffsetAnchor = offsetAnchor[1];
      if (xOffsetAnchor != 'l' && xOffsetAnchor != 'c' && xOffsetAnchor != 'r') {
         status = PV_FAILURE;
      }
      char yOffsetAnchor = offsetAnchor[0];
      if (yOffsetAnchor != 't' && yOffsetAnchor != 'c' && yOffsetAnchor != 'b') {
         status = PV_FAILURE;
      }
   }
   return status;
}

void InputLayer::ioParam_inputPath(enum ParamsIOFlag ioFlag) {
   char *tempString = nullptr;
   if (ioFlag == PARAMS_IO_WRITE) {
      tempString = strdup(mInputPath.c_str());
   }
   parent->parameters()->ioParamStringRequired(ioFlag, name, "inputPath", &tempString);
   if (ioFlag == PARAMS_IO_READ) {
      mInputPath = std::string(tempString);
      // Check if the input path ends in ".txt" and enable the file list if so
      std::string txt = ".txt";
      if (mInputPath.size() > txt.size()
          && mInputPath.compare(mInputPath.size() - txt.size(), txt.size(), txt) == 0) {
         mUsingFileList = true;
      }
      else {
         mUsingFileList = false;
      }
   }
   free(tempString);
}

void InputLayer::ioParam_useInputBCflag(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "useInputBCflag", &mUseInputBCflag, mUseInputBCflag);
}

int InputLayer::ioParam_offsets(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "offsetX", &mOffsetX, mOffsetX);
   parent->parameters()->ioParamValue(ioFlag, name, "offsetY", &mOffsetY, mOffsetY);
   return PV_SUCCESS;
}

void InputLayer::ioParam_offsetAnchor(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      char *offsetAnchor = nullptr;
      parent->parameters()->ioParamString(ioFlag, name, "offsetAnchor", &offsetAnchor, "tl");
      if (checkValidAnchorString(offsetAnchor) == PV_FAILURE) {
         Fatal() << "Invalid value for offsetAnchor\n";
      }
      if (strcmp(offsetAnchor, "tl") == 0) {
         mAnchor = Buffer<float>::NORTHWEST;
      }
      else if (strcmp(offsetAnchor, "tc") == 0) {
         mAnchor = Buffer<float>::NORTH;
      }
      else if (strcmp(offsetAnchor, "tr") == 0) {
         mAnchor = Buffer<float>::NORTHEAST;
      }
      else if (strcmp(offsetAnchor, "cl") == 0) {
         mAnchor = Buffer<float>::WEST;
      }
      else if (strcmp(offsetAnchor, "cc") == 0) {
         mAnchor = Buffer<float>::CENTER;
      }
      else if (strcmp(offsetAnchor, "cr") == 0) {
         mAnchor = Buffer<float>::EAST;
      }
      else if (strcmp(offsetAnchor, "bl") == 0) {
         mAnchor = Buffer<float>::SOUTHWEST;
      }
      else if (strcmp(offsetAnchor, "bc") == 0) {
         mAnchor = Buffer<float>::SOUTH;
      }
      else if (strcmp(offsetAnchor, "br") == 0) {
         mAnchor = Buffer<float>::SOUTHEAST;
      }
      else {
         if (parent->columnId() == 0) {
            ErrorLog().printf(
                  "%s: offsetAnchor must be a two-letter string.  The first character must be "
                  "\"t\", \"c\", or \"b\" (for top, center or bottom); and the second character "
                  "must be \"l\", \"c\", or \"r\" (for left, center or right).\n",
                  getDescription_c());
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      free(offsetAnchor);
   }
   else { // Writing
      // The opposite of above. Find a better way to do this that isn't so gross
      char *offsetAnchor = (char *)calloc(3, sizeof(char));
      offsetAnchor[2]    = '\0';
      switch (mAnchor) {
         case Buffer<float>::NORTH:
         case Buffer<float>::NORTHWEST:
         case Buffer<float>::NORTHEAST: offsetAnchor[0] = 't'; break;
         case Buffer<float>::WEST:
         case Buffer<float>::CENTER:
         case Buffer<float>::EAST: offsetAnchor[0] = 'c'; break;
         case Buffer<float>::SOUTHWEST:
         case Buffer<float>::SOUTH:
         case Buffer<float>::SOUTHEAST: offsetAnchor[0] = 'b'; break;
      }
      switch (mAnchor) {
         case Buffer<float>::NORTH:
         case Buffer<float>::CENTER:
         case Buffer<float>::SOUTH: offsetAnchor[1] = 'c'; break;
         case Buffer<float>::EAST:
         case Buffer<float>::NORTHEAST:
         case Buffer<float>::SOUTHEAST: offsetAnchor[1] = 'r'; break;
         case Buffer<float>::WEST:
         case Buffer<float>::NORTHWEST:
         case Buffer<float>::SOUTHWEST: offsetAnchor[1] = 'l'; break;
      }
      parent->parameters()->ioParamString(ioFlag, name, "offsetAnchor", &offsetAnchor, "tl");
      free(offsetAnchor);
   }
}

void InputLayer::ioParam_autoResizeFlag(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "autoResizeFlag", &mAutoResizeFlag, mAutoResizeFlag);
}

void InputLayer::ioParam_aspectRatioAdjustment(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "autoResizeFlag"));
   if (mAutoResizeFlag) {
      char *aspectRatioAdjustment = nullptr;
      if (ioFlag == PARAMS_IO_WRITE) {
         switch (mRescaleMethod) {
            case BufferUtils::CROP: aspectRatioAdjustment = strdup("crop"); break;
            case BufferUtils::PAD: aspectRatioAdjustment  = strdup("pad"); break;
         }
      }
      parent->parameters()->ioParamString(
            ioFlag, name, "aspectRatioAdjustment", &aspectRatioAdjustment, "crop");
      if (ioFlag == PARAMS_IO_READ) {
         assert(aspectRatioAdjustment);
         for (char *c = aspectRatioAdjustment; *c; c++) {
            *c = tolower(*c);
         }
      }
      if (strcmp(aspectRatioAdjustment, "crop") == 0) {
         mRescaleMethod = BufferUtils::CROP;
      }
      else if (strcmp(aspectRatioAdjustment, "pad") == 0) {
         mRescaleMethod = BufferUtils::PAD;
      }
      else {
         if (parent->columnId() == 0) {
            ErrorLog().printf(
                  "%s: aspectRatioAdjustment must be either \"crop\" or \"pad\".\n",
                  getDescription_c());
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      free(aspectRatioAdjustment);
   }
}

void InputLayer::ioParam_interpolationMethod(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "autoResizeFlag"));
   if (mAutoResizeFlag) {
      char *interpolationMethodString = nullptr;
      if (ioFlag == PARAMS_IO_READ) {
         parent->parameters()->ioParamString(
               ioFlag,
               name,
               "interpolationMethod",
               &interpolationMethodString,
               "bicubic",
               true /*warn if absent*/);
         assert(interpolationMethodString);
         for (char *c = interpolationMethodString; *c; c++) {
            *c = tolower(*c);
         }
         if (!strncmp(interpolationMethodString, "bicubic", strlen("bicubic"))) {
            mInterpolationMethod = BufferUtils::BICUBIC;
         }
         else if (
               !strncmp(interpolationMethodString, "nearestneighbor", strlen("nearestneighbor"))) {
            mInterpolationMethod = BufferUtils::NEAREST;
         }
         else {
            if (parent->columnId() == 0) {
               ErrorLog().printf(
                     "%s: interpolationMethod must be either \"bicubic\" or \"nearestNeighbor\".\n",
                     getDescription_c());
            }
            MPI_Barrier(parent->getCommunicator()->communicator());
            exit(EXIT_FAILURE);
         }
      }
      else {
         assert(ioFlag == PARAMS_IO_WRITE);
         switch (mInterpolationMethod) {
            case BufferUtils::BICUBIC: interpolationMethodString = strdup("bicubic"); break;
            case BufferUtils::NEAREST: interpolationMethodString = strdup("nearestNeighbor"); break;
         }
         parent->parameters()->ioParamString(
               ioFlag,
               name,
               "interpolationMethod",
               &interpolationMethodString,
               "bicubic",
               true /*warn if absent*/);
      }
      free(interpolationMethodString);
   }
}

void InputLayer::ioParam_inverseFlag(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "inverseFlag", &mInverseFlag, mInverseFlag);
}

void InputLayer::ioParam_normalizeLuminanceFlag(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "normalizeLuminanceFlag", &mNormalizeLuminanceFlag, mNormalizeLuminanceFlag);
}

void InputLayer::ioParam_normalizeStdDev(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "normalizeLuminanceFlag"));
   if (mNormalizeLuminanceFlag) {
      parent->parameters()->ioParamValue(
            ioFlag, name, "normalizeStdDev", &mNormalizeStdDev, mNormalizeStdDev);
   }
}
void InputLayer::ioParam_padValue(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "padValue", &mPadValue, mPadValue);
}

void InputLayer::ioParam_InitVType(enum ParamsIOFlag ioFlag) {
   assert(mInitVObject == NULL);
   return;
}

void InputLayer::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      triggerLayerName = NULL;
      triggerFlag      = false;
      parent->parameters()->handleUnnecessaryStringParameter(
            name, "triggerLayerName", NULL /*correct value*/);
   }
}

void InputLayer::ioParam_displayPeriod(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "displayPeriod", &mDisplayPeriod, mDisplayPeriod);
}

void InputLayer::ioParam_batchMethod(enum ParamsIOFlag ioFlag) {
   char *batchMethod = nullptr;
   if (ioFlag == PARAMS_IO_WRITE) {
      switch (mBatchMethod) {
         case BatchIndexer::BYFILE: batchMethod      = strdup("byFile"); break;
         case BatchIndexer::BYLIST: batchMethod      = strdup("byList"); break;
         case BatchIndexer::BYSPECIFIED: batchMethod = strdup("bySpecified"); break;
         case BatchIndexer::RANDOM: batchMethod      = strdup("random"); break;
      }
   }
   parent->parameters()->ioParamString(ioFlag, name, "batchMethod", &batchMethod, "byFile");
   if (strcmp(batchMethod, "byImage") == 0 || strcmp(batchMethod, "byFile") == 0) {
      mBatchMethod = BatchIndexer::BYFILE;
   }
   else if (strcmp(batchMethod, "byMovie") == 0 || strcmp(batchMethod, "byList") == 0) {
      mBatchMethod = BatchIndexer::BYLIST;
   }
   else if (strcmp(batchMethod, "bySpecified") == 0) {
      mBatchMethod = BatchIndexer::BYSPECIFIED;
   }
   else if (strcmp(batchMethod, "random") == 0) {
      mBatchMethod = BatchIndexer::RANDOM;
   }
   else {
      Fatal() << getName() << ": Input layer " << name
              << " batchMethod not recognized. Options "
                 "are \"byFile\", \"byList\", bySpecified, and random.\n";
   }
   free(batchMethod);
}

void InputLayer::ioParam_randomSeed(enum ParamsIOFlag ioFlag) {
   if (mBatchMethod == BatchIndexer::RANDOM) {
      parent->parameters()->ioParamValue(ioFlag, name, "randomSeed", &mRandomSeed, mRandomSeed);
   }
}

void InputLayer::ioParam_start_frame_index(enum ParamsIOFlag ioFlag) {
   int *paramsStartFrameIndex;
   int length = 0;
   if (ioFlag == PARAMS_IO_WRITE) {
      length                = mStartFrameIndex.size();
      paramsStartFrameIndex = static_cast<int *>(calloc(length, sizeof(int)));
      for (int i = 0; i < length; ++i) {
         paramsStartFrameIndex[i] = mStartFrameIndex.at(i);
      }
   }
   this->getParent()->parameters()->ioParamArray(
         ioFlag, this->getName(), "start_frame_index", &paramsStartFrameIndex, &length);
   FatalIf(
         length > 0 && length != parent->getNBatchGlobal(),
         "%s: start_frame_index requires either 0 or nbatch values.\n",
         getName());
   mStartFrameIndex.clear();
   mStartFrameIndex.resize(length < parent->getNBatchGlobal() ? parent->getNBatchGlobal() : length);
   if (length > 0) {
      for (int i = 0; i < length; ++i) {
         mStartFrameIndex.at(i) = paramsStartFrameIndex[i];
      }
   }
   free(paramsStartFrameIndex);
}

void InputLayer::ioParam_skip_frame_index(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "batchMethod"));
   if (mBatchMethod != BatchIndexer::BYSPECIFIED) {
      mSkipFrameIndex.resize(parent->getNBatchGlobal(), 0);
      // Earlier behavior made it a fatal error if skip_frame_index was used
      // and batchMethod was not bySpecified. Now the parameter is skipped and
      // a warning will be issued when params are scanned for unread values.
      return;
   }
   int *paramsSkipFrameIndex = nullptr;
   int length                = 0;
   if (ioFlag == PARAMS_IO_WRITE) {
      length               = mSkipFrameIndex.size();
      paramsSkipFrameIndex = static_cast<int *>(calloc(length, sizeof(int)));
      for (int i = 0; i < length; ++i) {
         paramsSkipFrameIndex[i] = mSkipFrameIndex.at(i);
      }
   }
   this->getParent()->parameters()->ioParamArray(
         ioFlag, this->getName(), "skip_frame_index", &paramsSkipFrameIndex, &length);
   FatalIf(
         length != parent->getNBatchGlobal(),
         "%s: skip_frame_index requires nbatch values.\n",
         getName());
   mSkipFrameIndex.clear();
   mSkipFrameIndex.resize(length < parent->getNBatchGlobal() ? parent->getNBatchGlobal() : length);
   if (length > 0) {
      for (int i = 0; i < length; ++i) {
         mSkipFrameIndex.at(i) = paramsSkipFrameIndex[i];
      }
   }
   free(paramsSkipFrameIndex);
}

void InputLayer::ioParam_writeFrameToTimestamp(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "displayPeriod"));
   if (mDisplayPeriod > 0) {
      parent->parameters()->ioParamValue(
            ioFlag, name, "writeFrameToTimestamp", &mWriteFrameToTimestamp, mWriteFrameToTimestamp);
   }
   else {
      mWriteFrameToTimestamp = false;
   }
}

void InputLayer::ioParam_resetToStartOnLoop(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "resetToStartOnLoop", &mResetToStartOnLoop, mResetToStartOnLoop);
}

BaseInputDeprecatedError::BaseInputDeprecatedError(const char *name, HyPerCol *hc) {
   Fatal() << "Movie, Image, MoviePvp, and ImagePvp are deprecated.\n"
           << "Use ImageLayer or PvpLayer instead. These replacements\n"
           << "accept the same parameters with the following changes:\n"
           << "  - ImageLayer assumes any inputPath ending in .txt is\n"
           << "    a Movie, and behaves accordingly. Set displayPeriod\n"
           << "    to 0 to display a single image within a file list.\n"
           << "    If inputPath ends in .png, .jpg, or .bmp, the layer\n"
           << "    displays a single Image.\n"
           << "  - PvpLayer no longer has the parameter pvpFrameIndex.\n"
           << "    Instead, use start_frame_index to specify which\n"
           << "    index to display. If displayPeriod != 0, PvpLayer\n"
           << "    behaves like a MoviePvp instead of an ImagePvp.\n"
           << "  - Jitter has been removed. Parameters related to it\n"
           << "    will be ignored.\n"
           << "  - useImageBCFlag is now useInputBCFlag.\n"
           << "  - batchMethod now expects byFile or byList instead of\n"
           << "    byImage or byMovie. bySpecified has not changed.\n"
           << "  - FilenameParsingGroundTruthLayer now acceps a param\n"
           << "    called inputLayerName instead of movieLayerName.\n";
}

} // end namespace PV
