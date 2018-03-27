/*
 * InputLayer.cpp
 * Formerly InputLayer.cpp
 */

#include "InputLayer.hpp"
#include "columns/RandomSeed.hpp"
#include "utils/BufferUtilsMPI.hpp"

#include <cfloat>

namespace PV {

InputLayer::InputLayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

InputLayer::~InputLayer() {
   delete mBorderExchanger;
   delete mTimestampStream;
}

int InputLayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

Response::Status InputLayer::allocateDataStructures() {
   auto status = HyPerLayer::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   if (mNeedInputRegionsPointer) {
      mInputRegionsAllBatchElements.resize(getNumExtendedAllBatches());
   }
   return Response::SUCCESS;
}

void InputLayer::initializeBatchIndexer() {
   // TODO: move check of size of mStartFrameIndex and mSkipFrameIndex here.
   pvAssert(getMPIBlock());
   pvAssert(getMPIBlock()->getRank() == 0);
   int localBatchCount  = getLayerLoc()->nbatch;
   int mpiBatchCount    = getMPIBlock()->getBatchDimension();
   int mpiGlobalCount   = getMPIBlock()->getGlobalBatchDimension();
   int globalBatchCount = localBatchCount * mpiGlobalCount;
   int batchOffset      = localBatchCount * getMPIBlock()->getStartBatch();
   int blockBatchCount  = localBatchCount * getMPIBlock()->getBatchDimension();
   int fileCount        = countInputImages();
   mBatchIndexer        = std::unique_ptr<BatchIndexer>(
         new BatchIndexer(
               std::string(name),
               globalBatchCount,
               batchOffset,
               blockBatchCount,
               fileCount,
               mBatchMethod,
               initializeFromCheckpointFlag));
   for (int b = 0; b < blockBatchCount; ++b) {
      mBatchIndexer->specifyBatching(
            b, mStartFrameIndex.at(batchOffset + b), mSkipFrameIndex.at(batchOffset + b));
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

Response::Status InputLayer::updateState(double time, double dt) {
   if (readyForNextFile()) {

      // Write file path to timestamp file
      if (mTimestampStream) {
         std::ostringstream outStrStream;
         outStrStream.precision(15);
         int kb0             = getLayerLoc()->kb0;
         int blockBatchCount = getLayerLoc()->nbatch * getMPIBlock()->getBatchDimension();
         for (int b = 0; b < blockBatchCount; ++b) {
            int index = mBatchIndexer->getIndex(b);
            outStrStream << "[" << getName() << "] time: " << time << ", batch element: " << b + kb0
                         << ", index: " << mBatchIndexer->getIndex(b) << ","
                         << describeInput(mBatchIndexer->getIndex(b)) << "\n";
         }
         size_t len = outStrStream.str().length();
         mTimestampStream->write(outStrStream.str().c_str(), len);
         mTimestampStream->flush();
      }

      // Read in the next file
      retrieveInputAndAdvanceIndex(time, dt);
   }
   return Response::SUCCESS;
}

void InputLayer::retrieveInput(double timef, double dt) {
   if (getMPIBlock()->getRank() == 0) {
      int displayPeriodIndex = std::floor(timef / (mDisplayPeriod * dt));
      if (displayPeriodIndex % mJitterChangeInterval == 0) {
         for (int b = 0; b < mRandomShiftX.size(); b++) {
            mRandomShiftX[b] = -mMaxShiftX + (mRNG() % (2 * mMaxShiftX + 1));
            mRandomShiftY[b] = -mMaxShiftY + (mRNG() % (2 * mMaxShiftY + 1));
            if (mXFlipEnabled) {
               mMirrorFlipX[b] = mXFlipToggle ? !mMirrorFlipX[b] : (mRNG() % 100) > 50;
            }
            if (mYFlipEnabled) {
               mMirrorFlipY[b] = mYFlipToggle ? !mMirrorFlipY[b] : (mRNG() % 100) > 50;
            }
         }
      }
   }

   int localNBatch = getLayerLoc()->nbatch;
   for (int m = 0; m < getMPIBlock()->getBatchDimension(); m++) {
      for (int b = 0; b < localNBatch; b++) {
         if (getMPIBlock()->getRank() == 0) {
            int blockBatchElement = b + localNBatch * m;
            int inputIndex        = mBatchIndexer->getIndex(blockBatchElement);
            mInputData.at(b)      = retrieveData(inputIndex);
            int width             = mInputData.at(b).getWidth();
            int height            = mInputData.at(b).getHeight();
            int features          = mInputData.at(b).getFeatures();
            mInputRegion.at(b)    = Buffer<float>(width, height, features);
            int const N           = mInputRegion.at(b).getTotalElements();
            for (int k = 0; k < N; k++) {
               mInputRegion.at(b).set(k, 1.0f);
            }
            fitBufferToGlobalLayer(mInputData.at(b), blockBatchElement);
            fitBufferToGlobalLayer(mInputRegion.at(b), blockBatchElement);
            // Now dataBuffer has input over the global layer. Apply normalizeLuminanceFlag, etc.
            normalizePixels(b);
            // Finally, crop to the part of the image covered by the MPIBlock.
            cropToMPIBlock(mInputData.at(b));
            cropToMPIBlock(mInputData.at(b));
         }
         // Each MPIBlock sends the local portions.
         scatterInput(b, m);
      }
   }
}

// Note: we call retrieveInput and then nextIndex because we update on the
// first timestep (even though we initialized in initializeActivity).
// If we could skip the update on the first timestep, we could call
// nextIndex first, and then call retrieveInput, which seems more natural.
void InputLayer::retrieveInputAndAdvanceIndex(double timef, double dt) {
   retrieveInput(timef, dt);
   if (mBatchIndexer) {
      int blockBatchCount = getLayerLoc()->nbatch * getMPIBlock()->getBatchDimension();
      for (int b = 0; b < blockBatchCount; b++) {
         mBatchIndexer->nextIndex(b);
      }
   }
}

int InputLayer::scatterInput(int localBatchIndex, int mpiBatchIndex) {
   int const procBatchIndex = getMPIBlock()->getBatchIndex();
   if (procBatchIndex != 0 and procBatchIndex != mpiBatchIndex) {
      return PV_SUCCESS;
   }
   PVLayerLoc const *loc = getLayerLoc();
   PVHalo const *halo    = &loc->halo;
   int activityWidth, activityHeight, activityLeft, activityTop;
   if (mUseInputBCflag) {
      activityWidth  = loc->nx + halo->lt + halo->rt;
      activityHeight = loc->ny + halo->up + halo->dn;
      activityLeft   = 0;
      activityTop    = 0;
   }
   else {
      activityWidth  = loc->nx;
      activityHeight = loc->ny;
      activityLeft   = halo->lt;
      activityTop    = halo->up;
   }
   Buffer<float> dataBuffer;
   Buffer<float> regionBuffer;

   if (getMPIBlock()->getRank() == 0) {
      dataBuffer   = mInputData.at(localBatchIndex);
      regionBuffer = mInputRegion.at(localBatchIndex);
   }
   else {
      dataBuffer.resize(activityWidth, activityHeight, loc->nf);
      regionBuffer.resize(activityWidth, activityHeight, loc->nf);
   }
   BufferUtils::scatter<float>(getMPIBlock(), dataBuffer, loc->nx, loc->ny, mpiBatchIndex, 0);
   BufferUtils::scatter<float>(getMPIBlock(), regionBuffer, loc->nx, loc->ny, mpiBatchIndex, 0);
   if (procBatchIndex != mpiBatchIndex) {
      return PV_SUCCESS;
   }

   // All processes that make it to this point have the indicated MPI batch index,
   // and dataBuffer has the correct data for the indicated batch index.
   // Clear the current activity for this batch element; then copy the input data over row by row.
   float *activityBuffer = &getActivity()[localBatchIndex * getNumExtended()];
   for (int n = 0; n < getNumExtended(); ++n) {
      activityBuffer[n] = mPadValue;
   }

   for (int y = 0; y < activityHeight; ++y) {
      for (int x = 0; x < activityWidth; ++x) {
         for (int f = 0; f < numFeatures; ++f) {
            int activityIndex = kIndex(
                  activityLeft + x,
                  activityTop + y,
                  f,
                  loc->nx + halo->lt + halo->rt,
                  loc->ny + halo->up + halo->dn,
                  numFeatures);
            if (regionBuffer.at(x, y, f) > 0.0f) {
               activityBuffer[activityIndex] = dataBuffer.at(x, y, f);
            }
         }
      }
   }
   if (mNeedInputRegionsPointer) {
      float *inputRegionBuffer =
            &getInputRegionsAllBatchElements()[localBatchIndex * getNumExtended()];
      for (int y = 0; y < activityHeight; ++y) {
         for (int x = 0; x < activityWidth; ++x) {
            for (int f = 0; f < numFeatures; ++f) {
               int activityIndex = kIndex(
                     activityLeft + x,
                     activityTop + y,
                     f,
                     loc->nx + halo->lt + halo->rt,
                     loc->ny + halo->up + halo->dn,
                     numFeatures);
               if (regionBuffer.at(x, y, f) > 0.0f) {
                  inputRegionBuffer[activityIndex] = regionBuffer.at(x, y, f);
               }
            }
         }
      }
   }

   return PV_SUCCESS;
}

void InputLayer::fitBufferToGlobalLayer(Buffer<float> &buffer, int blockBatchElement) {
   pvAssert(getMPIBlock()->getRank() == 0);
   const PVLayerLoc *loc  = getLayerLoc();
   int const xMargins     = mUseInputBCflag ? loc->halo.lt + loc->halo.rt : 0;
   int const yMargins     = mUseInputBCflag ? loc->halo.dn + loc->halo.up : 0;
   const int targetWidth  = loc->nxGlobal + xMargins;
   const int targetHeight = loc->nyGlobal + yMargins;

   FatalIf(
         buffer.getFeatures() != loc->nf,
         "ERROR: Input for layer %s has %d features, but layer has %d.\n",
         getName(),
         buffer.getFeatures(),
         loc->nf);

   if (mAutoResizeFlag) {
      BufferUtils::rescale(
            buffer, targetWidth, targetHeight, mRescaleMethod, mInterpolationMethod, mAnchor);
      buffer.translate(
            -mOffsetX + mRandomShiftX[blockBatchElement],
            -mOffsetY + mRandomShiftY[blockBatchElement]);
   }
   else {
      buffer.grow(targetWidth, targetHeight, mAnchor);
      buffer.translate(
            -mOffsetX + mRandomShiftX[blockBatchElement],
            -mOffsetY + mRandomShiftY[blockBatchElement]);
      buffer.crop(targetWidth, targetHeight, mAnchor);
   }

   if (mMirrorFlipX[blockBatchElement] || mMirrorFlipY[blockBatchElement]) {
      buffer.flip(mMirrorFlipX[blockBatchElement], mMirrorFlipY[blockBatchElement]);
   }
}

void InputLayer::normalizePixels(int batchElement) {
   Buffer<float> &dataBuffer         = mInputData.at(batchElement);
   Buffer<float> const &regionBuffer = mInputRegion.at(batchElement);
   int const totalElements           = dataBuffer.getTotalElements();
   pvAssert(totalElements == regionBuffer.getTotalElements());
   int validRegionCount = 0;
   for (int k = 0; k < totalElements; k++) {
      if (regionBuffer.at(k) > 0.0f) {
         validRegionCount++;
      }
   }
   if (validRegionCount == 0) {
      return;
   }
   if (mNormalizeLuminanceFlag) {
      if (mNormalizeStdDev) {
         float imageSum   = 0.0f;
         float imageSumSq = 0.0f;
         for (int k = 0; k < totalElements; k++) {
            if (regionBuffer.at(k) > 0.0f) {
               float const v = dataBuffer.at(k);
               imageSum += v;
               imageSumSq += v * v;
            }
         }

         // set mean to zero
         float imageAverage = imageSum / validRegionCount;
         for (int k = 0; k < totalElements; k++) {
            if (regionBuffer.at(k) > 0.0f) {
               float const v = dataBuffer.at(k);
               dataBuffer.set(k, v - imageAverage);
            }
         }

         // set std dev to 1
         float imageVariance = imageSumSq / validRegionCount - imageAverage * imageAverage;
         pvAssert(imageVariance >= 0);
         if (imageVariance > 0) {
            float imageStdDev = std::sqrt(imageVariance);
            for (int k = 0; k < totalElements; k++) {
               if (regionBuffer.at(k) > 0.0f) {
                  float const v = dataBuffer.at(k) / imageStdDev;
                  dataBuffer.set(k, v);
               }
            }
         }
         else {
            // Image is flat; set to identically zero.
            // This may not be necessary since we subtracted the mean,
            // but maybe there could be roundoff issues?
            for (int k = 0; k < totalElements; k++) {
               if (regionBuffer.at(k) > 0.0f) {
                  dataBuffer.set(k, 0.0f);
               }
            }
         }
      }
      else { // mNormalizeStdDev is false; normalize so max is one and min is zero.
         float imageMax = -std::numeric_limits<float>::max();
         float imageMin = std::numeric_limits<float>::max();
         for (int k = 0; k < totalElements; k++) {
            if (regionBuffer.at(k) > 0.0f) {
               float const v = dataBuffer.at(k);
               imageMax      = v > imageMax ? v : imageMax;
               imageMin      = v < imageMin ? v : imageMin;
            }
         }
         if (imageMax > imageMin) {
            float imageStretch = 1.0f / (imageMax - imageMin);
            for (int k = 0; k < totalElements; k++) {
               if (regionBuffer.at(k) > 0.0f) {
                  float const v = (dataBuffer.at(k) - imageMin) * imageStretch;
                  dataBuffer.set(k, v);
               }
            }
         }
         else {
            for (int k = 0; k < totalElements; k++) {
               if (regionBuffer.at(k) > 0.0f) {
                  dataBuffer.set(k, 0.0f);
               }
            }
         }
      }
   }
   if (mInverseFlag) {
      if (mNormalizeLuminanceFlag) {
         for (int k = 0; k < totalElements; k++) {
            if (regionBuffer.at(k) > 0.0f) {
               float const v = -dataBuffer.at(k);
               dataBuffer.set(k, v);
            }
         }
      }
      else {
         float imageMax = -std::numeric_limits<float>::max();
         float imageMin = std::numeric_limits<float>::max();
         for (int k = 0; k < totalElements; k++) {
            if (regionBuffer.at(k) > 0.0f) {
               float const v = dataBuffer.at(k);
               imageMax      = v > imageMax ? v : imageMax;
               imageMin      = v < imageMin ? v : imageMin;
            }
         }
         for (int k = 0; k < totalElements; k++) {
            if (regionBuffer.at(k) > 0.0f) {
               float const v = imageMax + imageMin - dataBuffer.at(k);
               dataBuffer.set(k, v);
            }
         }
      }
   }
}

void InputLayer::cropToMPIBlock(Buffer<float> &buffer) {
   const PVLayerLoc *loc = getLayerLoc();
   int const startX      = getMPIBlock()->getStartColumn() * loc->nx;
   int const startY      = getMPIBlock()->getStartRow() * loc->ny;
   buffer.translate(-startX, -startY);
   int const xMargins    = mUseInputBCflag ? loc->halo.lt + loc->halo.rt : 0;
   int const yMargins    = mUseInputBCflag ? loc->halo.dn + loc->halo.up : 0;
   int const blockWidth  = getMPIBlock()->getNumColumns() * loc->nx + xMargins;
   int const blockHeight = getMPIBlock()->getNumRows() * loc->ny + yMargins;
   buffer.crop(blockWidth, blockHeight, Buffer<float>::NORTHWEST);
}

double InputLayer::getDeltaUpdateTime() { return mDisplayPeriod > 0 ? mDisplayPeriod : DBL_MAX; }

int InputLayer::requireChannel(int channelNeeded, int *numChannelsResult) {
   if (parent->getCommunicator()->commRank() == 0) {
      ErrorLog().printf("%s cannot be a post-synaptic layer.\n", getDescription_c());
   }
   *numChannelsResult = 0;
   return PV_FAILURE;
}

void InputLayer::allocateV() { clayer->V = nullptr; }

void InputLayer::initializeV() { pvAssert(getV() == nullptr); }

void InputLayer::initializeActivity() {
   retrieveInput(parent->simulationTime(), parent->getDeltaTime());
}

int InputLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLayer::ioParamsFillGroup(ioFlag);
   ioParam_displayPeriod(ioFlag);
   ioParam_inputPath(ioFlag);
   ioParam_offsetAnchor(ioFlag);
   ioParam_offsets(ioFlag);
   ioParam_maxShifts(ioFlag);
   ioParam_flipsEnabled(ioFlag);
   ioParam_flipsToggle(ioFlag);
   ioParam_jitterChangeInterval(ioFlag);
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
   ioParam_resetToStartOnLoop(ioFlag);
   ioParam_writeFrameToTimestamp(ioFlag);
   return status;
}

Response::Status InputLayer::registerData(Checkpointer *checkpointer) {
   auto status = HyPerLayer::registerData(checkpointer);
   if (!Response::completed(status)) {
      return status;
   }
   if (checkpointer->getMPIBlock()->getRank() == 0) {
      mRNG.seed(mRandomSeed);
      int numBatch = getLayerLoc()->nbatch;
      int nBatch   = getMPIBlock()->getBatchDimension() * numBatch;
      mRandomShiftX.resize(nBatch);
      mRandomShiftY.resize(nBatch);
      mMirrorFlipX.resize(nBatch);
      mMirrorFlipY.resize(nBatch);
      mInputData.resize(numBatch);
      mInputRegion.resize(numBatch);
      initializeBatchIndexer();
      mBatchIndexer->setWrapToStartIndex(mResetToStartOnLoop);
      mBatchIndexer->registerData(checkpointer);

      if (mWriteFrameToTimestamp) {
         std::string timestampFilename = std::string("timestamps/");
         timestampFilename += name + std::string(".txt");
         std::string cpFileStreamLabel(getName());
         cpFileStreamLabel.append("_TimestampState");
         bool needToCreateFile = checkpointer->getCheckpointReadDirectory().empty();
         mTimestampStream      = new CheckpointableFileStream(
               timestampFilename, needToCreateFile, checkpointer, cpFileStreamLabel);
      }
   }
   return Response::SUCCESS;
}

Response::Status InputLayer::readStateFromCheckpoint(Checkpointer *checkpointer) {
   auto status = Response::NO_ACTION;
   if (initializeFromCheckpointFlag) {
      status = HyPerLayer::readStateFromCheckpoint(checkpointer);
      if (!Response::completed(status)) {
         return status;
      }
      if (mBatchIndexer) {
         pvAssert(getMPIBlock()->getRank() == 0);
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

int InputLayer::ioParam_maxShifts(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "maxShiftX", &mMaxShiftX, mMaxShiftX);
   parent->parameters()->ioParamValue(ioFlag, name, "maxShiftY", &mMaxShiftY, mMaxShiftY);
   return PV_SUCCESS;
}

int InputLayer::ioParam_flipsEnabled(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "xFlipEnabled", &mXFlipEnabled, mXFlipEnabled);
   parent->parameters()->ioParamValue(ioFlag, name, "yFlipEnabled", &mYFlipEnabled, mYFlipEnabled);
   return PV_SUCCESS;
}

int InputLayer::ioParam_flipsToggle(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "xFlipToggle", &mXFlipToggle, mXFlipToggle);
   parent->parameters()->ioParamValue(ioFlag, name, "yFlipToggle", &mYFlipToggle, mYFlipToggle);
   return PV_SUCCESS;
}

int InputLayer::ioParam_jitterChangeInterval(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "jitterChangeInterval", &mJitterChangeInterval, mJitterChangeInterval);
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
         if (parent->getCommunicator()->commRank() == 0) {
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
         if (parent->getCommunicator()->commRank() == 0) {
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
            if (parent->getCommunicator()->commRank() == 0) {
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
   parent->parameters()->ioParamValue(ioFlag, name, "randomSeed", &mRandomSeed, mRandomSeed);
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
   this->parent->parameters()->ioParamArray(
         ioFlag, this->getName(), "start_frame_index", &paramsStartFrameIndex, &length);
   FatalIf(
         length != 0 && length != parent->getNBatchGlobal(),
         "%s: start_frame_index requires either 0 or nbatch values.\n",
         getName());
   mStartFrameIndex.clear();
   mStartFrameIndex.resize(parent->getNBatchGlobal());
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
   this->parent->parameters()->ioParamArray(
         ioFlag, this->getName(), "skip_frame_index", &paramsSkipFrameIndex, &length);
   FatalIf(
         length != parent->getNBatchGlobal(),
         "%s: skip_frame_index requires nbatch values.\n",
         getName());
   mSkipFrameIndex.clear();
   mSkipFrameIndex.resize(length);
   for (int i = 0; i < length; ++i) {
      mSkipFrameIndex.at(i) = paramsSkipFrameIndex[i];
   }
   free(paramsSkipFrameIndex);
}

void InputLayer::ioParam_resetToStartOnLoop(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "batchMethod"));
   if (mBatchMethod == BatchIndexer::BYSPECIFIED) {
      parent->parameters()->ioParamValue(
            ioFlag, name, "resetToStartOnLoop", &mResetToStartOnLoop, mResetToStartOnLoop);
   }
   else {
      mResetToStartOnLoop = false;
   }
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

} // end namespace PV
