/*
 * InputActivityBuffer.cpp
 *
 *  Created on: Jul 22, 2015
 *      Author: Sheng Lundquist
 */

#include "InputActivityBuffer.hpp"
#include "arch/mpi/mpi.h"
#include "cMakeHeader.h"
#include "checkpointing/CheckpointEntryFilePosition.hpp"
#include "columns/RandomSeed.hpp"
#include "structures/PVLayerLoc.hpp"
#include "io/FileStreamBuilder.hpp"
#include "structures/MPIBlock.hpp"
#include "utils/BufferUtilsMPI.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include "utils/conversions.hpp"
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>

namespace PV {

InputActivityBuffer::InputActivityBuffer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

InputActivityBuffer::~InputActivityBuffer() {}

void InputActivityBuffer::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   ActivityBuffer::initialize(paramsIO, comm);
}

void InputActivityBuffer::setObjectType() { mObjectType = "InputActivityBuffer"; }

int InputActivityBuffer::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = ActivityBuffer::ioParamsFillGroup(ioSwitch);
   ioParam_displayPeriod(ioSwitch);
   ioParam_inputPath(ioSwitch);
   ioParam_offsetAnchor(ioSwitch);
   ioParam_offsets(ioSwitch);
   ioParam_jitterChangeInterval(ioSwitch);
   ioParam_jitterChangeIntervalUnit(ioSwitch);
   ioParam_maxShifts(ioSwitch);
   ioParam_flipsEnabled(ioSwitch);
   ioParam_flipsToggle(ioSwitch);
   ioParam_autoResizeFlag(ioSwitch);
   ioParam_aspectRatioAdjustment(ioSwitch);
   ioParam_interpolationMethod(ioSwitch);
   ioParam_inverseFlag(ioSwitch);
   ioParam_normalizeLuminanceFlag(ioSwitch);
   ioParam_normalizeStdDev(ioSwitch);
   ioParam_useInputBCflag(ioSwitch);
   ioParam_padValue(ioSwitch);
   ioParam_batchMethod(ioSwitch);
   ioParam_randomSeed(ioSwitch);
   ioParam_start_frame_index(ioSwitch);
   ioParam_skip_frame_index(ioSwitch);
   ioParam_resetToStartOnLoop(ioSwitch);
   ioParam_writeFrameToTimestamp(ioSwitch);
   return status;
}

void InputActivityBuffer::ioParam_displayPeriod(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "displayPeriod", &mDisplayPeriod);
}

void InputActivityBuffer::ioParam_inputPath(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "inputPath", &mInputPath);
   if (ioSwitch == ParamsIOSwitch::Read) {
      FatalIf(
            mInputPath.empty(),
            "Input layer \"%s\" parameter inputPath cannot be NULL or empty.\n",
            getName());
   }
}

void InputActivityBuffer::ioParam_useInputBCflag(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "useInputBCflag", &mUseInputBCflag);
}

void InputActivityBuffer::ioParam_offsets(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "offsetX", &mOffsetX);
   mParamsIO->ioParam(ioSwitch, "offsetY", &mOffsetY);
}

void InputActivityBuffer::ioParam_jitterChangeInterval(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "jitterChangeInterval", &mJitterChangeInterval);
}

void InputActivityBuffer::ioParam_jitterChangeIntervalUnit(ParamsIOSwitch ioSwitch) {
   assert(!mParamsIO->presentAndNotBeenRead("jitterChangeInterval"));
   if (mJitterChangeInterval > 0) {
      mParamsIO->ioParam(ioSwitch, "jitterChangeIntervalUnit", &mJitterChangeIntervalUnit);
      if (ioSwitch == ParamsIOSwitch::Read) {
         for (char &c : mJitterChangeIntervalUnit) {
            c = (char)std::tolower((int)c);
         }
         mJitterChangeIntervalInTimesteps = mJitterChangeInterval;
         if (mJitterChangeIntervalUnit == "displayperiod") {
            mJitterChangeIntervalUnit = "displayPeriod";
            mJitterChangeIntervalInTimesteps *= mDisplayPeriod;
         }
         FatalIf(
               mJitterChangeIntervalUnit != "displayPeriod" and
                     mJitterChangeIntervalUnit != "timestep",
               "\"%s\" jitterChangeIntervalUnit must be either \"displayPeriod\" or \"timestep\"\n",
               getName());
      }
   }
}

void InputActivityBuffer::ioParam_maxShifts(ParamsIOSwitch ioSwitch) {
   assert(!mParamsIO->presentAndNotBeenRead("jitterChangeInterval"));
   if (mJitterChangeInterval > 0) {
      mParamsIO->ioParam(ioSwitch, "maxShiftX", &mMaxShiftX);
      mParamsIO->ioParam(ioSwitch, "maxShiftY", &mMaxShiftY);
   }
}

void InputActivityBuffer::ioParam_flipsEnabled(ParamsIOSwitch ioSwitch) {
   assert(!mParamsIO->presentAndNotBeenRead("jitterChangeInterval"));
   if (mJitterChangeInterval > 0) {
      mParamsIO->ioParam(ioSwitch, "xFlipEnabled", &mXFlipEnabled);
      mParamsIO->ioParam(ioSwitch, "yFlipEnabled", &mYFlipEnabled);
   }
}

void InputActivityBuffer::ioParam_flipsToggle(ParamsIOSwitch ioSwitch) {
   assert(!mParamsIO->presentAndNotBeenRead("jitterChangeInterval"));
   if (mJitterChangeInterval > 0) {
      mParamsIO->ioParam(ioSwitch, "xFlipToggle", &mXFlipToggle);
      mParamsIO->ioParam(ioSwitch, "yFlipToggle", &mYFlipToggle);
   }
}

void InputActivityBuffer::ioParam_randomSeed(ParamsIOSwitch ioSwitch) {
   pvAssert(!mParamsIO->presentAndNotBeenRead("jitterChangeInterval"));
   if (mJitterChangeInterval > 0) {
      mParamsIO->ioParam(ioSwitch, "randomSeed", &mRandomSeed);
   }
}

void InputActivityBuffer::ioParam_offsetAnchor(ParamsIOSwitch ioSwitch) {
   std::string offsetAnchor;
   if (ioSwitch == ParamsIOSwitch::Read) {
      mParamsIO->ioParam(ioSwitch, "offsetAnchor", &offsetAnchor);
      if (offsetAnchor.size() != 2UL) {
         badOffsetAnchorString(offsetAnchor);
      }
      offsetAnchor[0] = (char)std::tolower((int)offsetAnchor[0]);
      offsetAnchor[1] = (char)std::tolower((int)offsetAnchor[1]);

      if (offsetAnchor == "tl" or offsetAnchor == "lt") {
         mAnchor = Buffer<float>::NORTHWEST;
      }
      else if (offsetAnchor == "tc" or offsetAnchor == "ct") {
         mAnchor = Buffer<float>::NORTH;
      }
      else if (offsetAnchor == "tr" or offsetAnchor == "rt") {
         mAnchor = Buffer<float>::NORTHEAST;
      }
      else if (offsetAnchor == "cl" or offsetAnchor == "lc") {
         mAnchor = Buffer<float>::WEST;
      }
      else if (offsetAnchor == "cc") {
         mAnchor = Buffer<float>::CENTER;
      }
      else if (offsetAnchor == "cr" or offsetAnchor == "rc") {
         mAnchor = Buffer<float>::EAST;
      }
      else if (offsetAnchor == "bl" or offsetAnchor == "lb") {
         mAnchor = Buffer<float>::SOUTHWEST;
      }
      else if (offsetAnchor == "bc" or offsetAnchor == "cb") {
         mAnchor = Buffer<float>::SOUTH;
      }
      else if (offsetAnchor == "br" or offsetAnchor == "rb") {
         mAnchor = Buffer<float>::SOUTHEAST;
      }
      else {
         badOffsetAnchorString(offsetAnchor);
      }
   }
   else { // Writing
      switch (mAnchor) {
         case Buffer<float>::CENTER: offsetAnchor = "cc"; break;
         case Buffer<float>::NORTH: offsetAnchor = "tc"; break;
         case Buffer<float>::NORTHEAST: offsetAnchor = "tr"; break;
         case Buffer<float>::EAST: offsetAnchor = "cr"; break;
         case Buffer<float>::SOUTHEAST: offsetAnchor = "br"; break;
         case Buffer<float>::SOUTH: offsetAnchor = "bc"; break;
         case Buffer<float>::SOUTHWEST: offsetAnchor = "bl"; break;
         case Buffer<float>::WEST: offsetAnchor = "cl"; break;
         case Buffer<float>::NORTHWEST: offsetAnchor = "tl"; break;
      }
      mParamsIO->ioParam(ioSwitch, "offsetAnchor", &offsetAnchor);
   }
}

void InputActivityBuffer::badOffsetAnchorString(std::string const &offsetAnchor) {
   Fatal().printf(
         "%s: offsetAnchor \"%s\" is not recognized. "
         "The offsetAnchor parameter must be a two-letter string. "
         "One character must be \"t\", \"c\", or \"b\" (for top, center or bottom); "
         "and the other character must be \"l\", \"c\", or \"r\" (for left, center or right).\n",
         getDescription_c(),
         offsetAnchor.c_str());
}

void InputActivityBuffer::ioParam_autoResizeFlag(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "autoResizeFlag", &mAutoResizeFlag);
}

void InputActivityBuffer::ioParam_aspectRatioAdjustment(ParamsIOSwitch ioSwitch) {
   assert(!mParamsIO->presentAndNotBeenRead("autoResizeFlag"));
   if (mAutoResizeFlag) {
      std::string aspectRatioAdjustment;
      if (ioSwitch == ParamsIOSwitch::Write) {
         switch (mRescaleMethod) {
            case BufferUtils::CROP: aspectRatioAdjustment = "crop"; break;
            case BufferUtils::PAD: aspectRatioAdjustment = "pad"; break;
            default: Fatal().printf("Unrecognized RescaleMethod %d\n", mRescaleMethod); break;
         }
      }
      mParamsIO->ioParam(ioSwitch, "aspectRatioAdjustment", &aspectRatioAdjustment);
      if (ioSwitch == ParamsIOSwitch::Read) {
         for (char &c : aspectRatioAdjustment) {
            c = tolower(c);
         }
         if (aspectRatioAdjustment == "crop") {
            mRescaleMethod = BufferUtils::CROP;
         }
         else if (aspectRatioAdjustment == "pad") {
            mRescaleMethod = BufferUtils::PAD;
         }
         else {
            if (mCommunicator->commRank() == 0) {
               ErrorLog().printf(
                     "%s: aspectRatioAdjustment must be either \"crop\" or \"pad\".\n",
                     getDescription_c());
            }
            MPI_Barrier(mCommunicator->communicator());
            std::exit(EXIT_FAILURE);
         }
      }
   }
}

void InputActivityBuffer::ioParam_interpolationMethod(ParamsIOSwitch ioSwitch) {
   pvAssert(!mParamsIO->presentAndNotBeenRead("autoResizeFlag"));
   if (mAutoResizeFlag) {
      std::string interpolationMethod;
      if (ioSwitch == ParamsIOSwitch::Write) {
         switch (mInterpolationMethod) {
            case BufferUtils::BICUBIC: interpolationMethod = "bicubic"; break;
            case BufferUtils::NEAREST: interpolationMethod = "nearestNeighbor"; break;
            default:
                  Fatal().printf("Unrecognized InterpolationMethod %d\n", mInterpolationMethod);
                  break;
         }
      }
      mParamsIO->ioParam(ioSwitch, "interpolationMethod", &interpolationMethod);
      if (ioSwitch == ParamsIOSwitch::Read) {
         for (char &c : interpolationMethod) {
            c = tolower(c);
         }
         if (interpolationMethod== "bicubic") {
            mInterpolationMethod = BufferUtils::BICUBIC;
         }
         else if (interpolationMethod == "nearestneighbor") {
            mInterpolationMethod = BufferUtils::NEAREST;
         }
         else {
            if (mCommunicator->commRank() == 0) {
               ErrorLog().printf(
                     "%s: interpolationMethod must be either \"bicubic\" or \"nearestNeighbor\".\n",
                     getDescription_c());
            }
            MPI_Barrier(mCommunicator->communicator());
            std::exit(EXIT_FAILURE);
         }
      }
   }
}

void InputActivityBuffer::ioParam_inverseFlag(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "inverseFlag", &mInverseFlag);
}

void InputActivityBuffer::ioParam_normalizeLuminanceFlag(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "normalizeLuminanceFlag", &mNormalizeLuminanceFlag);
}

void InputActivityBuffer::ioParam_normalizeStdDev(ParamsIOSwitch ioSwitch) {
   assert(!mParamsIO->presentAndNotBeenRead("normalizeLuminanceFlag"));
   if (mNormalizeLuminanceFlag) {
      mParamsIO->ioParam(ioSwitch, "normalizeStdDev", &mNormalizeStdDev);
   }
}
void InputActivityBuffer::ioParam_padValue(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "padValue", &mPadValue);
}

void InputActivityBuffer::ioParam_batchMethod(ParamsIOSwitch ioSwitch) {
   std::string batchMethodString;
   if (ioSwitch == ParamsIOSwitch::Write) {
      switch (mBatchMethod) {
         case BatchIndexer::BYFILE: batchMethodString = "byFile"; break;
         case BatchIndexer::BYLIST: batchMethodString = "byList"; break;
         case BatchIndexer::BYSPECIFIED: batchMethodString = "bySpecified"; break;
         case BatchIndexer::RANDOM: batchMethodString = "random"; break;
      }
   }
   mParamsIO->ioParam(ioSwitch, "batchMethod", &batchMethodString);
   if (batchMethodString == "byImage" || batchMethodString == "byFile") {
      mBatchMethod = BatchIndexer::BYFILE;
   }
   else if (batchMethodString == "byMovie" || batchMethodString == "byList") {
      mBatchMethod = BatchIndexer::BYLIST;
   }
   else if (batchMethodString == "bySpecified") {
      mBatchMethod = BatchIndexer::BYSPECIFIED;
   }
   else if (batchMethodString == "random") {
      mBatchMethod = BatchIndexer::RANDOM;
   }
   else {
      Fatal() << "Input layer " << getName() << " batchMethod not recognized. "
                 "Options are \"byFile\", \"byList\", bySpecified, and random.\n";
   }
}

void InputActivityBuffer::ioParam_start_frame_index(ParamsIOSwitch ioSwitch) {
   int length = 0;
   mParamsIO->ioParam(ioSwitch, "start_frame_index", &mStartFrameIndex);
}

void InputActivityBuffer::ioParam_skip_frame_index(ParamsIOSwitch ioSwitch) {
   pvAssert(!mParamsIO->presentAndNotBeenRead("batchMethod"));
   if (mBatchMethod != BatchIndexer::BYSPECIFIED) {
      // bySpecified is the only batchMethod that uses skip_frame_index.
      return;
   }
   mParamsIO->ioParam(ioSwitch, "skip_frame_index", &mSkipFrameIndex);
}

void InputActivityBuffer::ioParam_resetToStartOnLoop(ParamsIOSwitch ioSwitch) {
   assert(!mParamsIO->presentAndNotBeenRead("batchMethod"));
   if (mBatchMethod == BatchIndexer::BYSPECIFIED) {
      mParamsIO->ioParam(ioSwitch, "resetToStartOnLoop", &mResetToStartOnLoop);
   }
   else {
      mResetToStartOnLoop = false;
   }
}

void InputActivityBuffer::ioParam_writeFrameToTimestamp(ParamsIOSwitch ioSwitch) {
   assert(!mParamsIO->presentAndNotBeenRead("displayPeriod"));
   if (mDisplayPeriod > 0) {
      mParamsIO->ioParam(ioSwitch, "writeFrameToTimestamp", &mWriteFrameToTimestamp);
   }
   else {
      mWriteFrameToTimestamp = false;
   }
}

Response::Status InputActivityBuffer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = ActivityBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   bool sizeMismatch = false;
   int startLength   = (int)mStartFrameIndex.size();
   if (startLength == 0) {
      mStartFrameIndex.resize(message->mNBatchGlobal);
   }
   else if (startLength != message->mNBatchGlobal) {
      ErrorLog().printf(
            "%s: start_frame_index requires either 0 or nbatch=%d values.\n",
            getDescription_c(),
            message->mNBatchGlobal);
      sizeMismatch = true;
   }
   int skipLength = (int)mSkipFrameIndex.size();
   if (skipLength == 0) {
      mSkipFrameIndex.resize(message->mNBatchGlobal);
   }
   else if (skipLength != message->mNBatchGlobal) {
      ErrorLog().printf(
            "%s: start_frame_index requires either 0 or nbatch=%d values.\n",
            getDescription_c(),
            message->mNBatchGlobal);
      sizeMismatch = true;
   }
   FatalIf(sizeMismatch, "Unable to initialize %s\n", getDescription_c());

   return Response::SUCCESS;
}

void InputActivityBuffer::makeInputRegionsPointer(ActivityBuffer *activityBuffer) {
   mNeedInputRegionsPointer = true;
   mInputRegionTargets.emplace_back(activityBuffer);
}

Response::Status InputActivityBuffer::allocateDataStructures() {
   auto status = ActivityBuffer::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   if (mNeedInputRegionsPointer) {
      mInputRegionsAllBatchElements.resize(getBufferSizeAcrossBatch());
   }
   return Response::SUCCESS;
}

Response::Status InputActivityBuffer::registerData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = ActivityBuffer::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *checkpointer = message->mDataRegistry;
   if (mWriteFrameToTimestamp) {
      auto fileManager = getCommunicator()->getOutputFileManager();
      std::string timestampsDir("timestamps/");
      fileManager->ensureDirectoryExists(timestampsDir);
      std::string timestampPath = timestampsDir + getName() + ".txt";

      mTimestampStream = FileStreamBuilder(
                               fileManager,
                               timestampPath,
                               true /*text*/,
                               false /*not read-only*/,
                               checkpointer->getCheckpointReadDirectory()
                                     .empty() /*whether to clobber existing file*/,
                               checkpointer->doesVerifyWrites())
                               .get();
      auto checkpointEntry = std::make_shared<CheckpointEntryFilePosition>(
            getName(), std::string("TimestampState"), mTimestampStream);
      bool registerSucceeded = checkpointer->registerCheckpointEntry(
            checkpointEntry, false /*not constant for entire run*/);
      FatalIf(
            !registerSucceeded,
            "%s failed to register %s for checkpointing.\n",
            getDescription_c(),
            checkpointEntry->getName().c_str());
   }

   if (getCommunicator()->getIOMPIBlock()->getRank() == 0) {
      if (mJitterChangeIntervalInTimesteps > 0) {
         mRNG.seed(mRandomSeed);
      }
      int numBatch = getLayerLoc()->nbatch;
      int nBatch   = getCommunicator()->getIOMPIBlock()->getBatchDimension() * numBatch;
      mRandomShiftX.resize(nBatch);
      mRandomShiftY.resize(nBatch);
      mMirrorFlipX.resize(nBatch);
      mMirrorFlipY.resize(nBatch);
      mInputData.resize(numBatch);
      mInputRegion.resize(numBatch);
      initializeBatchIndexer();
      mBatchIndexer->setWrapToStartIndex(mResetToStartOnLoop);
      mBatchIndexer->registerData(message);
   }
   return Response::SUCCESS;
}

void InputActivityBuffer::initializeBatchIndexer() {
   // TODO: move check of size of mStartFrameIndex and mSkipFrameIndex here.
   auto ioMPIBlock = getCommunicator()->getIOMPIBlock();
   pvAssert(ioMPIBlock and ioMPIBlock->getRank() == 0);
   int localBatchCount  = getLayerLoc()->nbatch;
   int mpiGlobalCount   = ioMPIBlock->getGlobalBatchDimension();
   int globalBatchCount = localBatchCount * mpiGlobalCount;
   int batchOffset      = localBatchCount * ioMPIBlock->getStartBatch();
   int blockBatchCount  = localBatchCount * ioMPIBlock->getBatchDimension();
   int fileCount        = countInputImages();
   mBatchIndexer        = std::unique_ptr<BatchIndexer>(new BatchIndexer(
         std::string(getName()),
         globalBatchCount,
         batchOffset,
         blockBatchCount,
         fileCount,
         mBatchMethod,
         mInitializeFromCheckpointFlag));
   for (int b = 0; b < blockBatchCount; ++b) {
      mBatchIndexer->specifyBatching(
            b, mStartFrameIndex.at(batchOffset + b), mSkipFrameIndex.at(batchOffset + b));
      mBatchIndexer->initializeBatch(b);
   }
   mBatchIndexer->setRandomSeed(RandomSeed::instance()->getInitialSeed() + mRandomSeed);
}

std::string const &
InputActivityBuffer::getCurrentFilename(int localBatchIndex, int mpiBatchIndex) const {
   return mInputPath;
}

Response::Status
InputActivityBuffer::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   retrieveInput(0.0 /*simulationTime*/, message->mDeltaTime);
   return Response::SUCCESS;
}

void InputActivityBuffer::updateBufferCPU(double simTime, double deltaTime) {
   if (readyForNextFile(simTime, deltaTime)) {
      // Write file path to timestamp file
      writeToTimestampStream(simTime);

      // Read in the next file
      retrieveInputAndAdvanceIndex(simTime, deltaTime);
   }
}

void InputActivityBuffer::writeToTimestampStream(double simTime) {
   if (mTimestampStream) {
      std::ostringstream outStrStream;
      outStrStream.precision(15);
      PVLayerLoc const *loc = getLayerLoc();
      int kb0               = loc->kb0;
      int blockBatchCount   = loc->nbatch * getCommunicator()->getIOMPIBlock()->getBatchDimension();
      for (int b = 0; b < blockBatchCount; ++b) {
         int index = mBatchIndexer->getIndex(b);
         outStrStream << "[" << getName() << "] time: " << simTime << ", batch element: " << b + kb0
                      << ", index: " << index << "," << describeInput(index) << "\n";
      }
      size_t len = outStrStream.str().length();
      mTimestampStream->write(outStrStream.str().c_str(), len);
      mTimestampStream->flush();
   }
}

// Note: we call retrieveInput and then nextIndex because we update on the
// first timestep (even though we initialized in initializeActivity).
// If we could skip the update on the first timestep, we could call
// nextIndex first, and then call retrieveInput, which seems more natural.
void InputActivityBuffer::retrieveInputAndAdvanceIndex(double timef, double dt) {
   retrieveInput(timef, dt);
   if (mBatchIndexer) {
      int blockBatchDimension = getCommunicator()->getIOMPIBlock()->getBatchDimension();
      int blockBatchCount     = getLayerLoc()->nbatch * blockBatchDimension;
      for (int b = 0; b < blockBatchCount; b++) {
         mBatchIndexer->nextIndex(b);
      }
   }
}

// Virtual method used to spend multiple display periods on one file.
// Can be used to implement lists of collections or modifications to
// the loaded file, such as streaming audio or video.
bool InputActivityBuffer::readyForNextFile(double simTime, double deltaT) {
   // A display period <= 0 means we never change files
   return mDisplayPeriod > 0;
}

void InputActivityBuffer::retrieveInput(double simTime, double deltaTime) {
   auto ioMPIBlock = getCommunicator()->getIOMPIBlock();
   if (ioMPIBlock->getRank() == 0 and mJitterChangeIntervalInTimesteps > 0) {
      long timestep = std::nearbyint(simTime / deltaTime);
      if (timestep % mJitterChangeIntervalInTimesteps == 0) {
         for (std::size_t b = 0; b < mRandomShiftX.size(); b++) {
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
   for (int m = 0; m < ioMPIBlock->getBatchDimension(); m++) {
      for (int b = 0; b < localNBatch; b++) {
         if (ioMPIBlock->getRank() == 0) {
            int blockBatchElement = b + localNBatch * m;
            int inputIndex        = mBatchIndexer->getIndex(blockBatchElement);
            mInputData.at(b)      = retrieveData(inputIndex);
            int width             = mInputData.at(b).getWidth();
            int height            = mInputData.at(b).getHeight();
            int features          = mInputData.at(b).getFeatures();
            mInputRegion.at(b)    = Buffer<float>(width, height, features);
            int const N           = mInputRegion.at(b).getTotalElements();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif // PV_USE_OPENMP_THREADS
            for (int k = 0; k < N; k++) {
               mInputRegion.at(b).set(k, 1.0f);
            }
            fitBufferToGlobalLayer(mInputData.at(b), blockBatchElement);
            fitBufferToGlobalLayer(mInputRegion.at(b), blockBatchElement);
            // Now dataBuffer has input over the global layer. Apply normalizeLuminanceFlag, etc.
            normalizePixels(b);
            // Finally, crop to the part of the image covered by the MPIBlock.
            cropToMPIBlock(mInputData.at(b));
            cropToMPIBlock(mInputRegion.at(b));
         }
         if (getLayerLoc()->bcast) {
            // Broadcast the local portions to the non-root processes in the I/O MPIBlock
            broadcastInput(b, m);
         }
         else {
            // Scatter the local portions to the non-root processes in the I/O MPIBlock
            scatterInput(b, m);
         }
      }
   }
   for (auto &r : mInputRegionTargets) {
      auto *targetBufferData = r->getReadWritePointer();
      pvAssert(r->getBufferSizeAcrossBatch() == getBufferSizeAcrossBatch());
      for (int n = 0; n < getBufferSizeAcrossBatch(); n++) {
         targetBufferData[n] = getInputRegionsAllBatchElements()[n];
      }
   }
}

void InputActivityBuffer::fitBufferToGlobalLayer(Buffer<float> &buffer, int blockBatchElement) {
   pvAssert(getCommunicator()->getIOMPIBlock()->getRank() == 0);
   PVLayerLoc const *loc  = getLayerLoc();
   int targetWidth, targetHeight;
   if (loc->bcast) {
      targetWidth  = loc->nx;
      targetHeight = loc->ny;
   }
   else {
      int const xMargins = mUseInputBCflag ? loc->halo.lt + loc->halo.rt : 0;
      int const yMargins = mUseInputBCflag ? loc->halo.dn + loc->halo.up : 0;
      targetWidth        = loc->nxGlobal + xMargins;
      targetHeight       = loc->nyGlobal + yMargins;
   }

   FatalIf(
         buffer.getFeatures() != loc->nf,
         "ERROR: Input for layer %s has %d features, but layer has %d.\n",
         getName(),
         buffer.getFeatures(),
         loc->nf);

   if (mAutoResizeFlag) {
      bool sameDims = (buffer.getWidth() == targetWidth and buffer.getHeight() == targetHeight);
      bool noOffsetX = (mOffsetX == 0 and mMaxShiftX == 0);
      bool noOffsetY = (mOffsetY == 0 and mMaxShiftY == 0);
      if (sameDims and noOffsetX and noOffsetY) {
         WarnLog().printf(
               "Input layer \"%s\" has AutoResizeFlag set but no resizing is necessary.\n",
               getName());
      }
      else {
         BufferUtils::rescale(
               buffer, targetWidth, targetHeight, mRescaleMethod, mInterpolationMethod, mAnchor);
         buffer.translate(
               -mOffsetX + mRandomShiftX[blockBatchElement],
               -mOffsetY + mRandomShiftY[blockBatchElement]);
      }
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

void InputActivityBuffer::normalizePixels(int batchElement) {
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

void InputActivityBuffer::cropToMPIBlock(Buffer<float> &buffer) {
   const PVLayerLoc *loc = getLayerLoc();
   auto ioMPIBlock       = getCommunicator()->getIOMPIBlock();
   int const startX      = ioMPIBlock->getStartColumn() * loc->nx;
   int const startY      = ioMPIBlock->getStartRow() * loc->ny;
   buffer.translate(-startX, -startY);
   int const xMargins    = mUseInputBCflag ? loc->halo.lt + loc->halo.rt : 0;
   int const yMargins    = mUseInputBCflag ? loc->halo.dn + loc->halo.up : 0;
   int const blockWidth  = ioMPIBlock->getNumColumns() * loc->nx + xMargins;
   int const blockHeight = ioMPIBlock->getNumRows() * loc->ny + yMargins;
   buffer.crop(blockWidth, blockHeight, Buffer<float>::NORTHWEST);
}

void InputActivityBuffer::broadcastInput(int localBatchIndex, int mpiBatchIndex) {
   auto ioMPIBlock = getCommunicator()->getIOMPIBlock();
   if (ioMPIBlock->getRank() == 0) {
      for (int r = 0; r < ioMPIBlock->getNumRows(); ++r) {
         for (int c = 0; c < ioMPIBlock->getNumColumns(); ++c) {
            int destRank = ioMPIBlock->calcRankFromRowColBatch(r, c, mpiBatchIndex);
            if (destRank == ioMPIBlock->getRank()) { continue; }
            Buffer<float> &dataBuffer = mInputData.at(localBatchIndex);
            float const *dataPointer = dataBuffer.asVector().data();
            MPI_Send(
                  dataPointer,
                  getLayerLoc()->nf,
                  MPI_FLOAT,
                  destRank,
                  1760 + localBatchIndex /*tag*/,
                  ioMPIBlock->getComm());
         }
      }
   }
   else {
      int const procBatchIndex = ioMPIBlock->getBatchIndex();
      if (procBatchIndex != mpiBatchIndex) { return; }
      int sourceRank = 0;
      mInputData.resize(getLayerLoc()->nbatch);
      Buffer<float> &dataBuffer = mInputData.at(localBatchIndex);
      dataBuffer.resize(1, 1, getLayerLoc()->nf);
      float *dataPointer = &dataBuffer.asVector().at(0);
      MPI_Recv(
            dataPointer,
            getLayerLoc()->nf,
            MPI_FLOAT,
            sourceRank,
            1760 + localBatchIndex /*tag*/,
            ioMPIBlock->getComm(),
            MPI_STATUS_IGNORE);
   }

   if (ioMPIBlock->getBatchIndex() != mpiBatchIndex) {
      return;
   }
   // All processes that make it to this point have the indicated MPI batch index,
   // and dataBuffer has the correct data for the indicated batch index.
   // We now copy the data into the activity buffer.
   float *activityData  = getReadWritePointer();
   int const bufferSize = getBufferSize();

   Buffer<float> &dataBuffer = mInputData.at(localBatchIndex);
   // Sanity checks on buffer sizes
   FatalIf(
         dataBuffer.getHeight() != 1,
         "Buffer from disk should have ny=1, but it is %d\n", dataBuffer.getHeight());
   FatalIf(
         dataBuffer.getWidth() != 1,
         "Buffer from disk should have nx=1, but it is %d\n", dataBuffer.getHeight());
   FatalIf(
         dataBuffer.getFeatures() != bufferSize,
         "Buffer from disk should have %d features, but it has %d\n",
         dataBuffer.getHeight(), bufferSize);
   float *activityBatch = &activityData[localBatchIndex * bufferSize];
   for (int n = 0; n < bufferSize; ++n) {
      activityBatch[n] = dataBuffer.at(0,0,n);
   }
}

void InputActivityBuffer::scatterInput(int localBatchIndex, int mpiBatchIndex) {
   auto ioMPIBlock          = getCommunicator()->getIOMPIBlock();
   int const procBatchIndex = ioMPIBlock->getBatchIndex();
   if (procBatchIndex != 0 and procBatchIndex != mpiBatchIndex) {
      return;
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

   if (ioMPIBlock->getRank() == 0) {
      dataBuffer   = mInputData.at(localBatchIndex);
      regionBuffer = mInputRegion.at(localBatchIndex);
   }
   else {
      dataBuffer.resize(activityWidth, activityHeight, loc->nf);
      regionBuffer.resize(activityWidth, activityHeight, loc->nf);
   }
   BufferUtils::scatter<float>(ioMPIBlock, dataBuffer, loc->nx, loc->ny, mpiBatchIndex, 0);
   BufferUtils::scatter<float>(ioMPIBlock, regionBuffer, loc->nx, loc->ny, mpiBatchIndex, 0);
   if (procBatchIndex != mpiBatchIndex) {
      return;
   }

   // All processes that make it to this point have the indicated MPI batch index,
   // and dataBuffer has the correct data for the indicated batch index.
   // Clear the current activity for this batch element; then copy the input data over row by row.
   float *activityData  = getReadWritePointer();
   int const bufferSize = getBufferSize();
   float *activityBatch = &activityData[localBatchIndex * bufferSize];
   for (int n = 0; n < bufferSize; ++n) {
      activityBatch[n] = mPadValue;
   }

   int const numFeatures = getLayerLoc()->nf;
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
               activityBatch[activityIndex] = dataBuffer.at(x, y, f);
            }
         }
      }
   }
   if (mNeedInputRegionsPointer) {
      float *inputRegionBuffer = &mInputRegionsAllBatchElements[localBatchIndex * bufferSize];
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
}

} // namespace PV
