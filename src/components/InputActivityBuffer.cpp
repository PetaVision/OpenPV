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
#include "columns/Random.hpp"
#include "layers/HyPerLayer.hpp"
#include "structures/PVLayerLoc.hpp"
#include "io/FileStreamBuilder.hpp"
#include "structures/MPIBlock.hpp"
#include "utils/BufferUtilsMPI.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include "utils/cl_random.h"
#include "utils/conversions.hpp"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>

namespace PV {

InputActivityBuffer::InputActivityBuffer(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

InputActivityBuffer::~InputActivityBuffer() {}

void InputActivityBuffer::initialize(char const *name, PVParams *params, Communicator const *comm) {
   ActivityBuffer::initialize(name, params, comm);
}

void InputActivityBuffer::setObjectType() { mObjectType = "InputActivityBuffer"; }

int InputActivityBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ActivityBuffer::ioParamsFillGroup(ioFlag);
   ioParam_syncLayer(ioFlag);
   ioParam_displayPeriod(ioFlag);
   ioParam_inputPath(ioFlag);
   ioParam_offsetAnchor(ioFlag);
   ioParam_offsets(ioFlag);
   ioParam_jitterChangeInterval(ioFlag);
   ioParam_jitterChangeIntervalUnit(ioFlag);
   ioParam_maxShifts(ioFlag);
   ioParam_flipsEnabled(ioFlag);
   ioParam_flipsToggle(ioFlag);
   ioParam_autoResizeFlag(ioFlag);
   ioParam_aspectRatioAdjustment(ioFlag);
   ioParam_interpolationMethod(ioFlag);
   ioParam_inverseFlag(ioFlag);
   ioParam_normalizeLuminanceFlag(ioFlag);
   ioParam_normalizeStdDev(ioFlag);
   ioParam_useInputBCflag(ioFlag);
   ioParam_padValue(ioFlag);
   ioParam_batchMethod(ioFlag);
   ioParam_start_frame_index(ioFlag);
   ioParam_skip_frame_index(ioFlag);
   ioParam_resetToStartOnLoop(ioFlag);
   ioParam_writeFrameToTimestamp(ioFlag);

   return status;
}

void InputActivityBuffer::ioParam_syncLayer(enum ParamsIOFlag ioFlag) {
   bool warnIfAbsentFlag = false;
   parameters()->ioParamString(
         ioFlag, getName(), "syncLayer", &mSyncLayer, nullptr /*default*/, warnIfAbsentFlag);
   // The communicate stage will check that this is a valid input layer.
}

void InputActivityBuffer::ioParam_displayPeriod(enum ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(getName(), "syncLayer"));
   if (mSyncLayer == nullptr or mSyncLayer[0] == '\0') {
      parameters()->ioParamValue(ioFlag, getName(), "displayPeriod", &mDisplayPeriod, mDisplayPeriod);
   }
}

void InputActivityBuffer::ioParam_inputPath(enum ParamsIOFlag ioFlag) {
   char *tempString = nullptr;
   if (ioFlag == PARAMS_IO_WRITE) {
      tempString = strdup(mInputPath.c_str());
   }
   parameters()->ioParamStringRequired(ioFlag, getName(), "inputPath", &tempString);
   if (ioFlag == PARAMS_IO_READ) {
      mInputPath = std::string(tempString);
   }
   free(tempString);
}

void InputActivityBuffer::ioParam_useInputBCflag(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag,
         getName(),
         "useInputBCflag",
         &mUseInputBCflag,
         mUseInputBCflag);
}

void InputActivityBuffer::ioParam_offsets(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "offsetX", &mOffsetX, mOffsetX);
   parameters()->ioParamValue(ioFlag, getName(), "offsetY", &mOffsetY, mOffsetY);
}

void InputActivityBuffer::ioParam_jitterChangeInterval(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag, getName(), "jitterChangeInterval", &mJitterChangeInterval, mJitterChangeInterval);
}

void InputActivityBuffer::ioParam_jitterChangeIntervalUnit(enum ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(getName(), "jitterChangeInterval"));
   if (mJitterChangeInterval > 0) {
      parameters()->ioParamString(
            ioFlag,
            getName(),
            "jitterChangeIntervalUnit",
            &mJitterChangeIntervalUnit,
            "displayPeriod" /*default*/);
      if (ioFlag == PARAMS_IO_READ) {
         for (char *c = mJitterChangeIntervalUnit; *c != '\0'; c++) {
            *c = (char)std::tolower((int)*c);
         }
         mJitterChangeIntervalInTimesteps = mJitterChangeInterval;
         if (!strcmp(mJitterChangeIntervalUnit, "displayperiod")) {
            std::strncpy(
                  mJitterChangeIntervalUnit, "displayPeriod", strlen(mJitterChangeIntervalUnit));
            mJitterChangeIntervalInTimesteps *= mDisplayPeriod;
         }
         FatalIf(
               strcmp(mJitterChangeIntervalUnit, "displayPeriod") != 0
                     and strcmp(mJitterChangeIntervalUnit, "timestep") != 0,
               "\"%s\" jitterChangeIntervalUnit must be either \"displayPeriod\" or \"timestep\"\n",
               getName());
      }
   }
}

void InputActivityBuffer::ioParam_maxShifts(enum ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(getName(), "jitterChangeInterval"));
   if (mJitterChangeInterval > 0) {
      parameters()->ioParamValue(ioFlag, getName(), "maxShiftX", &mMaxShiftX, mMaxShiftX);
      parameters()->ioParamValue(ioFlag, getName(), "maxShiftY", &mMaxShiftY, mMaxShiftY);
   }
}

void InputActivityBuffer::ioParam_flipsEnabled(enum ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(getName(), "jitterChangeInterval"));
   if (mJitterChangeInterval > 0) {
      parameters()->ioParamValue(ioFlag, getName(), "xFlipEnabled", &mXFlipEnabled, mXFlipEnabled);
      parameters()->ioParamValue(ioFlag, getName(), "yFlipEnabled", &mYFlipEnabled, mYFlipEnabled);
   }
}

void InputActivityBuffer::ioParam_flipsToggle(enum ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(getName(), "jitterChangeInterval"));
   if (mJitterChangeInterval > 0) {
      parameters()->ioParamValue(ioFlag, getName(), "xFlipToggle", &mXFlipToggle, mXFlipToggle);
      parameters()->ioParamValue(ioFlag, getName(), "yFlipToggle", &mYFlipToggle, mYFlipToggle);
   }
}

void InputActivityBuffer::ioParam_offsetAnchor(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      char *offsetAnchor = nullptr;
      parameters()->ioParamString(ioFlag, getName(), "offsetAnchor", &offsetAnchor, "tl");
      if (offsetAnchor == nullptr or strlen(offsetAnchor) != (size_t)2) {
         badOffsetAnchorString(offsetAnchor);
      }
      offsetAnchor[0] = (char)std::tolower((int)offsetAnchor[0]);
      offsetAnchor[1] = (char)std::tolower((int)offsetAnchor[1]);
      if (strcmp(offsetAnchor, "tl") == 0 or strcmp(offsetAnchor, "lt") == 0) {
         mAnchor = Buffer<float>::NORTHWEST;
      }
      else if (strcmp(offsetAnchor, "tc") == 0 or strcmp(offsetAnchor, "ct") == 0) {
         mAnchor = Buffer<float>::NORTH;
      }
      else if (strcmp(offsetAnchor, "tr") == 0 or strcmp(offsetAnchor, "rt") == 0) {
         mAnchor = Buffer<float>::NORTHEAST;
      }
      else if (strcmp(offsetAnchor, "cl") == 0 or strcmp(offsetAnchor, "lc") == 0) {
         mAnchor = Buffer<float>::WEST;
      }
      else if (strcmp(offsetAnchor, "cc") == 0) {
         mAnchor = Buffer<float>::CENTER;
      }
      else if (strcmp(offsetAnchor, "cr") == 0 or strcmp(offsetAnchor, "rc") == 0) {
         mAnchor = Buffer<float>::EAST;
      }
      else if (strcmp(offsetAnchor, "bl") == 0 or strcmp(offsetAnchor, "lb") == 0) {
         mAnchor = Buffer<float>::SOUTHWEST;
      }
      else if (strcmp(offsetAnchor, "bc") == 0 or strcmp(offsetAnchor, "cb") == 0) {
         mAnchor = Buffer<float>::SOUTH;
      }
      else if (strcmp(offsetAnchor, "br") == 0 or strcmp(offsetAnchor, "rb") == 0) {
         mAnchor = Buffer<float>::SOUTHEAST;
      }
      else {
         badOffsetAnchorString(offsetAnchor);
      }
      free(offsetAnchor);
   }
   else { // Writing
      char *offsetAnchor = (char *)malloc((size_t)3);
      switch (mAnchor) {
         case Buffer<float>::CENTER: strncpy(offsetAnchor, "cc", 3UL); break;
         case Buffer<float>::NORTH: strncpy(offsetAnchor, "tc", 3UL); break;
         case Buffer<float>::NORTHEAST: strncpy(offsetAnchor, "tr", 3UL); break;
         case Buffer<float>::EAST: strncpy(offsetAnchor, "cr", 3UL); break;
         case Buffer<float>::SOUTHEAST: strncpy(offsetAnchor, "br", 3UL); break;
         case Buffer<float>::SOUTH: strncpy(offsetAnchor, "bc", 3UL); break;
         case Buffer<float>::SOUTHWEST: strncpy(offsetAnchor, "bl", 3UL); break;
         case Buffer<float>::WEST: strncpy(offsetAnchor, "cl", 3UL); break;
         case Buffer<float>::NORTHWEST: strncpy(offsetAnchor, "tl", 3UL); break;
      }
      parameters()->ioParamString(ioFlag, getName(), "offsetAnchor", &offsetAnchor, "tl");
      free(offsetAnchor);
   }
}

void InputActivityBuffer::badOffsetAnchorString(char const *offsetAnchor) {
   Fatal().printf(
         "%s: offsetAnchor %s is not recognized. The offsetAnchor parameter must be a two-letter "
         "string.  One character must be \"t\", \"c\", or \"b\" (for top, center or bottom); and "
         "the other character must be \"l\", \"c\", or \"r\" (for left, center or right).\n",
         getDescription_c(),
         offsetAnchor ? offsetAnchor : "NULL");
}

void InputActivityBuffer::ioParam_autoResizeFlag(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag,
         getName(),
         "autoResizeFlag",
         &mAutoResizeFlag,
         mAutoResizeFlag);
}

void InputActivityBuffer::ioParam_aspectRatioAdjustment(enum ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(getName(), "autoResizeFlag"));
   if (mAutoResizeFlag) {
      char *aspectRatioAdjustment = nullptr;
      if (ioFlag == PARAMS_IO_WRITE) {
         switch (mRescaleMethod) {
            case BufferUtils::CROP: aspectRatioAdjustment = strdup("crop"); break;
            case BufferUtils::PAD: aspectRatioAdjustment = strdup("pad"); break;
         }
      }
      parameters()->ioParamString(
            ioFlag, getName(), "aspectRatioAdjustment", &aspectRatioAdjustment, "crop");
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
         if (mCommunicator->commRank() == 0) {
            ErrorLog().printf(
                  "%s: aspectRatioAdjustment must be either \"crop\" or \"pad\".\n",
                  getDescription_c());
         }
         MPI_Barrier(mCommunicator->communicator());
         exit(EXIT_FAILURE);
      }
      free(aspectRatioAdjustment);
   }
}

void InputActivityBuffer::ioParam_interpolationMethod(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(getName(), "autoResizeFlag"));
   if (mAutoResizeFlag) {
      char *interpolationMethodString = nullptr;
      if (ioFlag == PARAMS_IO_READ) {
         parameters()->ioParamString(
               ioFlag,
               getName(),
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
         else if (!strncmp(
                        interpolationMethodString, "nearestneighbor", strlen("nearestneighbor"))) {
            mInterpolationMethod = BufferUtils::NEAREST;
         }
         else {
            if (mCommunicator->commRank() == 0) {
               ErrorLog().printf(
                     "%s: interpolationMethod must be either \"bicubic\" or \"nearestNeighbor\".\n",
                     getDescription_c());
            }
            MPI_Barrier(mCommunicator->communicator());
            exit(EXIT_FAILURE);
         }
      }
      else {
         assert(ioFlag == PARAMS_IO_WRITE);
         switch (mInterpolationMethod) {
            case BufferUtils::BICUBIC: interpolationMethodString = strdup("bicubic"); break;
            case BufferUtils::NEAREST: interpolationMethodString = strdup("nearestNeighbor"); break;
         }
         parameters()->ioParamString(
               ioFlag,
               getName(),
               "interpolationMethod",
               &interpolationMethodString,
               "bicubic",
               true /*warn if absent*/);
      }
      free(interpolationMethodString);
   }
}

void InputActivityBuffer::ioParam_inverseFlag(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "inverseFlag", &mInverseFlag, mInverseFlag);
}

void InputActivityBuffer::ioParam_normalizeLuminanceFlag(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag,
         getName(),
         "normalizeLuminanceFlag",
         &mNormalizeLuminanceFlag,
         mNormalizeLuminanceFlag);
}

void InputActivityBuffer::ioParam_normalizeStdDev(enum ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(getName(), "normalizeLuminanceFlag"));
   if (mNormalizeLuminanceFlag) {
      parameters()->ioParamValue(
            ioFlag, getName(), "normalizeStdDev", &mNormalizeStdDev, mNormalizeStdDev);
   }
}
void InputActivityBuffer::ioParam_padValue(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "padValue", &mPadValue, mPadValue);
}

void InputActivityBuffer::ioParam_batchMethod(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(getName(), "syncLayer"));
   if (mSyncLayer != nullptr and mSyncLayer[0] != '\0') {
      return;
   }
   char *batchMethod = nullptr;
   if (ioFlag == PARAMS_IO_WRITE) {
      switch (mBatchMethod) {
         case BYFILE: batchMethod = strdup("byFile"); break;
         case BYLIST: batchMethod = strdup("byList"); break;
         case BYSPECIFIED: batchMethod = strdup("bySpecified"); break;
         case RANDOM: batchMethod = strdup("random"); break;
      }
   }
   parameters()->ioParamString(ioFlag, getName(), "batchMethod", &batchMethod, "byFile");
   if (strcmp(batchMethod, "byImage") == 0 || strcmp(batchMethod, "byFile") == 0) {
      mBatchMethod = BYFILE;
   }
   else if (strcmp(batchMethod, "byMovie") == 0 || strcmp(batchMethod, "byList") == 0) {
      mBatchMethod = BYLIST;
   }
   else if (strcmp(batchMethod, "bySpecified") == 0) {
      mBatchMethod = BYSPECIFIED;
   }
   else if (strcmp(batchMethod, "random") == 0) {
      mBatchMethod = RANDOM;
   }
   else {
      Fatal() << "Input layer " << getName() << " batchMethod not recognized. "
                 "Options are \"byFile\", \"byList\", \"bySpecified\", and \"random\".\n";
   }
   free(batchMethod);
}

void InputActivityBuffer::ioParam_start_frame_index(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(getName(), "syncLayer"));
   if (mSyncLayer != nullptr and mSyncLayer[0] != '\0') {
      return;
   }
   pvAssert(!parameters()->presentAndNotBeenRead(getName(), "batchMethod"));
   FatalIf(
         mBatchMethod == RANDOM and this->parameters()->present(getName(), "start_frame_index"),
         "Input layer \"%s\": start_frame_index cannot be used with batchMethod = random\n",
         getName());
      
   int *paramsStartFrameIndex;
   int length = 0;
   if (ioFlag == PARAMS_IO_WRITE) {
      length                = mStartFrameIndex.size();
      paramsStartFrameIndex = static_cast<int *>(calloc(length, sizeof(int)));
      for (int i = 0; i < length; ++i) {
         paramsStartFrameIndex[i] = mStartFrameIndex.at(i);
      }
   }
   this->parameters()->ioParamArray(
         ioFlag, this->getName(), "start_frame_index", &paramsStartFrameIndex, &length);
   mStartFrameIndex.clear();
   mStartFrameIndex.resize(length);
   for (int i = 0; i < length; ++i) {
      mStartFrameIndex.at(i) = paramsStartFrameIndex[i];
   }
   free(paramsStartFrameIndex);
}

void InputActivityBuffer::ioParam_skip_frame_index(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(getName(), "syncLayer"));
   if (mSyncLayer != nullptr and mSyncLayer[0] != '\0') {
      return;
   }
   pvAssert(!parameters()->presentAndNotBeenRead(getName(), "batchMethod"));
   if (mBatchMethod != BYSPECIFIED) {
      // bySpecified is the only batchMethod that uses skip_frame_index.
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
   this->parameters()->ioParamArray(
         ioFlag, this->getName(), "skip_frame_index", &paramsSkipFrameIndex, &length);
   mSkipFrameIndex.clear();
   mSkipFrameIndex.resize(length);
   for (int i = 0; i < length; ++i) {
      mSkipFrameIndex.at(i) = std::max(paramsSkipFrameIndex[i], 1);
   }
   free(paramsSkipFrameIndex);
}

void InputActivityBuffer::ioParam_resetToStartOnLoop(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(getName(), "syncLayer"));
   if (mSyncLayer != nullptr and mSyncLayer[0] != '\0') {
      return;
   }
   assert(!parameters()->presentAndNotBeenRead(getName(), "batchMethod"));
   if (mBatchMethod == BYSPECIFIED) {
      parameters()->ioParamValue(
            ioFlag, getName(), "resetToStartOnLoop", &mResetToStartOnLoop, mResetToStartOnLoop);
   }
   else {
      mResetToStartOnLoop = false;
   }
}

void InputActivityBuffer::ioParam_writeFrameToTimestamp(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(getName(), "syncLayer"));
   if (mSyncLayer != nullptr and mSyncLayer[0] != '\0') {
      return;
   }
   assert(!parameters()->presentAndNotBeenRead(getName(), "displayPeriod"));
   if (mDisplayPeriod > 0) {
      parameters()->ioParamValue(
            ioFlag,
            getName(),
            "writeFrameToTimestamp",
            &mWriteFrameToTimestamp,
            mWriteFrameToTimestamp);
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
   auto *objectTable = message->mObjectTable;
   int inputCount        = countInputImages();
   mInputCount = inputCount ? inputCount : 1;
   if (mSyncLayer != nullptr and mSyncLayer[0] != '\0' and mSyncActivityBuffer == nullptr) {
      // check that parameter string matches an input layer
      auto syncedLayer = objectTable->findObject<HyPerLayer>(mSyncLayer);
      FatalIf(
            syncedLayer == nullptr,
            "Input layer \"%s\" set syncLayer to \"%s\" but this is not a layer in the column.\n",
            getName(),
            mSyncLayer);
      auto *syncedActivity = syncedLayer->getComponentByType<ActivityComponent>();
      pvAssert(syncedActivity != nullptr); // All layers should have an activity component
      mSyncActivityBuffer = syncedActivity->getComponentByType<InputActivityBuffer>();
      FatalIf(
            mSyncActivityBuffer == nullptr,
            "Input layer \"%s\" set syncLayer to \"%s\" but this is not an input layer.\n",
            getName(),
            mSyncLayer);
      // check that the synced layer has finished its communicate phase
      if (!mSyncActivityBuffer->getInitInfoCommunicatedFlag()) {
         WarnLog().printf(
               "Input layer \"%s\" must postpone until syncLayer \"%s\" finishes its "
               "Communicate stage.\n",
               getName(),
               mSyncLayer);
         status = status + Response::POSTPONE;
         return status;
      }
      // check that the synced layer and this layer have the same number of input images.
      FatalIf(
            mInputCount != mSyncActivityBuffer->getInputCount(),
            "Input layer \"%s\" and its syncLayer \"%s\" have different input counts "
            "(%d versus %d)\n",
            getName(),
            mSyncLayer,
            mInputCount,
            mSyncActivityBuffer->getInputCount());
      mDisplayPeriod = mSyncActivityBuffer->getDisplayPeriod();
      mBatchMethod = mSyncActivityBuffer->mBatchMethod;
      mStartFrameIndex = mSyncActivityBuffer->mStartFrameIndex;
      mSkipFrameIndex = mSyncActivityBuffer->mSkipFrameIndex;
      mResetToStartOnLoop = mSyncActivityBuffer->mResetToStartOnLoop;
      mWriteFrameToTimestamp = mSyncActivityBuffer->mWriteFrameToTimestamp;
   }
   else {
      FatalIf(
            !checkArrayLength(mStartFrameIndex, message->mNBatchGlobal),
            "Input layer %s: start_frame_index requires either 0 or nbatch=%d values.\n",
            getName(),
            message->mNBatchGlobal);
      FatalIf(
            !checkArrayLength(mSkipFrameIndex, message->mNBatchGlobal),
            "Input layer %s: skip_frame_index requires either 0 or nbatch=%d values.\n",
            getName(),
            message->mNBatchGlobal);
   }
   return Response::SUCCESS;
}

bool InputActivityBuffer::checkArrayLength(std::vector<int> &array, int correctLength) {
   bool correctSizeFlag;
   int length = static_cast<int>(array.size());
   if (length == 0) {
      array.resize(correctLength);
      correctSizeFlag = true;
   }
   else if (array.size() != correctLength) {
      correctSizeFlag = false;
   }
   else {
      correctSizeFlag = true;
   }
   return correctSizeFlag;
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
   if (mBatchMethod == RANDOM) {
      // Need to create Random object in all processes, to keep RandomSeed in sync,
      // even though only the root process will use it.
      taus_uint4 batchIndexRNG;
      if (mSyncActivityBuffer == nullptr) {
         Random batchIndexRandomObj(1);
         batchIndexRNG = *batchIndexRandomObj.getRNG(0);
         mBatchIndexRandomState.resize(4);
         mBatchIndexRandomState[0] = batchIndexRNG.s0;
         mBatchIndexRandomState[1] = batchIndexRNG.state.s1;
         mBatchIndexRandomState[2] = batchIndexRNG.state.s2;
         mBatchIndexRandomState[3] = batchIndexRNG.state.s3;
      }
      else {
         if (!mSyncActivityBuffer->getDataStructuresAllocatedFlag()) {
            return Response::POSTPONE;
         }
         mBatchIndexRandomState = mSyncActivityBuffer->mBatchIndexRandomState;
         batchIndexRNG.s0       = mBatchIndexRandomState[0];
         batchIndexRNG.state.s1 = mBatchIndexRandomState[1];
         batchIndexRNG.state.s2 = mBatchIndexRandomState[2];
         batchIndexRNG.state.s3 = mBatchIndexRandomState[3];
      }
      initializeBatchIndexer(&batchIndexRNG);
   }
   else {
      initializeBatchIndexer(nullptr);
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
      int numBatch = getLayerLoc()->nbatch;
      int nBatch   = getCommunicator()->getIOMPIBlock()->getBatchDimension() * numBatch;
      mRandomShiftX.resize(nBatch);
      mRandomShiftY.resize(nBatch);
      mMirrorFlipX.resize(nBatch);
      mMirrorFlipY.resize(nBatch);
      mInputData.resize(numBatch);
      mInputRegion.resize(numBatch);
      mBatchIndexer->setWrapToStartIndex(mResetToStartOnLoop);

      mFrameNumbers = mBatchIndexer->getIndices();
      auto *checkpointer = message->mDataRegistry;
      checkpointer->registerCheckpointData<int>(
            getName(),
            std::string("FrameNumbers"),
            mFrameNumbers.data(),
            mFrameNumbers.size(),
            false /*broadcastFlag*/,
            false /*constantEntireRunFlag*/);
   }
   if (mBatchMethod == RANDOM) {
      checkpointer->registerCheckpointData<unsigned int>(
            getName(),
            std::string("RandomState"),
            mBatchIndexRandomState.data(),
            mBatchIndexRandomState.size(),
            true /*broadcastFlag*/,
            false /*constantEntireRunFlag*/);
   }
   return Response::SUCCESS;
}

void InputActivityBuffer::initializeBatchIndexer(taus_uint4 *state) {
   auto ioMPIBlock = getCommunicator()->getIOMPIBlock();
   pvAssert(ioMPIBlock);
   if (ioMPIBlock->getRank() != 0) {
      return;
   }
   int localBatchCount  = getLayerLoc()->nbatch;
   int mpiGlobalCount   = ioMPIBlock->getGlobalBatchDimension();
   int globalBatchCount = localBatchCount * mpiGlobalCount;
   int batchOffset      = localBatchCount * ioMPIBlock->getStartBatch();
   int blockBatchCount  = localBatchCount * ioMPIBlock->getBatchDimension();
   std::vector<int> startIndices;
   std::vector<int> skipAmounts;
   switch (mBatchMethod) {
      case BYFILE:
         startIndices.resize(blockBatchCount);
         skipAmounts.resize(blockBatchCount, globalBatchCount);
         for (int b = 0; b < blockBatchCount; ++b) {
            startIndices[b] = (mStartFrameIndex[b] + batchOffset + b) % mInputCount;
         }
         break;
      case BYLIST:
         startIndices.resize(blockBatchCount);
         skipAmounts.resize(blockBatchCount, 1);
         for (int b = 0; b < blockBatchCount; ++b) {
            startIndices[b] =
                  (mStartFrameIndex[b] + (batchOffset + b) * (mInputCount / globalBatchCount) ) %
                  mInputCount;
         }
         break;
      case BYSPECIFIED:
         startIndices = mStartFrameIndex;
         skipAmounts = mSkipFrameIndex;
         break;
      case RANDOM:
         // batchMethod=random does not use startIndices or skipAmounts
         break;
      default:
         Fatal().printf(
               "InputActivityBuffer::initializeBatchIndexer called with bad BatchMethod %d\n",
               mBatchMethod);
         break;
   }
   if (mBatchMethod != RANDOM) {
      mBatchIndexer = std::unique_ptr<BatchIndexer>(new BatchIndexer(
            std::string(getName()),
            globalBatchCount,
            blockBatchCount,
            mInputCount,
            startIndices,
            skipAmounts));
   }
   else {
      mBatchIndexer = std::unique_ptr<BatchIndexer>(new BatchIndexer(
            std::string(getName()),
            globalBatchCount,
            blockBatchCount,
            mInputCount,
            batchOffset,
            *state));
   }
}

std::string const &
InputActivityBuffer::getCurrentFilename(int localBatchIndex, int mpiBatchIndex) const {
   return mInputPath;
}

Response::Status
InputActivityBuffer::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   Response::Status status = ActivityBuffer::initializeState(message);
   if (!Response::completed(status)) {
      return status;
   }
   if (mJitterChangeIntervalInTimesteps > 0) {
      mJitterRNG = std::make_shared<Random>(1);
      // All process must create the Random object to keep RandomSeed in sync
   }
   retrieveInput(0.0 /*simulationTime*/, message->mDeltaTime);
   return Response::SUCCESS;
}

Response::Status InputActivityBuffer::readStateFromCheckpoint(Checkpointer *checkpointer) {
   if (getCommunicator()->getIOMPIBlock()->getRank() == 0) {
      pvAssert(mInitializeFromCheckpointFlag);
      checkpointer->readNamedCheckpointEntry(getName(), "FrameNumbers", false /*not constant*/);
      mBatchIndexer->setIndices(mFrameNumbers);
      if (mBatchMethod == RANDOM) {
         taus_uint4 batchRNG;
         batchRNG.s0 = mBatchIndexRandomState[0];
         batchRNG.state.s1 = mBatchIndexRandomState[1];
         batchRNG.state.s2 = mBatchIndexRandomState[2];
         batchRNG.state.s3 = mBatchIndexRandomState[3];
         mBatchIndexer->setRandomState(batchRNG);
      }
   }
   return Response::SUCCESS;
}

Response::Status InputActivityBuffer::processCheckpointRead(double simTime) {
   if (getCommunicator()->getIOMPIBlock()->getRank() == 0) {
      mBatchIndexer->setIndices(mFrameNumbers);
      if (mBatchMethod == RANDOM) {
         taus_uint4 batchRNG;
         batchRNG.s0 = mBatchIndexRandomState[0];
         batchRNG.state.s1 = mBatchIndexRandomState[1];
         batchRNG.state.s2 = mBatchIndexRandomState[2];
         batchRNG.state.s3 = mBatchIndexRandomState[3];
         mBatchIndexer->setRandomState(batchRNG);
      }
   }
   return Response::SUCCESS;
}

Response::Status InputActivityBuffer::prepareCheckpointWrite(double simTime) {
   if (getCommunicator()->getIOMPIBlock()->getRank() == 0) {
      auto &frameNumbers = mBatchIndexer->getIndices();
      unsigned int N = frameNumbers.size();
      assert(static_cast<unsigned int>(mFrameNumbers.size()) == N);
      for (unsigned int n = 0; n < N; ++n) {
         mFrameNumbers[n] = frameNumbers[n];
      }
      if (mBatchMethod == RANDOM) {
         taus_uint4 const &batchRNG = mBatchIndexer->getRandomState();
         mBatchIndexRandomState[0] = batchRNG.s0;
         mBatchIndexRandomState[1] = batchRNG.state.s1;
         mBatchIndexRandomState[2] = batchRNG.state.s2;
         mBatchIndexRandomState[3] = batchRNG.state.s3;
      }
   }
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
      mBatchIndexer->advanceIndices();
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
            mRandomShiftX[b] = -mMaxShiftX + (mJitterRNG->randomUInt() % (2 * mMaxShiftX + 1));
            mRandomShiftY[b] = -mMaxShiftY + (mJitterRNG->randomUInt() % (2 * mMaxShiftY + 1));
            if (mXFlipEnabled) {
               mMirrorFlipX[b] =
                     mXFlipToggle ? !mMirrorFlipX[b] : (mJitterRNG->randomUInt() % 256U) >= 128U;
            }
            if (mYFlipEnabled) {
               mMirrorFlipY[b] =
                     mYFlipToggle ? !mMirrorFlipY[b] : (mJitterRNG->randomUInt() % 256U) >= 128U;
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
         "ERROR: Input data for layer \"%s\" has nf=%d features, but layer has nf=%d.\n",
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
