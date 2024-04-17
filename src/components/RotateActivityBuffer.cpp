#include "RotateActivityBuffer.hpp"
#include "checkpointing/CheckpointEntryFilePosition.hpp"
// #include "checkpointing/CheckpointEntryRandState.hpp"
#include "io/FileManager.hpp"
#include "io/FileStreamBuilder.hpp"
#include "utils/BufferUtilsMPI.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include <algorithm> // copy()
#include <cmath>
#include <utility>   // swap()

namespace PV {

RotateActivityBuffer::RotateActivityBuffer(
      char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

void RotateActivityBuffer::initialize(
      char const *name, PVParams *params, Communicator const *comm) {
   HyPerActivityBuffer::initialize(name, params, comm);
}

int RotateActivityBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerActivityBuffer::ioParamsFillGroup(ioFlag);
   if (status != PV_SUCCESS) {
      return status;
   }

   ioParam_angleMin(ioFlag);
   ioParam_angleMax(ioFlag);
   if (ioFlag == PARAMS_IO_READ and mAngleMax < mAngleMin) {
      WarnLog().printf(
            "%s: specified value of max (%f) is less than specified value of min (%f)\n",
            getDescription_c(),
            static_cast<double>(mAngleMax),
            static_cast<double>(mAngleMin));
      std::swap(mAngleMax, mAngleMin);
      WarnLog().printf(
            "Switching angleMin to %f and angleMax to %f\n",
            static_cast<double>(mAngleMax),
            static_cast<double>(mAngleMin));
   }
   pvAssert(mAngleMax >= mAngleMin);
   ioParam_angleUnits(ioFlag);
   ioParam_writeAnglesFile(ioFlag);
   return status;
}

void RotateActivityBuffer::ioParam_angleMin(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValueRequired(ioFlag, getName(), "angleMin", &mAngleMin);
}

void RotateActivityBuffer::ioParam_angleMax(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValueRequired(ioFlag, getName(), "angleMax", &mAngleMax);
}

void RotateActivityBuffer::ioParam_angleUnits(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      char *angleUnitsParam = nullptr;
      parameters()->ioParamStringRequired(ioFlag, getName(), "angleUnits", &angleUnitsParam);
      std::string angleUnitsString(angleUnitsParam);
      free(angleUnitsParam);
      for (auto &c : angleUnitsString) {
         c = tolower(c);
      }
      if (angleUnitsString == "degree") {
         angleUnitsString = "degrees";
      }
      if (angleUnitsString == "radian") {
         angleUnitsString = "radians";
      }
      if (angleUnitsString == "degrees") {
         mAngleConversionFactor = std::atan(1.0f) / 45.0f;
         mAngleUnitType = AngleUnitType::DEGREES;
      }
      else if (angleUnitsString == "radians") {
         mAngleConversionFactor = 1.0f;
         mAngleUnitType = AngleUnitType::RADIANS;
      }
   }
   else if (ioFlag == PARAMS_IO_WRITE) {
      std::string angleUnitsString;
      switch(mAngleUnitType) {
         case AngleUnitType::UNSET:
            Fatal().printf("%s AngleUnits parameter has not been set.\n", getDescription_c());
            break;
         case AngleUnitType::DEGREES:
            angleUnitsString = "degrees";
            break;
         case AngleUnitType::RADIANS:
            angleUnitsString = "radians";
            break;
         default:
            Fatal().printf("%s has an unrecognized AngleUnits parameter.\n", getDescription_c());
            break;
      }
      char *angleUnitsParam = strdup(angleUnitsString.c_str());
      FatalIf(
            angleUnitsParam == nullptr,
            "%s unable to write AngleUnitsString parameter\n",
            getDescription_c());
      parameters()->ioParamStringRequired(ioFlag, getName(), "angleUnits", &angleUnitsParam);
      free(angleUnitsParam);
   }
   else {
      Fatal().printf(
            "%s called ioParam_angleUnits with an unrecognized ioFlag.\n",
            getDescription_c());
   }
}

void RotateActivityBuffer::ioParam_writeAnglesFile(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamString(
         ioFlag, getName(), "writeAnglesFile", &mWriteAnglesFile, mWriteAnglesFile);
}

void RotateActivityBuffer::setObjectType() { mObjectType = "RotateActivityBuffer"; }

Response::Status RotateActivityBuffer::allocateDataStructures() {
   auto status = HyPerActivityBuffer::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   if (getCommunicator()->commRank() == 0) {
      mRandState = std::make_shared<Random>(getLayerLoc()->nbatchGlobal);
      auto ioMPIBlock = getCommunicator()->getIOMPIBlock();
      if (ioMPIBlock->getRank() == 0) {
         int numProcs = ioMPIBlock->getBatchDimension();
         int checkpointDataSize = 4 * getLayerLoc()->nbatch * numProcs;
         mRandStateCheckpointData.resize(checkpointDataSize);
      }
      else {
         mRandStateCheckpointData.resize(4 * getLayerLoc()->nbatch);
      }
   }
   return Response::Status::SUCCESS;
}

Response::Status RotateActivityBuffer::registerData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = HyPerActivityBuffer::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *checkpointer = message->mDataRegistry;
   if (getCommunicator()->getIOMPIBlock()->getRank() == 0) {
      if (mWriteAnglesFile) {
         auto fileManager = getCommunicator()->getOutputFileManager();
         fileManager->ensureDirectoryExists(".");
         mWriteAnglesStream = FileStreamBuilder(
               fileManager,
               mWriteAnglesFile,
               true /*textFlag*/,
               false /*readOnlyFlag*/,
               checkpointer->getCheckpointReadDirectory().empty() /*clobberFlag*/,
               checkpointer->doesVerifyWrites())
               .get();
         auto checkpointEntry = std::make_shared<CheckpointEntryFilePosition>(
               getName(), std::string("WriteAnglesFile"), mWriteAnglesStream);
         bool registerSucceeded = checkpointer->registerCheckpointEntry(
               checkpointEntry, false /*constantEntireRunFlag*/);
         FatalIf(
               !registerSucceeded,
               "%s failed to register %s for checkpointing.\n",
               getDescription_c(),
               checkpointEntry->getName().c_str());
      }
      std::string checkpointFilename(getName());
      checkpointFilename.append("_rand_state");
      auto checkpointEntry = std::make_shared<CheckpointEntryData<unsigned int>>(
         checkpointFilename,
         mRandStateCheckpointData.data(),
         mRandStateCheckpointData.size(),
         false /*broadcastingFlag*/);
      bool registerSucceeded =
            checkpointer->registerCheckpointEntry(checkpointEntry, false /*not constant*/);
      FatalIf(
            !registerSucceeded,
            "%s failed to register random state for checkpointing.\n",
            getDescription_c());
   }
   return Response::SUCCESS;
}

void RotateActivityBuffer::applyTransformCPU(
      Buffer<float> const &inputBuffer, Buffer<float> &outputBuffer, float angle) {
   int const nxRes = inputBuffer.getWidth();
   int const nxExt = outputBuffer.getWidth();
   int const xMargin = (nxExt - nxRes) / 2; // assumes left and right margins are equal
   int const nyRes = inputBuffer.getHeight();
   int const nyExt = outputBuffer.getHeight();
   int const yMargin = (nyExt - nyRes) / 2; // assumes bottom and top margins are equal
   int const nf = inputBuffer.getFeatures();

   float const xCenter = 0.5f * static_cast<float>(nxRes - 1);
   float const yCenter = 0.5f * static_cast<float>(nyRes - 1);

   int const numExtended = outputBuffer.getTotalElements();
   int const nbatch = getLayerLoc()->nbatch;
   float sina = std::sin(angle);
   float cosa = std::cos(angle);
   for (int kExt = 0; kExt < numExtended; ++kExt) {
      int const kxExt = kxPos(kExt, nxExt, nyExt, nf);
      int const kyExt = kyPos(kExt, nxExt, nyExt, nf);
      int const kf    = featureIndex(kExt, nxExt, nyExt, nf);
      float xExt = static_cast<float>(kxExt - xMargin) - xCenter;
      float yExt = static_cast<float>(kyExt - yMargin) - yCenter;

      float xSrc = cosa * xExt - sina * yExt + xCenter;
      float ySrc = sina * xExt + cosa * yExt + yCenter;

      float result = interpolate(inputBuffer, xSrc, ySrc, kf);
      outputBuffer.set(kExt, result);
   }
}

float RotateActivityBuffer::interpolate(
      Buffer<float> const &inputBuffer, float xSrc, float ySrc, int feature) {
   int const nx = inputBuffer.getWidth();
   int const ny = inputBuffer.getHeight();
   int const nf = inputBuffer.getFeatures();

   float xSrcFloor = std::floor(xSrc);
   float xSrcInt   = static_cast<int>(xSrcFloor);
   float xSrcFrac  = xSrc - xSrcFloor;

   float ySrcFloor = std::floor(ySrc);
   float ySrcInt   = static_cast<int>(ySrcFloor);
   float ySrcFrac  = ySrc - ySrcFloor;
   
   float valueTL = (xSrcInt >= 0 and xSrcInt < nx and ySrcInt >= 0 and ySrcInt < ny) ?
                   inputBuffer.at(xSrcInt, ySrcInt, feature) : 0.0f;
   valueTL *= (1.0f - xSrcFrac) * (1.0f - ySrcFrac);

   float valueTR = (xSrcInt + 1 >= 0 and xSrcInt + 1 < nx and ySrcInt >= 0 and ySrcInt < ny) ?
                   inputBuffer.at(xSrcInt + 1, ySrcInt, feature) : 0.0f;
   valueTR *= xSrcFrac * (1.0f - ySrcFrac);

   float valueBL = (xSrcInt >= 0 and xSrcInt < nx and ySrcInt + 1 >= 0 and ySrcInt + 1 < ny) ?
                   inputBuffer.at(xSrcInt, ySrcInt + 1, feature) : 0.0f;
   valueBL *= (1.0f - xSrcFrac) * ySrcFrac;

   float valueBR = (xSrcInt + 1 >= 0 and xSrcInt + 1 < nx and ySrcInt + 1 >= 0 and ySrcInt + 1 < ny) ?
                   inputBuffer.at(xSrcInt + 1, ySrcInt + 1, feature) : 0.0f;
   valueBR *= xSrcFrac * ySrcFrac;

   float value = valueTL + valueTR + valueBL + valueBR;
   return value;
}

void RotateActivityBuffer::transform(Buffer<float> &localVBuffer, int bLocal, float angleRadians) {
   auto localMPIBlock = getCommunicator()->getLocalMPIBlock();
   PVLayerLoc const *loc = getLayerLoc();
   pvAssert(localVBuffer.getWidth() == loc->nx);
   pvAssert(localVBuffer.getHeight() == loc->ny);
   float const* batchStartV = mInternalState->getBufferData(bLocal);
   auto &localVBufferVector = localVBuffer.asVector();
   std::copy(batchStartV, &batchStartV[mBufferSize], &localVBufferVector[0]);
   Buffer<float> globalV = BufferUtils::gather(
         localMPIBlock,
         localVBuffer,
         loc->nx,
         loc->ny,
         0 /* Local communicator always has batch dimension of 1 */,
         0 /* destination process */);
   // root process of local communicator now has entire batch element in globalV buffer.
   Buffer<float> activityBuffer;
   if (localMPIBlock->getRank() == 0) {
      // Set activityBuffer buffer to size of global extended buffer, to hold result of rotation
      int const nxGlobalExt = loc->nxGlobal + loc->halo.lt + loc->halo.rt;
      int const nyGlobalExt = loc->nyGlobal + loc->halo.dn + loc->halo.up;
      activityBuffer.resize(nxGlobalExt, nyGlobalExt, loc->nf);
      int globalMPIBatchIndex = localMPIBlock->getStartBatch() + localMPIBlock->getBatchIndex();
      int bGlobal = bLocal + loc->nbatch * globalMPIBatchIndex;
      applyTransformCPU(globalV, activityBuffer, angleRadians);
   }
   else {
      int const nxLocalExt = loc->nx + loc->halo.lt + loc->halo.rt;
      int const nyLocalExt = loc->ny + loc->halo.dn + loc->halo.up;
      activityBuffer.resize(nxLocalExt, nyLocalExt, loc->nf);
   }
   BufferUtils::scatter(localMPIBlock, activityBuffer, loc->nx, loc->ny, 0, 0);
   auto const &activityBufferVector = activityBuffer.asVector();
   int const numLocalExtended = activityBuffer.getTotalElements();
   std::copy(
         &activityBufferVector[0],
         &activityBufferVector[numLocalExtended],
         &mBufferData[bLocal * mBufferSize]);
}

void RotateActivityBuffer::updateBufferCPU(double simTime, double deltaTime) {
   PVLayerLoc const *loc = getLayerLoc();
   Buffer<float> localVBuffer(loc->nx, loc->ny, loc->nf);  
   float angleRadians = 0.0f;
   for (int bGlobal = 0; bGlobal < loc->nbatchGlobal; ++bGlobal) {
      if (getCommunicator()->getLocalMPIBlock()->getRank() == 0) {
         float angle = mRandState->uniformRandom(bGlobal, mAngleMin, mAngleMax);
         angleRadians = mAngleConversionFactor * angle;
      }
      int bLocal = bGlobal - loc->kb0;
      if (bLocal >= 0 and bLocal < loc->nbatch) {
         transform(localVBuffer, bLocal, angleRadians);
         // transform() only uses angleRadians argument if local rank is zero
      }

      if (mWriteAnglesStream) {
         auto const ioMPIBlock = getCommunicator()->getIOMPIBlock();
         pvAssert(ioMPIBlock->getRank() == 0);
         int batchElementStart = ioMPIBlock->getStartBatch() * loc->nbatch;
         int batchElementStop  = batchElementStart + ioMPIBlock->getBatchDimension() * loc->nbatch;
         if (bGlobal >= batchElementStart and bGlobal < batchElementStop) {
            mWriteAnglesStream->printf(
                  "t=%f, b=%d, %f\n",
                  simTime,
                  bGlobal,
                  static_cast<double>(angleRadians));
         }
      }
   }
}

void RotateActivityBuffer::copyRandStateToCheckpointData() {
   for (int b = 0; b < getLayerLoc()->nbatch; ++b) {
      int bGlobal = b + getLayerLoc()->kb0;
      mRandStateCheckpointData[4*b + 0] = mRandState->getRNG(bGlobal)->s0;
      mRandStateCheckpointData[4*b + 1] = mRandState->getRNG(bGlobal)->state.s1;
      mRandStateCheckpointData[4*b + 2] = mRandState->getRNG(bGlobal)->state.s2;
      mRandStateCheckpointData[4*b + 3] = mRandState->getRNG(bGlobal)->state.s3;
   }
}

void RotateActivityBuffer::copyCheckpointDataToRandState() {
   for (int b = 0; b < getLayerLoc()->nbatch; ++b) {
      int bGlobal = b + getLayerLoc()->kb0;
      mRandState->getRNG(bGlobal)->s0       = mRandStateCheckpointData[4*b + 0];
      mRandState->getRNG(bGlobal)->state.s1 = mRandStateCheckpointData[4*b + 1];
      mRandState->getRNG(bGlobal)->state.s2 = mRandStateCheckpointData[4*b + 2];
      mRandState->getRNG(bGlobal)->state.s3 = mRandStateCheckpointData[4*b + 3];
      InfoLog().printf(
            "local b = %d, global b = %d, state (%u,(%u,%u,%u))\n", 
            b, bGlobal,
            mRandState->getRNG(bGlobal)->s0,
            mRandState->getRNG(bGlobal)->state.s1,
            mRandState->getRNG(bGlobal)->state.s2,
            mRandState->getRNG(bGlobal)->state.s3);
   }
}

Response::Status RotateActivityBuffer::prepareCheckpointWrite(double simTime) {
      if (getCommunicator()->getLocalMPIBlock()->getRank() != 0) {
         return Response::SUCCESS;
      }
      if (getCommunicator()->getIOMPIBlock()->getRank() == 0) {
         int numProcs = getCommunicator()->getIOMPIBlock()->getBatchDimension();
         std::vector<MPI_Request> mpiRequests(numProcs - 1);
         copyRandStateToCheckpointData();
         for (int m = 1; m < numProcs; ++m) {
            int sourceProc = getCommunicator()->getIOMPIBlock()->calcRankFromRowColBatch(0, 0, m);
            unsigned int *destLoc = &mRandStateCheckpointData[4 * getLayerLoc()->nbatch * m];
            MPI_Irecv(
                  destLoc,
                  4 * getLayerLoc()->nbatch,
                  MPI_UNSIGNED,
                  sourceProc,
                  1111,
                  getCommunicator()->ioCommunicator(),
                  &mpiRequests[m - 1]);
         }
         MPI_Waitall(numProcs - 1, mpiRequests.data(), MPI_STATUSES_IGNORE);
      }
      else {
         copyRandStateToCheckpointData();
         MPI_Send(
               mRandStateCheckpointData.data(),
               4 * getLayerLoc()->nbatch,
               MPI_UNSIGNED,
               0,
               1111,
               getCommunicator()->ioCommunicator());
      }
      return Response::SUCCESS;
}

Response::Status RotateActivityBuffer::processCheckpointRead(double simTime) {
      if (getCommunicator()->getLocalMPIBlock()->getRank() != 0) {
         return Response::SUCCESS;
      }
      if (getCommunicator()->getIOMPIBlock()->getRank() == 0) {
         int numProcs = getCommunicator()->getIOMPIBlock()->getBatchDimension();
         std::vector<MPI_Request> mpiRequests(numProcs - 1);
         for (int m = 1; m < numProcs; ++m) {
            int destProc = getCommunicator()->getIOMPIBlock()->calcRankFromRowColBatch(0, 0, m);
            unsigned int *destLoc = &mRandStateCheckpointData[4 * getLayerLoc()->nbatch * m];
            MPI_Send(
                  destLoc,
                  4 * getLayerLoc()->nbatch,
                  MPI_UNSIGNED,
                  destProc,
                  1111,
                  getCommunicator()->ioCommunicator());
         }
         copyCheckpointDataToRandState();
      }
      else {
         MPI_Recv(
               mRandStateCheckpointData.data(),
               4 * getLayerLoc()->nbatch,
               MPI_UNSIGNED,
               0,
               1111,
               getCommunicator()->ioCommunicator(),
               MPI_STATUS_IGNORE);
         copyCheckpointDataToRandState();
      }
   return Response::SUCCESS;
}

} // namespace PV
