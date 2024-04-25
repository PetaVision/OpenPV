#include "ScaleXActivityBuffer.hpp"
#include "checkpointing/CheckpointEntryFilePosition.hpp"
#include "io/FileManager.hpp"
#include "io/FileStreamBuilder.hpp"
#include "utils/BufferUtilsMPI.hpp"
#include "utils/Interpolate.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include <algorithm> // copy()
#include <cmath>
#include <utility>   // swap()

namespace PV {

ScaleXActivityBuffer::ScaleXActivityBuffer(
      char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

void ScaleXActivityBuffer::initialize(
      char const *name, PVParams *params, Communicator const *comm) {
   HyPerActivityBuffer::initialize(name, params, comm);
}

int ScaleXActivityBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerActivityBuffer::ioParamsFillGroup(ioFlag);
   if (status != PV_SUCCESS) {
      return status;
   }

   ioParam_scaleFactorMin(ioFlag);
   ioParam_scaleFactorMax(ioFlag);
   if (ioFlag == PARAMS_IO_READ and mScaleFactorMax < mScaleFactorMin) {
      WarnLog().printf(
            "%s: specified value of max (%f) is less than specified value of min (%f)\n",
            getDescription_c(),
            static_cast<double>(mScaleFactorMax),
            static_cast<double>(mScaleFactorMin));
      std::swap(mScaleFactorMax, mScaleFactorMin);
      WarnLog().printf(
            "Switching scaleFactorMin to %f and scaleFactorMax to %f\n",
            static_cast<double>(mScaleFactorMax),
            static_cast<double>(mScaleFactorMin));
   }
   FatalIf(
         mScaleFactorMin < 0.0f and mScaleFactorMax > 0.0f,
         "%s interval [scaleFactorMin, scaleFactorMax] cannot contain zero "
         "(specified values [%f, %f]\n",
         getDescription_c(), (double)mScaleFactorMin, (double)mScaleFactorMax);
   float minScaleAllowed = 0.01f;
   FatalIf(
         mScaleFactorMin < minScaleAllowed,
         "%s parameter scaleFactorMin is too close to zero (value %f; minimum allowed %f)\n",
         getDescription_c(), (double)mScaleFactorMin, (double)minScaleAllowed);
   pvAssert(mScaleFactorMax >= mScaleFactorMin);
   ioParam_writeScaleFactorsFile(ioFlag);
   return status;
}

void ScaleXActivityBuffer::ioParam_scaleFactorMin(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValueRequired(ioFlag, getName(), "scaleFactorMin", &mScaleFactorMin);
}

void ScaleXActivityBuffer::ioParam_scaleFactorMax(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValueRequired(ioFlag, getName(), "scaleFactorMax", &mScaleFactorMax);
}

void ScaleXActivityBuffer::ioParam_writeScaleFactorsFile(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamString(
         ioFlag,
         getName(),
         "writeScaleFactorsFile",
         &mWriteScaleFactorsFile,
         mWriteScaleFactorsFile);
}

void ScaleXActivityBuffer::setObjectType() { mObjectType = "ScaleXActivityBuffer"; }

Response::Status ScaleXActivityBuffer::allocateDataStructures() {
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

Response::Status ScaleXActivityBuffer::registerData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = HyPerActivityBuffer::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *checkpointer = message->mDataRegistry;
   if (getCommunicator()->getIOMPIBlock()->getRank() == 0) {
      if (mWriteScaleFactorsFile) {
         auto fileManager = getCommunicator()->getOutputFileManager();
         fileManager->ensureDirectoryExists(".");
         mWriteScaleFactorsStream = FileStreamBuilder(
               fileManager,
               mWriteScaleFactorsFile,
               true /*textFlag*/,
               false /*readOnlyFlag*/,
               checkpointer->getCheckpointReadDirectory().empty() /*clobberFlag*/,
               checkpointer->doesVerifyWrites())
               .get();
         auto checkpointEntry = std::make_shared<CheckpointEntryFilePosition>(
               getName(), std::string("WriteScaleFactorsFile"), mWriteScaleFactorsStream);
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

void ScaleXActivityBuffer::applyTransformCPU(
      Buffer<float> const &inputBuffer, Buffer<float> &outputBuffer, float scaleFactor) {
   int const nxRes = inputBuffer.getWidth();
   int const nxExt = outputBuffer.getWidth();
   int const xMargin = (nxExt - nxRes) / 2; // assumes left and right margins are equal
   int const nyRes = inputBuffer.getHeight();
   int const nyExt = outputBuffer.getHeight();
   int const yMargin = (nyExt - nyRes) / 2; // assumes bottom and top margins are equal
   int const nf = inputBuffer.getFeatures();

   float const xCenter = 0.5f * static_cast<float>(nxRes - 1);

   int const numExtended = outputBuffer.getTotalElements();
   float recipScale = 1.0f / scaleFactor;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
   for (int kExt = 0; kExt < numExtended; ++kExt) {
      int const kxExt = kxPos(kExt, nxExt, nyExt, nf);
      float xExt = static_cast<float>(kxExt - xMargin) - xCenter;
      float xSrc = recipScale * xExt + xCenter;

      int const kyExt = kyPos(kExt, nxExt, nyExt, nf);
      float ySrc = static_cast<float>(kyExt - yMargin);

      int const kf = featureIndex(kExt, nxExt, nyExt, nf);

      float result = interpolate(inputBuffer, xSrc, ySrc, kf);
      outputBuffer.set(kExt, result);
   }
}

void ScaleXActivityBuffer::transform(Buffer<float> &localVBuffer, int bLocal, float scaleFactor) {
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
      // Set activityBuffer buffer to size of global extended buffer, to hold result of scaling
      int const nxGlobalExt = loc->nxGlobal + loc->halo.lt + loc->halo.rt;
      int const nyGlobalExt = loc->nyGlobal + loc->halo.dn + loc->halo.up;
      activityBuffer.resize(nxGlobalExt, nyGlobalExt, loc->nf);
      applyTransformCPU(globalV, activityBuffer, scaleFactor);
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

void ScaleXActivityBuffer::updateBufferCPU(double simTime, double deltaTime) {
   PVLayerLoc const *loc = getLayerLoc();
   Buffer<float> localVBuffer(loc->nx, loc->ny, loc->nf);  
   float scaleFactor = 0.0f;
   for (int bGlobal = 0; bGlobal < loc->nbatchGlobal; ++bGlobal) {
      if (getCommunicator()->getLocalMPIBlock()->getRank() == 0) {
         scaleFactor = mRandState->uniformRandom(bGlobal, mScaleFactorMin, mScaleFactorMax);
      }
      int bLocal = bGlobal - loc->kb0;
      if (bLocal >= 0 and bLocal < loc->nbatch) {
         transform(localVBuffer, bLocal, scaleFactor);
         // transform() only uses scaleFactor argument if local rank is zero
      }

      if (mWriteScaleFactorsStream) {
         auto const ioMPIBlock = getCommunicator()->getIOMPIBlock();
         pvAssert(ioMPIBlock->getRank() == 0);
         int batchElementStart = ioMPIBlock->getStartBatch() * loc->nbatch;
         int batchElementStop  = batchElementStart + ioMPIBlock->getBatchDimension() * loc->nbatch;
         if (bGlobal >= batchElementStart and bGlobal < batchElementStop) {
            mWriteScaleFactorsStream->printf(
                  "t=%f, b=%d, %f\n", simTime, bGlobal, static_cast<double>(scaleFactor));
         }
      }
   }
}

void ScaleXActivityBuffer::copyRandStateToCheckpointData() {
   for (int b = 0; b < getLayerLoc()->nbatch; ++b) {
      int bGlobal = b + getLayerLoc()->kb0;
      mRandStateCheckpointData[4*b + 0] = mRandState->getRNG(bGlobal)->s0;
      mRandStateCheckpointData[4*b + 1] = mRandState->getRNG(bGlobal)->state.s1;
      mRandStateCheckpointData[4*b + 2] = mRandState->getRNG(bGlobal)->state.s2;
      mRandStateCheckpointData[4*b + 3] = mRandState->getRNG(bGlobal)->state.s3;
   }
}

void ScaleXActivityBuffer::copyCheckpointDataToRandState() {
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

Response::Status ScaleXActivityBuffer::prepareCheckpointWrite(double simTime) {
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

Response::Status ScaleXActivityBuffer::processCheckpointRead(double simTime) {
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
