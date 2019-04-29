/*
 * LayerOutputComponent.cpp
 *
 *  Created on: Dec 3, 2018
 *      Author: peteschultz
 */

#include "LayerOutputComponent.hpp"

#include "utils/BufferUtilsMPI.hpp"
#include "utils/BufferUtilsPvp.hpp"

namespace PV {

LayerOutputComponent::LayerOutputComponent(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

LayerOutputComponent::LayerOutputComponent() {}

LayerOutputComponent::~LayerOutputComponent() {
   delete mIOTimer;
   delete mOutputStateStream;
}

void LayerOutputComponent::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   BaseObject::initialize(name, params, comm);
}

void LayerOutputComponent::setObjectType() { mObjectType = "LayerOutputComponent"; }

void LayerOutputComponent::initMessageActionMap() {
   BaseObject::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<LayerOutputStateMessage const>(msgptr);
      return respondLayerOutputState(castMessage);
   };
   mMessageActionMap.emplace("LayerOutputState", action);
}

int LayerOutputComponent::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_writeStep(ioFlag);
   ioParam_initialWriteTime(ioFlag);
   return PV_SUCCESS;
}

void LayerOutputComponent::ioParam_writeStep(enum ParamsIOFlag ioFlag) {
   bool warnIfAbsent = false; // If not in params, will be set in CommunicateInitInfo stage
   // If writing a derived class that overrides ioParam_writeStep, check if the setDefaultWriteStep
   // method also needs to be overridden.
   parameters()->ioParamValue(ioFlag, name, "writeStep", &mWriteStep, mWriteStep, warnIfAbsent);
}

void LayerOutputComponent::ioParam_initialWriteTime(enum ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(name, "writeStep"));
   if (mWriteStep >= 0.0) {
      parameters()->ioParamValue(ioFlag, name, "initialWriteTime", &mInitialWriteTime, 0.0);
      if (ioFlag == PARAMS_IO_READ && mWriteStep > 0.0 && mInitialWriteTime < 0.0) {
         double storeInitialWriteTime = mInitialWriteTime;
         while (mInitialWriteTime < 0.0) {
            mInitialWriteTime += mWriteStep;
         }
         if (mCommunicator->globalCommRank() == 0) {
            WarnLog(warningMessage);
            warningMessage.printf(
                  "%s: initialWriteTime %f is negative.  Adjusting "
                  "initialWriteTime:\n",
                  getDescription_c(),
                  mInitialWriteTime);
            warningMessage.printf(
                  "    initialWriteTime adjusted from %f to %f\n",
                  storeInitialWriteTime,
                  mInitialWriteTime);
         }
      }
   }
}

Response::Status LayerOutputComponent::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   if (!parameters()->present(getName(), "writeStep")) {
      setDefaultWriteStep(message);
   }
   auto status = BaseObject::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   mWriteTime = mInitialWriteTime;
   mPublisher = message->mObjectTable->findObject<BasePublisherComponent>(getName());
   FatalIf(mPublisher == nullptr, "%s requires a BasePublisherComponent.\n", getDescription_c());
   return Response::SUCCESS;
}

void LayerOutputComponent::setDefaultWriteStep(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   mWriteStep = message->mDeltaTime;
   // Call ioParamValue to generate the warnIfAbsent warning.
   parameters()->ioParamValue(PARAMS_IO_READ, name, "writeStep", &mWriteStep, mWriteStep, true);
}

Response::Status LayerOutputComponent::registerData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = BaseObject::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *checkpointer = message->mDataRegistry;
   checkpointer->registerCheckpointData(
         std::string(getName()),
         std::string("nextWrite"),
         &mWriteTime,
         (std::size_t)1,
         true /*broadcast*/,
         false /*not constant*/);
   if (mWriteStep >= 0.0) {
      openOutputStateFile(message);
      if (mPublisher->getSparseLayer()) {
         checkpointer->registerCheckpointData(
               std::string(getName()),
               std::string("numframes_sparse"),
               &mWriteActivitySparseCalls,
               (std::size_t)1,
               true /*broadcast*/,
               false /*not constant*/);
      }
      else {
         checkpointer->registerCheckpointData(
               std::string(getName()),
               std::string("numframes"),
               &mWriteActivityCalls,
               (std::size_t)1,
               true /*broadcast*/,
               false /*not constant*/);
      }
   }

   // Timers

   mIOTimer = new Timer(getName(), "layer", "io     ");
   checkpointer->registerTimer(mIOTimer);

   return Response::SUCCESS;
}

int LayerOutputComponent::openOutputStateFile(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   pvAssert(mWriteStep >= 0);

   auto *checkpointer = message->mDataRegistry;
   if (checkpointer->getMPIBlock()->getRank() == 0) {
      std::string outputStatePath(getName());
      outputStatePath.append(".pvp");

      std::string checkpointLabel(getName());
      checkpointLabel.append("_filepos");

      bool createFlag    = checkpointer->getCheckpointReadDirectory().empty();
      mOutputStateStream = new CheckpointableFileStream(
            outputStatePath.c_str(), createFlag, checkpointer, checkpointLabel);
      mOutputStateStream->respond(message); // CheckpointableFileStream needs to register data
   }
   return PV_SUCCESS;
}

Response::Status LayerOutputComponent::respondLayerOutputState(
      std::shared_ptr<LayerOutputStateMessage const> message) {
   mIOTimer->start();
   auto status = outputState(message->mTime, message->mDeltaTime);
   mIOTimer->stop();
   return status;
}

Response::Status LayerOutputComponent::outputState(double simTime, double deltaTime) {
   if (simTime >= (mWriteTime - (deltaTime / 2)) and mWriteStep >= 0) {
      mWriteTime += mWriteStep;
      PVLayerCube cube = mPublisher->getPublisher()->createCube(0 /*delay*/);
      if (mPublisher->getSparseLayer()) {
         writeActivitySparse(simTime, cube);
      }
      else {
         writeActivity(simTime, cube);
      }
   }
   return Response::SUCCESS;
}

void LayerOutputComponent::writeActivitySparse(double simTime, PVLayerCube &cube) {
   PVLayerLoc const *loc = &cube.loc;
   int const nxExt       = loc->nx + loc->halo.lt + loc->halo.rt;
   int const nyExt       = loc->ny + loc->halo.dn + loc->halo.up;
   int const nf          = loc->nf;
   int const numExtended = nxExt * nyExt * nf;
   pvAssert(cube.numItems == loc->nbatch * numExtended);

   int const mpiBatchDimension = getMPIBlock()->getBatchDimension();
   int const numFrames         = mpiBatchDimension * loc->nbatch;
   for (int frame = 0; frame < numFrames; frame++) {
      int const localBatchIndex = frame % loc->nbatch;
      int const mpiBatchIndex   = frame / loc->nbatch; // Integer division
      pvAssert(mpiBatchIndex * loc->nbatch + localBatchIndex == frame);

      SparseList<float> list;
      auto *activeIndicesBatch   = (SparseList<float>::Entry const *)cube.activeIndices;
      auto *activeIndicesElement = &activeIndicesBatch[localBatchIndex * numExtended];
      for (long int k = 0; k < cube.numActive[localBatchIndex]; k++) {
         SparseList<float>::Entry entry = activeIndicesElement[k];
         int index                      = (int)entry.index;

         // Location is local extended; need global restricted.
         // Get local restricted coordinates.
         int x = kxPos(index, nxExt, nyExt, nf) - loc->halo.lt;
         if (x < 0 or x >= loc->nx) {
            continue;
         }
         int y = kyPos(index, nxExt, nyExt, nf) - loc->halo.up;
         if (y < 0 or y >= loc->ny) {
            continue;
         }
         // Convert to global restricted coordinates.
         x += loc->kx0;
         y += loc->ky0;
         int f = featureIndex(index, nxExt, nyExt, nf);

         // Get global restricted index.
         entry.index = (uint32_t)kIndex(x, y, f, loc->nxGlobal, loc->nyGlobal, nf);
         list.addEntry(entry);
      }
      auto gatheredList =
            BufferUtils::gatherSparse(getMPIBlock(), list, mpiBatchIndex, 0 /*root process*/);
      if (getMPIBlock()->getRank() == 0) {
         long fpos = mOutputStateStream->getOutPos();
         if (fpos == 0L) {
            BufferUtils::ActivityHeader header = BufferUtils::buildSparseActivityHeader<float>(
                  loc->nx * getMPIBlock()->getNumColumns(),
                  loc->ny * getMPIBlock()->getNumRows(),
                  loc->nf,
                  0 /* numBands */); // numBands will be set by call to incrementNBands.
            header.timestamp = simTime;
            BufferUtils::writeActivityHeader(*mOutputStateStream, header);
         }
         BufferUtils::writeSparseFrame(*mOutputStateStream, &gatheredList, simTime);
      }
   }
   mWriteActivitySparseCalls += numFrames;
   updateNBands(mWriteActivitySparseCalls);
}

// write non-spiking activity
void LayerOutputComponent::writeActivity(double simTime, PVLayerCube &cube) {
   PVLayerLoc const *loc = &cube.loc;

   int const nxExtLocal  = loc->nx + loc->halo.lt + loc->halo.rt;
   int const nyExtLocal  = loc->ny + loc->halo.dn + loc->halo.up;
   int const nf          = loc->nf;
   int const numExtLocal = nxExtLocal * nyExtLocal * nf;
   pvAssert(cube.numItems == loc->nbatch * numExtLocal);

   int const mpiBatchDimension = getMPIBlock()->getBatchDimension();
   int const numFrames         = mpiBatchDimension * loc->nbatch;
   for (int frame = 0; frame < numFrames; frame++) {
      int const localBatchIndex = frame % loc->nbatch;
      int const mpiBatchIndex   = frame / loc->nbatch; // Integer division
      pvAssert(mpiBatchIndex * loc->nbatch + localBatchIndex == frame);

      float const *data = &cube.data[localBatchIndex * numExtLocal];
      Buffer<float> localBuffer(data, nxExtLocal, nyExtLocal, nf);
      localBuffer.crop(loc->nx, loc->ny, Buffer<float>::CENTER);
      Buffer<float> blockBuffer = BufferUtils::gather<float>(
            getMPIBlock(), localBuffer, loc->nx, loc->ny, mpiBatchIndex, 0 /*root process*/);
      // At this point, the rank-zero process has the entire block for the batch element,
      // regardless of what the mpiBatchIndex is.
      if (getMPIBlock()->getRank() == 0) {
         long fpos = mOutputStateStream->getOutPos();
         if (fpos == 0L) {
            BufferUtils::ActivityHeader header = BufferUtils::buildActivityHeader<float>(
                  loc->nx * getMPIBlock()->getNumColumns(),
                  loc->ny * getMPIBlock()->getNumRows(),
                  loc->nf,
                  0 /* numBands */); // numBands will be set by call to incrementNBands.
            header.timestamp = simTime;
            BufferUtils::writeActivityHeader(*mOutputStateStream, header);
         }
         BufferUtils::writeFrame<float>(*mOutputStateStream, &blockBuffer, simTime);
      }
   }
   mWriteActivityCalls += numFrames;
   updateNBands(mWriteActivityCalls);
}

void LayerOutputComponent::updateNBands(int const numCalls) {
   // Only the root process needs to maintain INDEX_NBANDS, so only the root process modifies
   // numCalls. This way, writeActivityCalls does not need to be coordinated across MPI
   if (mOutputStateStream != nullptr) {
      long int fpos = mOutputStateStream->getOutPos();
      mOutputStateStream->setOutPos(sizeof(int) * INDEX_NBANDS, true /*fromBeginning*/);
      mOutputStateStream->write(&numCalls, (long)sizeof(numCalls));
      mOutputStateStream->setOutPos(fpos, true /*fromBeginning*/);
   }
}

} // namespace PV
