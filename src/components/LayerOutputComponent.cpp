/*
 * LayerOutputComponent.cpp
 *
 *  Created on: Dec 3, 2018
 *      Author: peteschultz
 */

#include "LayerOutputComponent.hpp"
#include "checkpointing/CheckpointEntryFilePosition.hpp"

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
   auto *objectTable = message->mObjectTable;
   mLayerGeometry = objectTable->findObject<LayerGeometry>(getName());
   FatalIf(mLayerGeometry == nullptr, "%s requires a LayerGeometry.\n", getDescription_c());
   mPublisher     = objectTable->findObject<BasePublisherComponent>(getName());
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
   auto outputFileManager = getCommunicator()->getOutputFileManager();
   std::string outputStatePath(getName());
   outputStatePath.append(".pvp");

   // If the file exists and CheckpointReadDirectory is empty, we need to
   // clobber the file.
   if (checkpointer->getCheckpointReadDirectory().empty()) {
      outputFileManager->open(
            outputStatePath, std::ios_base::out, checkpointer->doesVerifyWrites());
   }

   PVLayerLoc const *loc = mLayerGeometry->getLayerLoc();
   if (mPublisher->getSparseLayer()) {
      mDenseFile  = nullptr;
      mSparseFile = std::make_shared<SparseLayerFile>(
            outputFileManager,
            outputStatePath,
            *loc,
            true /*dataExtendedFlag*/,
            false /*fileExtendedFlag*/,
            false /*readOnlyFlag*/,
            checkpointer->doesVerifyWrites());
      mSparseListVector.resize(loc->nbatch);
      mSparseFile->respond(message); // SparseLayerFile needs to register data
   }
   else {
      mDenseFile = std::make_shared<LayerFile>(
            outputFileManager,
            outputStatePath,
            *loc,
            true /*dataExtendedFlag*/,
            false /*fileExtendedFlag*/,
            false /*readOnlyFlag*/,
            checkpointer->doesVerifyWrites());
      mDenseFile->respond(message); // LayerFile needs to register data
      mSparseFile  = nullptr;
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
      if (mSparseFile) {
         pvAssert(!mDenseFile);
         writeActivitySparse(simTime, cube);
      }
      else {
         pvAssert(mDenseFile);
         writeActivity(simTime, cube);
      }
   }
   return Response::SUCCESS;
}

void LayerOutputComponent::writeActivitySparse(double simTime, PVLayerCube &cube) {
   pvAssert(mSparseFile);

   PVLayerLoc const *loc = &cube.loc;
   pvAssert(loc->nbatch == mLayerGeometry->getLayerLoc()->nbatch);
   // Should check that other fields of loc agree with mLayerGeometry->getLayerLoc().
   int const nxExtLocal  = loc->nx + loc->halo.lt + loc->halo.rt;
   int const nyExtLocal  = loc->ny + loc->halo.dn + loc->halo.up;
   int const nf          = loc->nf;
   int const numExtLocal = nxExtLocal * nyExtLocal * nf;
   pvAssert(cube.numItems == loc->nbatch * numExtLocal);

   for (int b = 0; b < loc->nbatch; ++b) {
      auto *activeIndicesBatch   = (SparseList<float>::Entry const *)cube.activeIndices;
      auto *activeIndicesElement = &activeIndicesBatch[b * numExtLocal];
      mSparseListVector[b].reset(nxExtLocal, nyExtLocal, nf);

      for (long int k = 0; k < cube.numActive[b]; k++) {
         SparseList<float>::Entry const &entry = activeIndicesElement[k];
         mSparseListVector[b].addEntry(entry);
      }
      mSparseFile->setListLocation(&mSparseListVector[b], b);
   }
   mSparseFile->write(simTime);
   int blockBatchDimension = getCommunicator()->getIOMPIBlock()->getBatchDimension();
   mWriteActivitySparseCalls += blockBatchDimension * loc->nbatch;
}

// write non-spiking activity
void LayerOutputComponent::writeActivity(double simTime, PVLayerCube &cube) {
   pvAssert(mDenseFile);

   PVLayerLoc const *loc = &cube.loc;
   // Should check that this is the same as mLayerGeometry->getLayerLoc()
   int const nxExtLocal  = loc->nx + loc->halo.lt + loc->halo.rt;
   int const nyExtLocal  = loc->ny + loc->halo.dn + loc->halo.up;
   int const nf          = loc->nf;
   int const numExtLocal = nxExtLocal * nyExtLocal * nf;
   pvAssert(cube.numItems == loc->nbatch * numExtLocal);

   std::vector<float> activity(cube.numItems);
   for (int n = 0; n < cube.numItems; ++n) {
      activity[n] = cube.data[n];
   }
   for (int b = 0; b < loc->nbatch; ++b) {
      mDenseFile->setDataLocation(&activity[b * numExtLocal], b);
   }
   mDenseFile->write(simTime);
   int blockBatchDimension = getCommunicator()->getIOMPIBlock()->getBatchDimension();
   mWriteActivityCalls += blockBatchDimension * loc->nbatch;
}

} // namespace PV
