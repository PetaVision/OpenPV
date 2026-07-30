/*
 * LayerOutputComponent.cpp
 *
 *  Created on: Dec 3, 2018
 *      Author: peteschultz
 */

#include "LayerOutputComponent.hpp"

namespace PV {

LayerOutputComponent::LayerOutputComponent(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

LayerOutputComponent::LayerOutputComponent() {}

LayerOutputComponent::~LayerOutputComponent() {
   delete mInitialIOTimer;
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
   parameters()->ioParamValue(ioFlag, getName(), "writeStep", &mWriteStep, mWriteStep, warnIfAbsent);
}

void LayerOutputComponent::ioParam_initialWriteTime(enum ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(getName(), "writeStep"));
   if (mWriteStep >= 0.0) {
      parameters()->ioParamValue(ioFlag, getName(), "initialWriteTime", &mInitialWriteTime, 0.0);
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
   parameters()->ioParamValue(PARAMS_IO_READ, getName(), "writeStep", &mWriteStep, mWriteStep, true);
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

   mInitialIOTimer = new Timer(getName(), "layer", "initialio ");
   checkpointer->registerTimer(mInitialIOTimer);
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

   mDenseFile            = nullptr;
   mDenseBroadcastFile   = nullptr;
   mSparseFile           = nullptr;
   mSparseBroadcastFile  = nullptr;
   PVLayerLoc const *loc = mLayerGeometry->getLayerLoc();
   bool broadcastFlag    = mLayerGeometry->getBroadcastFlag();
   bool sparseLayerFlag  = mPublisher->getSparseLayerFlag();
   if (sparseLayerFlag /* TODO implement sparse broadcast layer files */) {
      if (broadcastFlag) {
         mSparseBroadcastFile = std::make_shared<SparseBroadcastLayerFile>(
               outputFileManager,
               outputStatePath,
               loc->nf,
               loc->nbatch,
               false /*readOnlyFlag*/,
               checkpointer->getCheckpointReadDirectory().empty() /*clobberFlag*/,
               checkpointer->doesVerifyWrites());
         mSparseListVector.resize(loc->nbatch);
         mSparseBroadcastFile->respond(message); // SparseBroadcastLayerFile needs to register data
      }
      else {
         mSparseFile = std::make_shared<SparseLayerFile>(
               outputFileManager,
               outputStatePath,
               *loc,
               true /*dataExtendedFlag*/,
               false /*fileExtendedFlag*/,
               false /*readOnlyFlag*/,
               checkpointer->getCheckpointReadDirectory().empty() /*clobberFlag*/,
               checkpointer->doesVerifyWrites());
         mSparseListVector.resize(loc->nbatch);
         mSparseFile->respond(message); // SparseLayerFile needs to register data
      }
   }
   else {
      if (broadcastFlag) {
         mDenseBroadcastFile = std::make_shared<BroadcastLayerFile>(
               outputFileManager,
               outputStatePath,
               loc->nf,
               loc->nbatch,
               false /*readOnlyFlag*/,
               checkpointer->getCheckpointReadDirectory().empty() /*clobberFlag*/,
               checkpointer->doesVerifyWrites());
         mDenseBroadcastFile->respond(message); // BroadcastLayerFile needs to register data
      }
      else {
         mDenseFile = std::make_shared<LayerFile>(
               outputFileManager,
               outputStatePath,
               *loc,
               true /*dataExtendedFlag*/,
               false /*fileExtendedFlag*/,
               false /*readOnlyFlag*/,
               checkpointer->getCheckpointReadDirectory().empty() /*clobberFlag*/,
               checkpointer->doesVerifyWrites());
         mDenseFile->respond(message); // LayerFile needs to register data
      }
   }
   return PV_SUCCESS;
}

Response::Status LayerOutputComponent::respondLayerOutputState(
      std::shared_ptr<LayerOutputStateMessage const> message) {
   Timer *timer = message->mTime ? mIOTimer : mInitialIOTimer;
   timer->start();
   auto status = outputState(message->mTime, message->mDeltaTime);
   timer->stop();
   return status;
}

Response::Status LayerOutputComponent::outputState(double simTime, double deltaTime) {
   if (simTime >= (mWriteTime - (deltaTime / 2)) and mWriteStep >= 0) {
      mWriteTime += mWriteStep;
      PVLayerCube cube = mPublisher->getPublisher()->createCube(0 /*delay*/);
      if (mSparseFile) {
         pvAssert(!(mDenseFile or mDenseBroadcastFile or mSparseBroadcastFile));
         writeActivitySparse(simTime, cube);
      }
      else if (mSparseBroadcastFile) {
         pvAssert(!(mDenseFile or mDenseBroadcastFile or mSparseFile));
         writeActivitySparseBroadcast(simTime, cube);
      }
      else if (mDenseFile) {
         pvAssert(!(mDenseBroadcastFile or mSparseFile or mSparseBroadcastFile));
         writeActivityDense(simTime, cube);
      }
      else if (mDenseBroadcastFile) {
         pvAssert(!(mDenseFile or mSparseFile or mSparseBroadcastFile));
         writeActivityDenseBroadcast(simTime, cube);
      }
   }
   return Response::SUCCESS;
}

// write sparse, non-broadcast activity
void LayerOutputComponent::writeActivitySparse(double simTime, PVLayerCube &cube) {
   pvAssert(mSparseFile);

   PVLayerLoc const *loc = &cube.loc;
   pvAssert(loc->nbatch == mLayerGeometry->getLayerLoc()->nbatch);
   // Should check that other fields of loc agree with mLayerGeometry->getLayerLoc().
   int const nxExtLocal   = loc->nx + loc->halo.lt + loc->halo.rt;
   int const nyExtLocal   = loc->ny + loc->halo.dn + loc->halo.up;
   int const nf           = loc->nf;
   long const numExtLocal = (long)nxExtLocal * (long)nyExtLocal * (long)nf;
   pvAssert(cube.numItems == loc->nbatch * numExtLocal);

   for (int b = 0; b < loc->nbatch; ++b) {
      auto *activeIndicesBatch   = (SparseList<float>::Entry const *)cube.activeIndices;
      auto *activeIndicesElement = &activeIndicesBatch[b * numExtLocal];
      mSparseListVector[b].reset(nxExtLocal, nyExtLocal, nf);

      for (long k = 0; k < cube.numActive[b]; k++) {
         SparseList<float>::Entry const &entry = activeIndicesElement[k];
         mSparseListVector[b].addEntry(entry);
      }
      mSparseFile->setListLocation(&mSparseListVector[b], b);
   }
   mSparseFile->write(simTime);
   int blockBatchDimension = getCommunicator()->getIOMPIBlock()->getBatchDimension();
   mWriteActivitySparseCalls += blockBatchDimension * loc->nbatch;
}

// write sparse, broadcast activity
void LayerOutputComponent::writeActivitySparseBroadcast(double simTime, PVLayerCube &cube) {
   pvAssert(mSparseBroadcastFile);
   PVLayerLoc const *loc = &cube.loc;
   int const nf          = loc->nf;
   int const nbatch      = loc->nbatch;
   pvAssert(cube.numItems == nbatch * nf);
   for (int b = 0; b < loc->nbatch; ++b) {
      auto *activeIndicesBatch   = (SparseList<float>::Entry const *)cube.activeIndices;
      auto *activeIndicesElement = &activeIndicesBatch[b * nf];
      mSparseListVector[b].reset(1, 1, nf);

      for (long k = 0; k < cube.numActive[b]; k++) {
         SparseList<float>::Entry const &entry = activeIndicesElement[k];
         mSparseListVector[b].addEntry(entry);
      }
      mSparseBroadcastFile->setListLocation(&mSparseListVector[b], b);
   }
   mSparseBroadcastFile->write(simTime);
   int blockBatchDimension = getCommunicator()->getIOMPIBlock()->getBatchDimension();
   mWriteActivitySparseCalls += blockBatchDimension * loc->nbatch;
}

// write non-sparse, nonbroadcast activity
void LayerOutputComponent::writeActivityDense(double simTime, PVLayerCube &cube) {
   pvAssert(mDenseFile);

   PVLayerLoc const *loc = &cube.loc;
   // Should check that this is the same as mLayerGeometry->getLayerLoc()
   int const nxExtLocal   = loc->nx + loc->halo.lt + loc->halo.rt;
   int const nyExtLocal   = loc->ny + loc->halo.dn + loc->halo.up;
   int const nf           = loc->nf;
   long const numExtLocal = (long)nxExtLocal * (long)nyExtLocal * (long)nf;
   pvAssert(cube.numItems == loc->nbatch * numExtLocal);

   std::vector<float> activity(cube.numItems);
   for (long n = 0; n < cube.numItems; ++n) {
      activity[n] = cube.data[n];
   }
   for (int b = 0; b < loc->nbatch; ++b) {
      mDenseFile->setDataLocation(&activity[b * numExtLocal], b);
   }
   mDenseFile->write(simTime);
   int blockBatchDimension = getCommunicator()->getIOMPIBlock()->getBatchDimension();
   mWriteActivityCalls += blockBatchDimension * loc->nbatch;
}

// write non-sparse, broadcast activity
void LayerOutputComponent::writeActivityDenseBroadcast(double simTime, PVLayerCube &cube) {
   pvAssert(mDenseBroadcastFile);

   PVLayerLoc const *loc = &cube.loc;
   int const nf          = loc->nf;
   int const nbatch      = loc->nbatch;
   pvAssert(cube.numItems == nbatch * nf);

   std::vector<float> activity(cube.numItems);
   for (int n = 0; n < cube.numItems; ++n) {
      activity[n] = cube.data[n];
   }
   for (int b = 0; b < nbatch; ++b) {
      mDenseBroadcastFile->setDataLocation(&activity[b * nf], b);
   }
   mDenseBroadcastFile->write(simTime);
   int blockBatchDimension = getCommunicator()->getIOMPIBlock()->getBatchDimension();
   mWriteActivityCalls += blockBatchDimension * nbatch;
}

} // namespace PV
