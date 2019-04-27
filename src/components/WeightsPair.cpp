/*
 * WeightsPair.cpp
 *
 *  Created on: Nov 17, 2017
 *      Author: Pete Schultz
 */

#include "WeightsPair.hpp"
#include "io/WeightsFileIO.hpp"
#include "layers/HyPerLayer.hpp"
#include "utils/TransposeWeights.hpp"

namespace PV {

WeightsPair::WeightsPair(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

WeightsPair::~WeightsPair() { delete mOutputStateStream; }

void WeightsPair::initialize(char const *name, PVParams *params, Communicator const *comm) {
   WeightsPairInterface::initialize(name, params, comm);
}

void WeightsPair::initMessageActionMap() {
   WeightsPairInterface::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<ConnectionFinalizeUpdateMessage const>(msgptr);
      return respondConnectionFinalizeUpdate(castMessage);
   };
   mMessageActionMap.emplace("ConnectionFinalizeUpdate", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<ConnectionOutputMessage const>(msgptr);
      return respondConnectionOutput(castMessage);
   };
   mMessageActionMap.emplace("ConnectionOutput", action);
}

void WeightsPair::setObjectType() { mObjectType = "WeightsPair"; }

int WeightsPair::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_writeStep(ioFlag);
   ioParam_initialWriteTime(ioFlag);
   ioParam_writeCompressedWeights(ioFlag);
   ioParam_writeCompressedCheckpoints(ioFlag);
   return PV_SUCCESS;
}

void WeightsPair::ioParam_writeStep(enum ParamsIOFlag ioFlag) {
   bool warnIfAbsent = false; // If not in params, will be set in CommunicateInitInfo stage
   // If writing a derived class that overrides ioParam_writeStep, check if the setDefaultWriteStep
   // method also needs to be overridden.
   parameters()->ioParamValue(ioFlag, name, "writeStep", &mWriteStep, mWriteStep, warnIfAbsent);
}

void WeightsPair::ioParam_initialWriteTime(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "writeStep"));
   if (mWriteStep >= 0) {
      parameters()->ioParamValue(
            ioFlag,
            name,
            "initialWriteTime",
            &mInitialWriteTime,
            mInitialWriteTime,
            true /*warnifabsent*/);
      if (ioFlag == PARAMS_IO_READ) {
         if (mWriteStep > 0 && mInitialWriteTime < 0.0) {
            if (mCommunicator->globalCommRank() == 0) {
               WarnLog(adjustInitialWriteTime);
               adjustInitialWriteTime.printf(
                     "%s: initialWriteTime %f earlier than starting time 0.0.  Adjusting "
                     "initialWriteTime:\n",
                     getDescription_c(),
                     mInitialWriteTime);
               adjustInitialWriteTime.flush();
            }
            while (mInitialWriteTime < 0.0) {
               mInitialWriteTime += mWriteStep; // TODO: this hangs if writeStep is zero.
            }
            if (mCommunicator->globalCommRank() == 0) {
               InfoLog().printf(
                     "%s: initialWriteTime adjusted to %f\n",
                     getDescription_c(),
                     mInitialWriteTime);
            }
         }
         mWriteTime = mInitialWriteTime;
      }
   }
}

void WeightsPair::ioParam_writeCompressedWeights(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "writeStep"));
   if (mWriteStep >= 0) {
      parameters()->ioParamValue(
            ioFlag,
            name,
            "writeCompressedWeights",
            &mWriteCompressedWeights,
            mWriteCompressedWeights,
            true /*warnifabsent*/);
   }
}

void WeightsPair::ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag,
         name,
         "writeCompressedCheckpoints",
         &mWriteCompressedCheckpoints,
         mWriteCompressedCheckpoints,
         true /*warnifabsent*/);
}

Response::Status WeightsPair::respondConnectionFinalizeUpdate(
      std::shared_ptr<ConnectionFinalizeUpdateMessage const> message) {
   finalizeUpdate(message->mTime, message->mDeltaT);
   return Response::SUCCESS;
}

Response::Status
WeightsPair::respondConnectionOutput(std::shared_ptr<ConnectionOutputMessage const> message) {
   outputState(message->mTime);
   return Response::SUCCESS;
}

Response::Status
WeightsPair::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = WeightsPairInterface::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   pvAssert(mConnectionData); // set during WeightsPairInterface::communicateInitInfo()

   mArborList = message->mObjectTable->findObject<ArborList>(getName());
   pvAssert(mArborList);

   if (!mArborList->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the ArborList component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return status + Response::POSTPONE;
   }

   if (mSharedWeights == nullptr) {
      mSharedWeights = message->mObjectTable->findObject<SharedWeights>(getName());
      pvAssert(mSharedWeights);
   }

   if (!mSharedWeights->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the SharedWeights component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return status + Response::POSTPONE;
   }

   if (!parameters()->present(getName(), "writeStep")) {
      setDefaultWriteStep(message);
   }

   return status;
}

void WeightsPair::createPreWeights(std::string const &weightsName) {
   pvAssert(mPreWeights == nullptr and mInitInfoCommunicatedFlag);
   mPreWeights = new Weights(
         weightsName,
         mPatchSize->getPatchSizeX(),
         mPatchSize->getPatchSizeY(),
         mPatchSize->getPatchSizeF(),
         mConnectionData->getPre()->getLayerLoc(),
         mConnectionData->getPost()->getLayerLoc(),
         mArborList->getNumAxonalArbors(),
         mSharedWeights->getSharedWeights(),
         -std::numeric_limits<double>::infinity() /*timestamp*/);
}

void WeightsPair::createPostWeights(std::string const &weightsName) {
   pvAssert(mPostWeights == nullptr and mInitInfoCommunicatedFlag);
   PVLayerLoc const *preLoc  = mConnectionData->getPre()->getLayerLoc();
   PVLayerLoc const *postLoc = mConnectionData->getPost()->getLayerLoc();
   int nxpPre                = mPatchSize->getPatchSizeX();
   int nxpPost               = PatchSize::calcPostPatchSize(nxpPre, preLoc->nx, postLoc->nx);
   int nypPre                = mPatchSize->getPatchSizeY();
   int nypPost               = PatchSize::calcPostPatchSize(nypPre, preLoc->ny, postLoc->ny);
   mPostWeights              = new Weights(
         weightsName,
         nxpPost,
         nypPost,
         preLoc->nf /* number of features in post patch */,
         postLoc,
         preLoc,
         mArborList->getNumAxonalArbors(),
         mSharedWeights->getSharedWeights(),
         -std::numeric_limits<double>::infinity() /*timestamp*/);
}

void WeightsPair::setDefaultWriteStep(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   mWriteStep = message->mDeltaTime;
   // Call ioParamValue to generate the warnIfAbsent warning.
   parameters()->ioParamValue(PARAMS_IO_READ, name, "writeStep", &mWriteStep, mWriteStep, true);
}

void WeightsPair::allocatePreWeights() {
   pvAssert(mPreWeights);
   mPreWeights->setMargins(
         mConnectionData->getPre()->getLayerLoc()->halo,
         mConnectionData->getPost()->getLayerLoc()->halo);
#ifdef PV_USE_CUDA
   mPreWeights->setCudaDevice(mCudaDevice);
#endif // PV_USE_CUDA
   mPreWeights->allocateDataStructures();
}

void WeightsPair::allocatePostWeights() {
   pvAssert(mPostWeights);
   mPostWeights->setMargins(
         mConnectionData->getPost()->getLayerLoc()->halo,
         mConnectionData->getPre()->getLayerLoc()->halo);
#ifdef PV_USE_CUDA
   mPostWeights->setCudaDevice(mCudaDevice);
#endif // PV_USE_CUDA
   mPostWeights->allocateDataStructures();
}

Response::Status
WeightsPair::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = WeightsPairInterface::registerData(message);
   if (status != Response::SUCCESS) {
      return status;
   }
   needPre();
   allocatePreWeights();
   auto *checkpointer = message->mDataRegistry;
   mPreWeights->checkpointWeightPvp(checkpointer, getName(), "W", mWriteCompressedCheckpoints);
   if (mWriteStep >= 0) {
      checkpointer->registerCheckpointData(
            std::string(name),
            "nextWrite",
            &mWriteTime,
            (std::size_t)1,
            true /*broadcast*/,
            false /*not constant*/);

      openOutputStateFile(message);
   }

   return Response::SUCCESS;
}

void WeightsPair::finalizeUpdate(double timestamp, double deltaTime) {
   pvAssert(mPreWeights);
#ifdef PV_USE_CUDA
   mPreWeights->copyToGPU();
#endif // PV_USE_CUDA
   if (mPostWeights) {
      double const timestampPre  = mPreWeights->getTimestamp();
      double const timestampPost = mPostWeights->getTimestamp();
      if (timestampPre > timestampPost) {
         TransposeWeights::transpose(mPreWeights, mPostWeights, mCommunicator);
         mPostWeights->setTimestamp(timestampPre);
      }
#ifdef PV_USE_CUDA
      mPostWeights->copyToGPU();
#endif // PV_USE_CUDA
   }
}

void WeightsPair::openOutputStateFile(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   if (mWriteStep >= 0) {
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
   }
}

Response::Status WeightsPair::readStateFromCheckpoint(Checkpointer *checkpointer) {
   if (getInitializeFromCheckpointFlag()) {
      checkpointer->readNamedCheckpointEntry(
            std::string(name), std::string("W"), !mPreWeights->getWeightsArePlastic());
      return Response::SUCCESS;
   }
   else {
      return Response::NO_ACTION;
   }
}

void WeightsPair::outputState(double timestamp) {
   if ((mWriteStep >= 0) && (timestamp >= mWriteTime)) {
      mWriteTime += mWriteStep;

      WeightsFileIO weightsFileIO(mOutputStateStream, getMPIBlock(), mPreWeights);
      weightsFileIO.writeWeights(timestamp, mWriteCompressedWeights);
   }
   else if (mWriteStep < 0) {
      // If writeStep is negative, we never call writeWeights, but someone might restart from a
      // checkpoint with a different writeStep, so we maintain writeTime.
      mWriteTime = timestamp;
   }
}

} // namespace PV
