/*
 * WeightsPair.cpp
 *
 *  Created on: Nov 17, 2017
 *      Author: Pete Schultz
 */

#include "WeightsPair.hpp"
#include "components/LayerGeometry.hpp"
#include "io/BroadcastPreWeightsFile.hpp"
#include "io/LocalPatchWeightsFile.hpp"
#include "io/SharedWeightsFile.hpp"
#include "layers/HyPerLayer.hpp"
#include "utils/TransposeWeights.hpp"

namespace PV {

WeightsPair::WeightsPair(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

WeightsPair::~WeightsPair() {}

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
   parameters()->ioParamValue(ioFlag, getName(), "writeStep", &mWriteStep, mWriteStep, warnIfAbsent);
}

void WeightsPair::ioParam_initialWriteTime(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(getName(), "writeStep"));
   if (mWriteStep >= 0) {
      parameters()->ioParamValue(
            ioFlag,
            getName(),
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
   pvAssert(!parameters()->presentAndNotBeenRead(getName(), "writeStep"));
   if (mWriteStep >= 0) {
      parameters()->ioParamValue(
            ioFlag,
            getName(),
            "writeCompressedWeights",
            &mWriteCompressedWeights,
            mWriteCompressedWeights,
            true /*warnifabsent*/);
   }
}

void WeightsPair::ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag,
         getName(),
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

Weights::WeightsType WeightsPair::calcWeightsType(HyPerLayer *pre, HyPerLayer *post) {
   Weights::WeightsType weightsType;
   if (mSharedWeights->getSharedWeightsFlag()) {
      weightsType = Weights::WeightsType::SHARED;
   }
   else {
      auto *preGeometry = pre->getComponentByType<LayerGeometry>();
      bool preIsBroadcast = preGeometry->getBroadcastFlag();
      if (preIsBroadcast) {
         weightsType = Weights::WeightsType::BROADCASTPRE;
      }
      else {
         weightsType = Weights::WeightsType::LOCALPATCH;
      }
   }
   return weightsType;
}

void WeightsPair::createPreWeights(std::string const &weightsName) {
   pvAssert(mPreWeights == nullptr and mInitInfoCommunicatedFlag);
   HyPerLayer *pre  = mConnectionData->getPre();
   HyPerLayer *post = mConnectionData->getPost();
   Weights::WeightsType weightsType = calcWeightsType(pre, post);
   mPreWeights = new Weights(
         weightsName,
         mPatchSize->getPatchSizeX(),
         mPatchSize->getPatchSizeY(),
         mPatchSize->getPatchSizeF(),
         mConnectionData->getPre()->getLayerLoc(),
         mConnectionData->getPost()->getLayerLoc(),
         mArborList->getNumAxonalArbors(),
         weightsType,
         std::numeric_limits<double>::lowest() /*timestamp, set to value "close to" -infinity*/);
}

void WeightsPair::createPostWeights(std::string const &weightsName) {
   pvAssert(mPostWeights == nullptr and mInitInfoCommunicatedFlag);
   HyPerLayer *pre  = mConnectionData->getPre();
   HyPerLayer *post = mConnectionData->getPost();
   auto *preLoc     = pre->getLayerLoc();
   auto *postLoc    = post->getLayerLoc();
   int nxpPre       = mPatchSize->getPatchSizeX();
   int nxpPost      = PatchSize::calcPostPatchSize(nxpPre, preLoc->nx, postLoc->nx);
   int nypPre       = mPatchSize->getPatchSizeY();
   int nypPost      = PatchSize::calcPostPatchSize(nypPre, preLoc->ny, postLoc->ny);
   Weights::WeightsType weightsType = calcWeightsType(post, pre);
   mPostWeights = new Weights(
         weightsName,
         nxpPost,
         nypPost,
         preLoc->nf /* number of features in post patch */,
         postLoc,
         preLoc,
         mArborList->getNumAxonalArbors(),
         weightsType,
         std::numeric_limits<double>::lowest() /*timestamp, set to value "close to" -infinity*/);
}

void WeightsPair::setDefaultWriteStep(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   mWriteStep = message->mDeltaTime;
   // Call ioParamValue to generate the warnIfAbsent warning.
   parameters()->ioParamValue(PARAMS_IO_READ, getName(), "writeStep", &mWriteStep, mWriteStep, true);
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
            std::string(getName()),
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
   if (mWriteStep < 0) { return; }

   auto *checkpointer = message->mDataRegistry;
   auto outputFileManager = getCommunicator()->getOutputFileManager();
   std::string outputStatePath(getName());
   outputStatePath.append(".pvp");

   switch (mPreWeights->getWeightsType()) {
      case Weights::WeightsType::SHARED:
          mWeightsFile = std::make_shared<SharedWeightsFile>(
                outputFileManager,
                outputStatePath,
                mPreWeights->getData(),
                getWriteCompressedWeights(),
                false /*readOnlyFlag*/,
                checkpointer->getCheckpointReadDirectory().empty() /*clobberFlag*/,
                checkpointer->doesVerifyWrites());
          break;
      case Weights::WeightsType::LOCALPATCH:
         mWeightsFile = std::make_shared<LocalPatchWeightsFile>(
               outputFileManager,
               outputStatePath,
               mPreWeights->getData(),
               mConnectionData->getPre()->getLayerLoc(),
               mConnectionData->getPost()->getLayerLoc(),
               true /*fileExtendedFlag*/,
               getWriteCompressedWeights(),
               false /*readOnlyFlag*/,
               checkpointer->getCheckpointReadDirectory().empty() /*clobberFlag*/,
               checkpointer->doesVerifyWrites());
          break;
      case Weights::WeightsType::BROADCASTPRE:
         mWeightsFile = std::make_shared<BroadcastPreWeightsFile>(
               outputFileManager,
               outputStatePath,
               mPreWeights->getData(),
               mConnectionData->getPre()->getLayerLoc()->nf,
               getWriteCompressedWeights(),
               false /*readOnlyFlag*/,
               checkpointer->getCheckpointReadDirectory().empty() /*clobberFlag*/,
               checkpointer->doesVerifyWrites());
          break;
      default:
         Fatal().printf("Unrecognized WeightsType %d\n", mPreWeights->getWeightsType());
         break;
   }
   mWeightsFile->respond(message); // WeightsFile needs to register filepos
}

Response::Status WeightsPair::readStateFromCheckpoint(Checkpointer *checkpointer) {
   if (getInitializeFromCheckpointFlag()) {
      checkpointer->readNamedCheckpointEntry(
            std::string(getName()), std::string("W"), !mPreWeights->getWeightsArePlastic());
      return Response::SUCCESS;
   }
   else {
      return Response::NO_ACTION;
   }
}

void WeightsPair::outputState(double timestamp) {
   if ((mWriteStep >= 0) && (timestamp >= mWriteTime)) {
      mWriteTime += mWriteStep;
      mWeightsFile->write(timestamp);
   }
   else if (mWriteStep < 0) {
      // If writeStep is negative, we never call writeWeights, but someone might restart from a
      // checkpoint with a different writeStep, so we maintain writeTime.
      mWriteTime = timestamp;
   }
}

} // namespace PV
