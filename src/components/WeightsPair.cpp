/*
 * WeightsPair.cpp
 *
 *  Created on: Nov 17, 2017
 *      Author: Pete Schultz
 */

#include "WeightsPair.hpp"
#include "columns/HyPerCol.hpp"
#include "io/WeightsFileIO.hpp"
#include "layers/HyPerLayer.hpp"
#include "utils/MapLookupByType.hpp"
#include "utils/TransposeWeights.hpp"

namespace PV {

WeightsPair::WeightsPair(char const *name, HyPerCol *hc) { initialize(name, hc); }

WeightsPair::~WeightsPair() { delete mOutputStateStream; }

int WeightsPair::initialize(char const *name, HyPerCol *hc) {
   return WeightsPairInterface::initialize(name, hc);
}

int WeightsPair::setDescription() {
   description.clear();
   description.append("WeightsPair").append(" \"").append(getName()).append("\"");
   return PV_SUCCESS;
}

int WeightsPair::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_writeStep(ioFlag);
   ioParam_initialWriteTime(ioFlag);
   ioParam_writeCompressedWeights(ioFlag);
   ioParam_writeCompressedCheckpoints(ioFlag);
   return PV_SUCCESS;
}

void WeightsPair::ioParam_writeStep(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "writeStep", &mWriteStep, parent->getDeltaTime(), true /*warn if absent */);
}

void WeightsPair::ioParam_initialWriteTime(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "writeStep"));
   if (mWriteStep >= 0) {
      double startTime = parent->getStartTime();
      parent->parameters()->ioParamValue(
            ioFlag, name, "initialWriteTime", &mInitialWriteTime, startTime, true /*warnifabsent*/);
      if (ioFlag == PARAMS_IO_READ) {
         if (mWriteStep > 0 && mInitialWriteTime < startTime) {
            if (parent->getCommunicator()->globalCommRank() == 0) {
               WarnLog(adjustInitialWriteTime);
               adjustInitialWriteTime.printf(
                     "%s: initialWriteTime %f earlier than starting time %f.  Adjusting "
                     "initialWriteTime:\n",
                     getDescription_c(),
                     mInitialWriteTime,
                     startTime);
               adjustInitialWriteTime.flush();
            }
            while (mInitialWriteTime < startTime) {
               mInitialWriteTime += mWriteStep; // TODO: this hangs if writeStep is zero.
            }
            if (parent->getCommunicator()->globalCommRank() == 0) {
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
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "writeStep"));
   if (mWriteStep >= 0) {
      parent->parameters()->ioParamValue(
            ioFlag,
            name,
            "writeCompressedWeights",
            &mWriteCompressedWeights,
            mWriteCompressedWeights,
            true /*warnifabsent*/);
   }
}

void WeightsPair::ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag,
         name,
         "writeCompressedCheckpoints",
         &mWriteCompressedCheckpoints,
         mWriteCompressedCheckpoints,
         true /*warnifabsent*/);
}

int WeightsPair::respond(std::shared_ptr<BaseMessage const> message) {
   int status = WeightsPairInterface::respond(message);
   if (status != PV_SUCCESS) {
      return status;
   }
   else if (
         auto castMessage =
               std::dynamic_pointer_cast<ConnectionFinalizeUpdateMessage const>(message)) {
      return respondConnectionFinalizeUpdate(castMessage);
   }
   else if (auto castMessage = std::dynamic_pointer_cast<ConnectionOutputMessage const>(message)) {
      return respondConnectionOutput(castMessage);
   }
   else {
      return status;
   }
}

int WeightsPair::respondConnectionFinalizeUpdate(
      std::shared_ptr<ConnectionFinalizeUpdateMessage const> message) {
   finalizeUpdate(message->mTime, message->mDeltaT);
   return PV_SUCCESS;
}

int WeightsPair::respondConnectionOutput(std::shared_ptr<ConnectionOutputMessage const> message) {
   outputState(message->mTime);
   return PV_SUCCESS;
}

int WeightsPair::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = WeightsPairInterface::communicateInitInfo(message);
   if (status != PV_SUCCESS) {
      return status;
   }
   pvAssert(mConnectionData); // set during WeightsPairInterface::communicateInitInfo()
   pvAssert(mArborList); // set during WeightsPairInterface::communicateInitInfo()

   if (mSharedWeights == nullptr) {
      mSharedWeights = mapLookupByType<SharedWeights>(message->mHierarchy, getDescription());
   }
   FatalIf(
         mSharedWeights == nullptr,
         "%s requires an SharedWeights component.\n",
         getDescription_c());

   if (!mSharedWeights->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the SharedWeights component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return PV_POSTPONE;
   }

   return status;
}

void WeightsPair::createPreWeights() {
   pvAssert(mPreWeights == nullptr and mInitInfoCommunicatedFlag);
   mPreWeights = new Weights(
         std::string(name),
         mPatchSize->getPatchSizeX(),
         mPatchSize->getPatchSizeY(),
         mPatchSize->getPatchSizeF(),
         mConnectionData->getPre()->getLayerLoc(),
         mConnectionData->getPost()->getLayerLoc(),
         mArborList->getNumAxonalArbors(),
         mSharedWeights->getSharedWeights(),
         -std::numeric_limits<double>::infinity() /*timestamp*/);
}

void WeightsPair::createPostWeights() {
   pvAssert(mPostWeights == nullptr and mInitInfoCommunicatedFlag);
   PVLayerLoc const *preLoc  = mConnectionData->getPre()->getLayerLoc();
   PVLayerLoc const *postLoc = mConnectionData->getPost()->getLayerLoc();
   int nxpPre                = mPatchSize->getPatchSizeX();
   int nxpPost               = PatchSize::calcPostPatchSize(nxpPre, preLoc->nx, postLoc->nx);
   int nypPre                = mPatchSize->getPatchSizeY();
   int nypPost               = PatchSize::calcPostPatchSize(nypPre, preLoc->ny, postLoc->ny);
   mPostWeights              = new Weights(
         std::string(name),
         nxpPost,
         nypPost,
         preLoc->nf /* number of features in post patch */,
         postLoc,
         preLoc,
         mArborList->getNumAxonalArbors(),
         mSharedWeights->getSharedWeights(),
         -std::numeric_limits<double>::infinity() /*timestamp*/);
}

int WeightsPair::allocateDataStructures() {
   if (mPreWeights) {
      allocatePreWeights();
   }
   if (mPostWeights) {
      allocatePostWeights();
   }
   return PV_SUCCESS;
}

void WeightsPair::allocatePreWeights() {
   mPreWeights->setMargins(
         mConnectionData->getPre()->getLayerLoc()->halo,
         mConnectionData->getPost()->getLayerLoc()->halo);
   mPreWeights->allocateDataStructures();
}

void WeightsPair::allocatePostWeights() {
   mPostWeights->setMargins(
         mConnectionData->getPost()->getLayerLoc()->halo,
         mConnectionData->getPre()->getLayerLoc()->halo);
   mPostWeights->allocateDataStructures();
}

int WeightsPair::registerData(Checkpointer *checkpointer) {
   int status = WeightsPairInterface::registerData(checkpointer);
   needPre();
   mPreWeights->checkpointWeightPvp(checkpointer, "W", mWriteCompressedCheckpoints);
   if (status != PV_SUCCESS) {
      return PV_SUCCESS;
   }
   if (mWriteStep >= 0) {
      checkpointer->registerCheckpointData(
            std::string(name),
            "nextWrite",
            &mWriteTime,
            (std::size_t)1,
            true /*broadcast*/,
            false /*not constant*/);

      openOutputStateFile(checkpointer);
   }

   return status;
}

void WeightsPair::finalizeUpdate(double timestamp, double deltaTime) {
   if (mPostWeights) {
      double const timestampPre  = mPreWeights->getTimestamp();
      double const timestampPost = mPostWeights->getTimestamp();
      if (timestampPre > timestampPost) {
         TransposeWeights::transpose(mPreWeights, mPostWeights, parent->getCommunicator());
         mPostWeights->setTimestamp(timestampPre);
      }
   }
}

void WeightsPair::openOutputStateFile(Checkpointer *checkpointer) {
   if (mWriteStep >= 0) {
      if (checkpointer->getMPIBlock()->getRank() == 0) {
         std::string outputStatePath(getName());
         outputStatePath.append(".pvp");

         std::string checkpointLabel(getName());
         checkpointLabel.append("_filepos");

         bool createFlag    = checkpointer->getCheckpointReadDirectory().empty();
         mOutputStateStream = new CheckpointableFileStream(
               outputStatePath.c_str(), createFlag, checkpointer, checkpointLabel);
      }
   }
}

int WeightsPair::readStateFromCheckpoint(Checkpointer *checkpointer) {
   if (mConnectionData->getInitializeFromCheckpointFlag()) {
      checkpointer->readNamedCheckpointEntry(
            std::string(name), std::string("W"), !mPreWeights->getWeightsArePlastic());
   }
   return PV_SUCCESS;
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
