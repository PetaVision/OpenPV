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

namespace PV {

WeightsPair::WeightsPair(char const *name, HyPerCol *hc) { initialize(name, hc); }

WeightsPair::~WeightsPair() {
   delete mPreWeights;
   delete mPostWeights;
   delete mOutputStateStream;
}

int WeightsPair::initialize(char const *name, HyPerCol *hc) {
   return BaseObject::initialize(name, hc);
}

int WeightsPair::setDescription() {
   description.clear();
   description.append("WeightsPair").append(" \"").append(getName()).append("\"");
   return PV_SUCCESS;
}

int WeightsPair::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_nxp(ioFlag);
   ioParam_nyp(ioFlag);
   ioParam_nfp(ioFlag);
   ioParam_sharedWeights(ioFlag);
   ioParam_writeStep(ioFlag);
   ioParam_initialWriteTime(ioFlag);
   ioParam_writeCompressedWeights(ioFlag);
   ioParam_writeCompressedCheckpoints(ioFlag);
   return PV_SUCCESS;
}

void WeightsPair::ioParam_nxp(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "nxp", &mPatchSizeX, 1);
}

void WeightsPair::ioParam_nyp(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "nyp", &mPatchSizeY, 1);
}

void WeightsPair::ioParam_nfp(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "nfp", &mPatchSizeF, -1, false);
   if (ioFlag == PARAMS_IO_READ && mPatchSizeF < 0 && !parent->parameters()->present(name, "nfp")
       && parent->getCommunicator()->globalCommRank() == 0) {
      InfoLog().printf(
            "%s: nfp will be set in the communicateInitInfo() stage.\n", getDescription_c());
   }
}

void WeightsPair::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   // TODO: deliver methods with GPU must balk if shared flag is off.
   parent->parameters()->ioParamValue(
         ioFlag, name, "sharedWeights", &mSharedWeights, true /*default*/, true /*warn if absent*/);
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
            if (parent->columnId() == 0) {
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
   int status = BaseObject::respond(message);
   if (status != PV_SUCCESS) {
      return status;
   }
   else if (auto castMessage = std::dynamic_pointer_cast<ConnectionOutputMessage const>(message)) {
      return respondConnectionOutput(castMessage);
   }
   else {
      return status;
   }
}

int WeightsPair::respondConnectionOutput(std::shared_ptr<ConnectionOutputMessage const> message) {
   outputState(message->mTime);
   return PV_SUCCESS;
}

int WeightsPair::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = BaseObject::communicateInitInfo(message);
   if (status != PV_SUCCESS) {
      return status;
   }
   if (mConnectionData == nullptr) {
      mConnectionData = mapLookupByType<ConnectionData>(message->mHierarchy, getDescription());
   }
   pvAssert(mConnectionData != nullptr);

   if (!mConnectionData->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the ConnectionData component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return PV_POSTPONE;
   }

   HyPerLayer *pre           = mConnectionData->getPre();
   HyPerLayer *post          = mConnectionData->getPost();
   PVLayerLoc const *preLoc  = pre->getLayerLoc();
   PVLayerLoc const *postLoc = post->getLayerLoc();

   // Margins
   int xmargin         = requiredConvolveMargin(preLoc->nx, postLoc->nx, getPatchSizeX());
   int receivedxmargin = 0;
   int statusx         = pre->requireMarginWidth(xmargin, &receivedxmargin, 'x');
   if (statusx != PV_SUCCESS) {
      ErrorLog().printf(
            "Margin Failure for layer %s.  Received x-margin is %d, but %s requires margin of at "
            "least %d\n",
            pre->getDescription_c(),
            receivedxmargin,
            name,
            xmargin);
      status = PV_MARGINWIDTH_FAILURE;
   }
   int ymargin         = requiredConvolveMargin(preLoc->ny, postLoc->ny, getPatchSizeY());
   int receivedymargin = 0;
   int statusy         = pre->requireMarginWidth(ymargin, &receivedymargin, 'y');
   if (statusy != PV_SUCCESS) {
      ErrorLog().printf(
            "Margin Failure for layer %s.  Received y-margin is %d, but %s requires margin of at "
            "least %d\n",
            pre->getDescription_c(),
            receivedymargin,
            name,
            ymargin);
      status = PV_MARGINWIDTH_FAILURE;
   }

   if (mPatchSizeF < 0) {
      mPatchSizeF = postLoc->nf;
      if (mWarnDefaultNfp && parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s setting nfp to number of postsynaptic features = %d.\n",
               getDescription_c(),
               mPatchSizeF);
      }
   }
   if (mPatchSizeF != postLoc->nf) {
      if (parent->columnId() == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "Params file specifies %d features for %s,\n", mPatchSizeF, getDescription_c());
         errorMessage.printf(
               "but %d features for post-synaptic layer %s\n", postLoc->nf, post->getName());
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(PV_FAILURE);
   }
   // Currently, the only acceptable number for mPatchSizeF is the number of post-synaptic features.
   // However, we may add flexibility on this score in the future, e.g. MPI in feature space
   // with each feature connecting to only a few nearby features.
   // Accordingly, we still keep ioParam_nfp.

   needPre();

   return status;
}

void WeightsPair::needPre() {
   if (mPreWeights == nullptr) {
      mPreWeights = new Weights(std::string(name));
   }
}

void WeightsPair::needPost() {
   if (mPostWeights == nullptr) {
      needPre();
      mPostWeights = new PostWeights(std::string(name), mPreWeights);
   }
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
   mPreWeights->initialize(
         mPatchSizeX,
         mPatchSizeY,
         mPatchSizeF,
         mConnectionData->getPre()->getLayerLoc(),
         mConnectionData->getPost()->getLayerLoc(),
         mConnectionData->getNumAxonalArbors(),
         mSharedWeights,
         0.0 /* timestamp */);
   mPreWeights->allocateDataStructures();
}

void WeightsPair::allocatePostWeights() {
   auto *postWeights = dynamic_cast<PostWeights *>(mPostWeights);
   FatalIf(
         postWeights == nullptr,
         "WeightsPair::allocateDataStructures called with mPostWeights set, "
         "but not a PostWeights object.\n",
         name);
   // If a derived sets mPostWeights to a non-PostWeights object (e.g. TransposeConn),
   // it needs to override allocateDataStructures.
   postWeights->initializePostWeights(mPreWeights);
   mPostWeights->allocateDataStructures();
}

int WeightsPair::registerData(Checkpointer *checkpointer) {
   int status = BaseObject::registerData(checkpointer);
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
