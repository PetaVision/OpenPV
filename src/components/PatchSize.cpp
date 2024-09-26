/*
 * PatchSize.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: Pete Schultz
 */

#include "PatchSize.hpp"
#include "components/LayerGeometry.hpp"

namespace PV {

PatchSize::PatchSize(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

PatchSize::~PatchSize() {}

void PatchSize::initialize(char const *name, PVParams *params, Communicator const *comm) {
   BaseObject::initialize(name, params, comm);
}

void PatchSize::setObjectType() { mObjectType = "PatchSize"; }

int PatchSize::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_nxp(ioFlag);
   ioParam_nyp(ioFlag);
   ioParam_nfp(ioFlag);
   return PV_SUCCESS;
}

void PatchSize::ioParam_nxp(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "nxp", &mPatchSizeX, mPatchSizeX, false);
   if (ioFlag == PARAMS_IO_READ && mPatchSizeF < 0 && !parameters()->present(getName(), "nxp")
       && mCommunicator->globalCommRank() == 0) {
      InfoLog().printf(
            "%s: nxp will be set in the communicateInitInfo() stage.\n", getDescription_c());
   }
}

void PatchSize::ioParam_nyp(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "nyp", &mPatchSizeY, mPatchSizeY, false);
   if (ioFlag == PARAMS_IO_READ && mPatchSizeF < 0 && !parameters()->present(getName(), "nyp")
       && mCommunicator->globalCommRank() == 0) {
      InfoLog().printf(
            "%s: nyp will be set in the communicateInitInfo() stage.\n", getDescription_c());
   }
}

void PatchSize::ioParam_nfp(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "nfp", &mPatchSizeF, mPatchSizeF, false);
   if (ioFlag == PARAMS_IO_READ && mPatchSizeF < 0 && !parameters()->present(getName(), "nfp")
       && mCommunicator->globalCommRank() == 0) {
      InfoLog().printf(
            "%s: nfp will be set in the communicateInitInfo() stage.\n", getDescription_c());
   }
}

Response::Status
PatchSize::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = BaseObject::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   mConnectionData = message->mObjectTable->findObject<ConnectionData>(getName());
   pvAssert(mConnectionData);

   if (!mConnectionData->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the ConnectionData component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return Response::POSTPONE;
   }

   HyPerLayer *pre = mConnectionData->getPre();
   if (!pre->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the pre-synaptic layer has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return Response::POSTPONE;
   }

   HyPerLayer *post = mConnectionData->getPost();
   if (!post->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the post-synaptic layer has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return Response::POSTPONE;
   }

   setPatchSizeX(pre, post);
   setPatchSizeY(pre, post);
   setPatchSizeF(pre, post);

   return Response::SUCCESS;
}

int PatchSize::calcPostPatchSize(int prePatchSize, int numNeuronsPre, int numNeuronsPost) {
   if (numNeuronsPre == numNeuronsPost) {
      return prePatchSize;
   }
   else if (numNeuronsPre > numNeuronsPost) {
      std::div_t scaleDivision = div(numNeuronsPre, numNeuronsPost);
      FatalIf(
            scaleDivision.rem != 0,
            "calcPostPatchSize called with numNeuronsPre (%d) greater than numNeuronsPost (%d), "
            "but not an integer multiple.\n",
            numNeuronsPre,
            numNeuronsPost);
      return prePatchSize * scaleDivision.quot;
   }
   else {
      std::div_t const scaleDivision = div(numNeuronsPost, numNeuronsPre);
      FatalIf(
            scaleDivision.rem != 0,
            "calcPostPatchSize called with numNeuronsPost (%d) greater than numNeuronsPre (%d), "
            "but not an integer multiple.\n",
            numNeuronsPost,
            numNeuronsPre);
      int const scaleFactor         = scaleDivision.quot;
      std::div_t const newPatchSize = div(prePatchSize, scaleFactor);
      FatalIf(
            newPatchSize.rem != 0,
            "calcPostPatchSize called with scale factor of numNeuronsPost/numNeuronsPre = %d, "
            "but prePatchSize (%d) is not an integer multiple of the scale factor.\n",
            scaleFactor,
            prePatchSize);
      return prePatchSize / scaleFactor;
   }
}

void PatchSize::setPatchSizeX(HyPerLayer *pre, HyPerLayer *post) {
   auto *preLayerGeometry = pre->getComponentByType<LayerGeometry>();
   bool isBroadcastPre = preLayerGeometry->getBroadcastFlag();
   if (isBroadcastPre) {
      int correctPatchSizeX = post->getLayerLoc()->nxGlobal;
      if (mPatchSizeX < 0) { mPatchSizeX = correctPatchSizeX; }
      else if (mPatchSizeX * mCommunicator->numCommColumns() == correctPatchSizeX) {
         WarnLog().printf(
               "%s setting nxp to %d, the nxGlobal of post-synaptic layer \"%s\".\n", 
               getDescription_c(), correctPatchSizeX, post->getName());
         mPatchSizeX = correctPatchSizeX;
      }
      FatalIf(
            mPatchSizeX != correctPatchSizeX,
            "%s has a pre-synaptic broadcast layer, so nxp must be the global post-synaptic nx.\n"
            "    (given nxp = %d; global post-synaptic nx = %d)\n",
            getDescription_c(), mPatchSizeX, correctPatchSizeX);
   }
   else { // pre-layer is not a broadcast layer
      FatalIf(
            mPatchSizeX < 0,
            "%s has a non-broadcast pre-synaptic layer, but param nxp was not set\n",
            getDescription_c());
   }
}

void PatchSize::setPatchSizeY(HyPerLayer *pre, HyPerLayer *post) {
   auto *preLayerGeometry = pre->getComponentByType<LayerGeometry>();
   bool isBroadcastPre = preLayerGeometry->getBroadcastFlag();
   if (isBroadcastPre) {
      int correctPatchSizeY = post->getLayerLoc()->nyGlobal;
      if (mPatchSizeY < 0) { mPatchSizeY = correctPatchSizeY; }
      else if (mPatchSizeY * mCommunicator->numCommColumns() == correctPatchSizeY) {
         WarnLog().printf(
               "%s setting nyp to %d, the ny of post-synaptic layer \"%s\".\n", 
               getDescription_c(), correctPatchSizeY, post->getName());
         mPatchSizeY = correctPatchSizeY;
      }
      FatalIf(
            mPatchSizeY != correctPatchSizeY,
            "%s has a pre-synaptic broadcast layer, so nyp must be the global post-synaptic ny.\n"
            "    (given nyp = %d; global post-synaptic ny = %d)\n",
            getDescription_c(), mPatchSizeY, correctPatchSizeY);
   }
   else { // pre-layer is not a broadcast layer
      FatalIf(
            mPatchSizeY < 0,
            "%s has a non-broadcast pre-synaptic layer, but param nyp was not set\n",
            getDescription_c());
   }
}

void PatchSize::setPatchSizeF(HyPerLayer *pre, HyPerLayer *post) {
   int const nfPost = post->getLayerLoc()->nf;

   if (mPatchSizeF < 0) {
      mPatchSizeF = nfPost;
      if (mWarnDefaultNfp && mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s setting nfp to number of postsynaptic features = %d.\n",
               getDescription_c(),
               mPatchSizeF);
      }
   }
   if (mPatchSizeF != nfPost) {
      if (mCommunicator->globalCommRank() == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "Params file specifies %d features for %s,\n", mPatchSizeF, getDescription_c());
         errorMessage.printf(
               "but %d features for post-synaptic layer %s\n", nfPost, post->getName());
      }
      MPI_Barrier(mCommunicator->globalCommunicator());
      exit(PV_FAILURE);
   }
   // Currently, the only acceptable number for mPatchSizeF is the number of post-synaptic features.
   // However, we may add flexibility on this score in the future, e.g. MPI in feature space
   // with each feature connecting to only a few nearby features.
   // Accordingly, we still keep ioParam_nfp.
}

} // namespace PV
