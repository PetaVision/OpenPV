/*
 * PatchSize.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: Pete Schultz
 */

#include "PatchSize.hpp"
#include "components/LayerGeometry.hpp"

namespace PV {

PatchSize::PatchSize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

PatchSize::~PatchSize() {}

void PatchSize::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   BaseObject::initialize(paramsIO, comm);
}

void PatchSize::setObjectType() { mObjectType = "PatchSize"; }

int PatchSize::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   ioParam_nxp(ioSwitch);
   ioParam_nyp(ioSwitch);
   ioParam_nfp(ioSwitch);
   return PV_SUCCESS;
}

void PatchSize::ioParam_nxp(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "nxp", &mNxp, false /*warnIfAbsentFlag*/);
   if (ioSwitch == ParamsIOSwitch::Read && mNxp < 0 && !mParamsIO->isPresent("nxp")
       && mCommunicator->globalCommRank() == 0) {
      InfoLog().printf(
            "%s: nxp will be set in the communicateInitInfo() stage.\n", getDescription_c());
   }
}

void PatchSize::ioParam_nyp(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "nyp", &mNyp, false /*warnIfAbsentFlag*/);
   if (ioSwitch == ParamsIOSwitch::Read && mNyp < 0 && !mParamsIO->isPresent("nyp")
       && mCommunicator->globalCommRank() == 0) {
      InfoLog().printf(
            "%s: nyp will be set in the communicateInitInfo() stage.\n", getDescription_c());
   }
}

void PatchSize::ioParam_nfp(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "nfp", &mNfp, false /*warnIfAbsentFlag*/);
   if (ioSwitch == ParamsIOSwitch::Read && mNfp < 0 && !mParamsIO->isPresent("nfp")
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
   bool isBroadcastPre  = pre->getComponentByType<LayerGeometry>()->getBroadcastFlag();
   bool isBroadcastPost = post->getComponentByType<LayerGeometry>()->getBroadcastFlag();
   if (isBroadcastPre) {
      int correctNxp = isBroadcastPost ? 1 : post->getLayerLoc()->nxGlobal;
      if (mNxp < 0) {
         mNxp = correctNxp;
         InfoLog().printf(
               "%s setting nxp to %d, the size of post-synaptic layer \"%s\".\n",
               getDescription_c(), correctNxp, post->getName());
      }
      FatalIf(
            mNxp != correctNxp,
            "%s has a pre-synaptic broadcast layer, so nxp must be the same as the post-synaptic nx.\n"
            "    (given nxp = %d; post-synaptic nx = %d)\n",
            getDescription_c(), mNxp, correctNxp);
   }
   else { // pre-layer is not a broadcast layer
      FatalIf(
            mNxp < 0,
            "%s has a non-broadcast pre-synaptic layer, but param nxp was not set\n",
            getDescription_c());
   }
   mPatchSizeX = mNxp;
   if (isBroadcastPre and !isBroadcastPost) {
      mPatchSizeX = mPatchSizeX / mCommunicator->numCommColumns();
   }
}

void PatchSize::setPatchSizeY(HyPerLayer *pre, HyPerLayer *post) {
   bool isBroadcastPre  = pre->getComponentByType<LayerGeometry>()->getBroadcastFlag();
   bool isBroadcastPost = post->getComponentByType<LayerGeometry>()->getBroadcastFlag();
   if (isBroadcastPre) {
      int correctNyp = isBroadcastPost ? 1 : post->getLayerLoc()->nyGlobal;
      if (mNyp < 0) {
         mNyp = correctNyp;
         InfoLog().printf(
               "%s setting nyp to %d, the size of post-synaptic layer \"%s\".\n",
               getDescription_c(), correctNyp, post->getName());
      }
      FatalIf(
            mNyp != correctNyp,
            "%s has a pre-synaptic broadcast layer, so nyp must be the same as the post-synaptic ny.\n"
            "    (given nyp = %d; post-synaptic ny = %d)\n",
            getDescription_c(), mNyp, correctNyp);
   }
   else { // pre-layer is not a broadcast layer
      FatalIf(
            mNyp < 0,
            "%s has a non-broadcast pre-synaptic layer, but param nyp was not set\n",
            getDescription_c());
   }
   mPatchSizeY = mNyp;
   if (isBroadcastPre and !isBroadcastPost) {
      mPatchSizeY = mPatchSizeY / mCommunicator->numCommRows();
   }
}

void PatchSize::setPatchSizeF(HyPerLayer *pre, HyPerLayer *post) {
   int const nfPost = post->getLayerLoc()->nf;

   if (mNfp < 0) {
      mNfp = nfPost;
      if (mWarnDefaultNfp && mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s setting nfp to number of postsynaptic features = %d.\n",
               getDescription_c(),
               nfPost);
      }
   }
   if (mNfp != nfPost) {
      if (mCommunicator->globalCommRank() == 0) {
         ErrorLog().printf(
               "Params file specifies %d features for %s, "
               "but %d features for post-synaptic layer %s\n",
               mNfp, getDescription_c(), nfPost, post->getName());
      }
      MPI_Barrier(mCommunicator->globalCommunicator());
      exit(PV_FAILURE);
   }
   mPatchSizeF = mNfp;
   // Currently, the only acceptable number for mPatchSizeF is the number of post-synaptic features.
   // However, we may add flexibility on this score in the future, e.g. MPI in feature space
   // with each feature connecting to only a few nearby features.
   // Accordingly, we still keep ioParam_nfp.
}

} // namespace PV
