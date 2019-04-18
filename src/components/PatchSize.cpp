/*
 * PatchSize.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: Pete Schultz
 */

#include "PatchSize.hpp"
#include "columns/HyPerCol.hpp"
#include "utils/MapLookupByType.hpp"

namespace PV {

PatchSize::PatchSize(char const *name, HyPerCol *hc) { initialize(name, hc); }

PatchSize::~PatchSize() {}

int PatchSize::initialize(char const *name, HyPerCol *hc) {
   return BaseObject::initialize(name, hc);
}

void PatchSize::setObjectType() { mObjectType = "PatchSize"; }

int PatchSize::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_nxp(ioFlag);
   ioParam_nyp(ioFlag);
   ioParam_nfp(ioFlag);
   return PV_SUCCESS;
}

void PatchSize::ioParam_nxp(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "nxp", &mPatchSizeX, mPatchSizeX);
}

void PatchSize::ioParam_nyp(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "nyp", &mPatchSizeY, mPatchSizeY);
}

void PatchSize::ioParam_nfp(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "nfp", &mPatchSizeF, mPatchSizeF, false);
   if (ioFlag == PARAMS_IO_READ && mPatchSizeF < 0 && !parent->parameters()->present(name, "nfp")
       && parent->getCommunicator()->globalCommRank() == 0) {
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
   mConnectionData = mapLookupByType<ConnectionData>(message->mHierarchy, getDescription());
   FatalIf(
         mConnectionData == nullptr,
         "%s received CommunicateInitInfo message without a ConnectionData component.\n",
         getDescription_c());

   if (!mConnectionData->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the ConnectionData component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return Response::POSTPONE;
   }

   HyPerLayer *post = mConnectionData->getPost();
   int const nfPost = post->getLayerLoc()->nf;

   if (mPatchSizeF < 0) {
      mPatchSizeF = nfPost;
      if (mWarnDefaultNfp && parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s setting nfp to number of postsynaptic features = %d.\n",
               getDescription_c(),
               mPatchSizeF);
      }
   }
   if (mPatchSizeF != nfPost) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "Params file specifies %d features for %s,\n", mPatchSizeF, getDescription_c());
         errorMessage.printf(
               "but %d features for post-synaptic layer %s\n", nfPost, post->getName());
      }
      MPI_Barrier(parent->getCommunicator()->globalCommunicator());
      exit(PV_FAILURE);
   }
   // Currently, the only acceptable number for mPatchSizeF is the number of post-synaptic features.
   // However, we may add flexibility on this score in the future, e.g. MPI in feature space
   // with each feature connecting to only a few nearby features.
   // Accordingly, we still keep ioParam_nfp.

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

} // namespace PV
