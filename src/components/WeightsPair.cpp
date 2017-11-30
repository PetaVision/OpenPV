/*
 * WeightsPair.cpp
 *
 *  Created on: Nov 17, 2017
 *      Author: Pete Schultz
 */

#include "WeightsPair.hpp"
#include "columns/HyPerCol.hpp"
#include "layers/HyPerLayer.hpp"
#include "utils/MapLookupByType.hpp"

namespace PV {

WeightsPair::WeightsPair(char const *name, HyPerCol *hc) { initialize(name, hc); }

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
   if (ioFlag == PARAMS_IO_READ && mPatchSizeX < 0 && !parent->parameters()->present(name, "nfp")
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

   return status;
}

int WeightsPair::allocateDataStructures() {
   setPreWeights(
         new Weights(
               std::string(name),
               mPatchSizeX,
               mPatchSizeY,
               mPatchSizeF,
               mConnectionData->getPre()->getLayerLoc(),
               mConnectionData->getPost()->getLayerLoc(),
               mConnectionData->getNumAxonalArbors(),
               mSharedWeights,
               0.0));
   mNeedPre = true;
   if (mNeedPre) {
      getPreWeights()->allocateDataStructures();
   }
   if (mNeedPost) {
      setPostWeights(new PostWeights(std::string(name), mPreWeights));
   }
   return PV_SUCCESS;
}

} // namespace PV
