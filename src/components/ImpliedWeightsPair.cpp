/*
 * ImpliedWeightsPair.cpp
 *
 *  Created on: Nov 17, 2017
 *      Author: Pete Schultz
 */

#include "ImpliedWeightsPair.hpp"
#include "columns/HyPerCol.hpp"
#include "components/ImpliedWeights.hpp"

namespace PV {

ImpliedWeightsPair::ImpliedWeightsPair(char const *name, HyPerCol *hc) { initialize(name, hc); }

ImpliedWeightsPair::~ImpliedWeightsPair() {}

int ImpliedWeightsPair::initialize(char const *name, HyPerCol *hc) {
   return WeightsPairInterface::initialize(name, hc);
}

int ImpliedWeightsPair::setDescription() {
   description.clear();
   description.append("ImpliedWeightsPair").append(" \"").append(getName()).append("\"");
   return PV_SUCCESS;
}

void ImpliedWeightsPair::createPreWeights(std::string const &weightsName) {
   pvAssert(mPreWeights == nullptr and mInitInfoCommunicatedFlag);
   mPreWeights = new ImpliedWeights(
         weightsName,
         mPatchSize->getPatchSizeX(),
         mPatchSize->getPatchSizeY(),
         mPatchSize->getPatchSizeF(),
         mConnectionData->getPre()->getLayerLoc(),
         mConnectionData->getPost()->getLayerLoc(),
         mArborList->getNumAxonalArbors(),
         true /*sharedWeights*/,
         -std::numeric_limits<double>::infinity() /*timestamp*/);
}

void ImpliedWeightsPair::createPostWeights(std::string const &weightsName) {
   pvAssert(mPostWeights == nullptr and mInitInfoCommunicatedFlag);
   PVLayerLoc const *preLoc  = mConnectionData->getPre()->getLayerLoc();
   PVLayerLoc const *postLoc = mConnectionData->getPost()->getLayerLoc();
   int nxpPre                = mPatchSize->getPatchSizeX();
   int nxpPost               = PatchSize::calcPostPatchSize(nxpPre, preLoc->nx, postLoc->nx);
   int nypPre                = mPatchSize->getPatchSizeY();
   int nypPost               = PatchSize::calcPostPatchSize(nypPre, preLoc->ny, postLoc->ny);
   mPostWeights              = new ImpliedWeights(
         weightsName,
         nxpPost,
         nypPost,
         preLoc->nf /* number of features in post patch */,
         postLoc,
         preLoc,
         mArborList->getNumAxonalArbors(),
         true /*sharedWeights*/,
         -std::numeric_limits<double>::infinity() /*timestamp*/);
}

} // namespace PV
