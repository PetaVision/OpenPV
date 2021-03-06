/*
 * ImpliedWeightsPair.cpp
 *
 *  Created on: Nov 17, 2017
 *      Author: Pete Schultz
 */

#include "ImpliedWeightsPair.hpp"
#include "components/ImpliedWeights.hpp"

namespace PV {

ImpliedWeightsPair::ImpliedWeightsPair(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

ImpliedWeightsPair::~ImpliedWeightsPair() {}

void ImpliedWeightsPair::initialize(char const *name, PVParams *params, Communicator const *comm) {
   WeightsPairInterface::initialize(name, params, comm);
}

void ImpliedWeightsPair::setObjectType() { mObjectType = "ImpliedWeightsPair"; }

void ImpliedWeightsPair::createPreWeights(std::string const &weightsName) {
   pvAssert(mPreWeights == nullptr and mInitInfoCommunicatedFlag);
   mPreWeights = new ImpliedWeights(
         weightsName,
         mPatchSize->getPatchSizeX(),
         mPatchSize->getPatchSizeY(),
         mPatchSize->getPatchSizeF(),
         mConnectionData->getPre()->getLayerLoc(),
         mConnectionData->getPost()->getLayerLoc(),
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
         -std::numeric_limits<double>::infinity() /*timestamp*/);
}

} // namespace PV
