/*
 * TransposePatchSize.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: pschultz
 */

#include "TransposePatchSize.hpp"

namespace PV {

TransposePatchSize::TransposePatchSize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

TransposePatchSize::TransposePatchSize() {}

TransposePatchSize::~TransposePatchSize() {}

void TransposePatchSize::initialize(char const *name, PVParams *params, Communicator const *comm) {
   PatchSize::initialize(name, params, comm);
}

void TransposePatchSize::setObjectType() { mObjectType = "TransposePatchSize"; }

void TransposePatchSize::setPatchSizeX(HyPerLayer *pre, HyPerLayer *post) {
   int const nxpOrig            = mOriginalPatchSize->getPatchSizeX();
   auto *originalConnectionData = mOriginalPatchSize->getConnectionData();
   pvAssert(originalConnectionData);
   PVLayerLoc const *originalPreLoc  = originalConnectionData->getPre()->getLayerLoc();
   PVLayerLoc const *originalPostLoc = originalConnectionData->getPost()->getLayerLoc();
   mPatchSizeX = calcPostPatchSize(nxpOrig, originalPreLoc->nx, originalPostLoc->nx);
   parameters()->handleUnnecessaryParameter(getName(), "nxp", mNxp);
}

void TransposePatchSize::setPatchSizeY(HyPerLayer *pre, HyPerLayer *post) {
   int const nypOrig            = mOriginalPatchSize->getPatchSizeY();
   auto *originalConnectionData = mOriginalPatchSize->getConnectionData();
   pvAssert(originalConnectionData);
   PVLayerLoc const *originalPreLoc  = originalConnectionData->getPre()->getLayerLoc();
   PVLayerLoc const *originalPostLoc = originalConnectionData->getPost()->getLayerLoc();
   mPatchSizeY = calcPostPatchSize(nypOrig, originalPreLoc->ny, originalPostLoc->ny);
   parameters()->handleUnnecessaryParameter(getName(), "nyp", mNyp);
}

void TransposePatchSize::setPatchSizeF(HyPerLayer *pre, HyPerLayer *post) {
   PatchSize::setPatchSizeF(pre, post);
   parameters()->handleUnnecessaryParameter(getName(), "nfp", mNfp);
}

} // namespace PV
