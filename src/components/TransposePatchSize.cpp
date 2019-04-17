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

void TransposePatchSize::setPatchSize(PatchSize *originalPatchSize) {
   auto *originalConnectionData = originalPatchSize->getConnectionData();
   pvAssert(originalConnectionData);
   int const nxpOrig                 = originalPatchSize->getPatchSizeX();
   int const nypOrig                 = originalPatchSize->getPatchSizeY();
   PVLayerLoc const *originalPreLoc  = originalConnectionData->getPre()->getLayerLoc();
   PVLayerLoc const *originalPostLoc = originalConnectionData->getPost()->getLayerLoc();
   mPatchSizeX = calcPostPatchSize(nxpOrig, originalPreLoc->nx, originalPostLoc->nx);
   mPatchSizeY = calcPostPatchSize(nypOrig, originalPreLoc->ny, originalPostLoc->ny);
   mPatchSizeF = -1;
}

} // namespace PV
