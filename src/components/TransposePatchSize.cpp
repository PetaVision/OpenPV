/*
 * TransposePatchSize.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: pschultz
 */

#include "TransposePatchSize.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

TransposePatchSize::TransposePatchSize(char const *name, HyPerCol *hc) { initialize(name, hc); }

TransposePatchSize::TransposePatchSize() {}

TransposePatchSize::~TransposePatchSize() {}

int TransposePatchSize::initialize(char const *name, HyPerCol *hc) {
   return PatchSize::initialize(name, hc);
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
