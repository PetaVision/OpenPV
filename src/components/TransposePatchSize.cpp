/*
 * TransposePatchSize.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: pschultz
 */

#include "TransposePatchSize.hpp"

namespace PV {

TransposePatchSize::TransposePatchSize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

TransposePatchSize::TransposePatchSize() {}

TransposePatchSize::~TransposePatchSize() {}

void TransposePatchSize::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   PatchSize::initialize(paramsIO, comm);
}

void TransposePatchSize::setObjectType() { mObjectType = "TransposePatchSize"; }

void TransposePatchSize::setPatchSizeX(HyPerLayer *pre, HyPerLayer *post) {
   mOriginalPatchSizeX          = mOriginalPatchSize->getPatchSizeX();
   auto *originalConnectionData = mOriginalPatchSize->getConnectionData();
   pvAssert(originalConnectionData);
   PVLayerLoc const *originalPreLoc  = originalConnectionData->getPre()->getLayerLoc();
   PVLayerLoc const *originalPostLoc = originalConnectionData->getPost()->getLayerLoc();
   mPatchSizeX = calcPostPatchSize(mOriginalPatchSizeX, originalPreLoc->nx, originalPostLoc->nx);
   mParamsIO->handleUnnecessaryParameter("nxp", mNxp);
}

void TransposePatchSize::setPatchSizeY(HyPerLayer *pre, HyPerLayer *post) {
   mOriginalPatchSizeY          = mOriginalPatchSize->getPatchSizeY();
   int const nypOrig            = mOriginalPatchSize->getPatchSizeY();
   auto *originalConnectionData = mOriginalPatchSize->getConnectionData();
   pvAssert(originalConnectionData);
   PVLayerLoc const *originalPreLoc  = originalConnectionData->getPre()->getLayerLoc();
   PVLayerLoc const *originalPostLoc = originalConnectionData->getPost()->getLayerLoc();
   mPatchSizeY = calcPostPatchSize(mOriginalPatchSizeY, originalPreLoc->ny, originalPostLoc->ny);
   mParamsIO->handleUnnecessaryParameter("nyp", mNyp);
}

void TransposePatchSize::setPatchSizeF(HyPerLayer *pre, HyPerLayer *post) {
   mOriginalPatchSizeF = mOriginalPatchSize->getPatchSizeF();
   PatchSize::setPatchSizeF(pre, post);
   mParamsIO->handleUnnecessaryParameter("nfp", mNfp);
}

} // namespace PV
