/*
 * CloneLayerGeometry.cpp
 */

#include "CloneLayerGeometry.hpp"
#include "components/OriginalLayerNameParam.hpp"

namespace PV {

CloneLayerGeometry::CloneLayerGeometry(
      char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

CloneLayerGeometry::CloneLayerGeometry() {}

CloneLayerGeometry::~CloneLayerGeometry() {}

void CloneLayerGeometry::initialize(char const *name, PVParams *params, Communicator const *comm) {
   LayerGeometry::initialize(name, params, comm);
}

void CloneLayerGeometry::setObjectType() { mObjectType = "CloneLayerGeometry"; }

void CloneLayerGeometry::ioParam_broadcastFlag(enum ParamsIOFlag ioFlag) {}

void CloneLayerGeometry::ioParam_nxScale(enum ParamsIOFlag ioFlag) {}

void CloneLayerGeometry::ioParam_nyScale(enum ParamsIOFlag ioFlag) {}

void CloneLayerGeometry::ioParam_nf(enum ParamsIOFlag ioFlag) {}

Response::Status CloneLayerGeometry::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {

   auto *objectTable            = message->mObjectTable;
   auto *originalLayerNameParam = objectTable->findObject<OriginalLayerNameParam>(getName());
   FatalIf(
         originalLayerNameParam == nullptr,
         "%s could not find an OriginalLayerNameParam.\n",
         getDescription_c());
   if (!originalLayerNameParam->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }

   // Retrieve original layer's LayerGeometry
   char const *originalLayerName = originalLayerNameParam->getLinkedObjectName();
   auto originalGeometry = objectTable->findObject<LayerGeometry>(originalLayerName);
   FatalIf(
         originalGeometry == nullptr,
         "%s could not find an LayerGeometry component within layer \"%s\".\n",
         getDescription_c(),
         originalLayerName);
   if (!originalGeometry->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }

   mBroadcastFlag = originalGeometry->getBroadcastFlag();
   mLayerLoc      = *originalGeometry->getLayerLoc();
   mXScale        = originalGeometry->getXScale();
   mYScale        = originalGeometry->getYScale();
   mNxScale       = std::exp2(-static_cast<float>(mXScale));
   mNyScale       = std::exp2(-static_cast<float>(mYScale));
   mNumFeatures   = mLayerLoc.nf;

   mNumNeurons           = (long)mLayerLoc.nx * (long)mLayerLoc.ny * (long)mLayerLoc.nf;
   mNumNeuronsAllBatches = mNumNeurons * mLayerLoc.nbatch;

   updateNumExtended();

   return Response::SUCCESS;
}

} // namespace PV
