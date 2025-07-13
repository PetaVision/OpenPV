/*
 * CloneLayerGeometry.cpp
 */

#include "CloneLayerGeometry.hpp"
#include "components/OriginalLayerNameParam.hpp"

namespace PV {

CloneLayerGeometry::CloneLayerGeometry(
      
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

CloneLayerGeometry::CloneLayerGeometry() {}

CloneLayerGeometry::~CloneLayerGeometry() {}

void CloneLayerGeometry::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   LayerGeometry::initialize(params, defaults, comm);
}

void CloneLayerGeometry::setObjectType() { mObjectType = "CloneLayerGeometry"; }

void CloneLayerGeometry::ioParam_broadcastFlag(ParamsIOSwitch ioSwitch) {}

void CloneLayerGeometry::ioParam_nxScale(ParamsIOSwitch ioSwitch) {}

void CloneLayerGeometry::ioParam_nyScale(ParamsIOSwitch ioSwitch) {}

void CloneLayerGeometry::ioParam_nf(ParamsIOSwitch ioSwitch) {}

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
   std::string const &originalLayerName = originalLayerNameParam->getLinkedObjectName();
   auto originalGeometry = objectTable->findObject<LayerGeometry>(originalLayerName);
   FatalIf(
         originalGeometry == nullptr,
         "%s could not find an LayerGeometry component within layer \"%s\".\n",
         getDescription_c(),
         originalLayerName.c_str());
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

   mNumNeurons           = mLayerLoc.nx * mLayerLoc.ny * mLayerLoc.nf;
   mNumNeuronsAllBatches = mNumNeurons * mLayerLoc.nbatch;

   updateNumExtended();

   return Response::SUCCESS;
}

} // namespace PV
