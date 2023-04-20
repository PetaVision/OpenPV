#include "TargetLayerComponent.hpp"

#include "utils/PVLog.hpp"

namespace PV {

TargetLayerComponent::TargetLayerComponent(char const *objName, PVParams *params) {
   initialize(objName, params);
}

TargetLayerComponent::~TargetLayerComponent() { free(mTargetLayerName); }

Response::Status TargetLayerComponent::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   if (mTargetLayer == nullptr) {
      mTargetLayer = message->mObjectTable->findObject<HyPerLayer>(mTargetLayerName);
      FatalIf(
            mTargetLayer == nullptr,
            "Probe %s targetLayer \"%s\" is not a layer in the column.\n",
            getName_c(),
            mTargetLayerName);
   }
   return Response::SUCCESS;
}

void TargetLayerComponent::initialize(char const *objName, PVParams *params) {
   ProbeComponent::initialize(objName, params);
}

void TargetLayerComponent::ioParam_targetLayer(enum ParamsIOFlag ioFlag) {
   getParams()->ioParamString(
         ioFlag,
         getName_c(),
         "targetLayer",
         &mTargetLayerName,
         nullptr /*default*/,
         false /*warnIfAbsent*/);

   // If targetLayer is not present, check for targetName. targetName as a parameter for
   // layer probes was deprecated in favor of targetLayer on Oct 20, 2022.
   // Once targetName is removed as a synonym, the above code can simply call the
   // PVParams::ioParamStringRequired() function.
   if (mTargetLayerName == nullptr or mTargetLayerName[0] == '\0') {
      getParams()->ioParamString(
            ioFlag,
            getName_c(),
            "targetName",
            &mTargetLayerName,
            nullptr /*default*/,
            false /*warnIfAbsent*/);
      if (mTargetLayerName != nullptr and mTargetLayerName[0] != '\0') {
         WarnLog().printf(
               "Probe %s parameter targetName is deprecated. "
               "Use targetLayer for layer probes instead.\n",
               getName_c());
      }
   }
   FatalIf(
         mTargetLayerName == nullptr or mTargetLayerName[0] == '\0',
         "Probe %s requires the targetLayer string parameter to be set\n",
         getName_c());
}

void TargetLayerComponent::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_targetLayer(ioFlag);
}

} // namespace PV
