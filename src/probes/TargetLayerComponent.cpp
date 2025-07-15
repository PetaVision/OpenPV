#include "TargetLayerComponent.hpp"

#include "utils/PVLog.hpp"

namespace PV {

TargetLayerComponent::TargetLayerComponent(std::shared_ptr<ParamsIO> paramsIO) {
   initialize(paramsIO);
}

TargetLayerComponent::~TargetLayerComponent() {}

Response::Status TargetLayerComponent::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   if (mTargetLayer == nullptr) {
      mTargetLayer = message->mObjectTable->findObject<HyPerLayer>(mTargetLayerName);
      FatalIf(
            mTargetLayer == nullptr,
            "Probe %s targetLayer \"%s\" is not a layer in the column.\n",
            getName_c(),
            mTargetLayerName.c_str());
   }
   return Response::SUCCESS;
}

void TargetLayerComponent::initialize(std::shared_ptr<ParamsIO> paramsIO) {
   ProbeComponent::initialize(paramsIO);
}

void TargetLayerComponent::ioParam_targetLayer(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "targetLayer", &mTargetLayerName, false /*warnIfAbsentFlag*/);

   // If targetLayer is not present, check for targetName. targetName as a parameter for
   // layer probes was deprecated in favor of targetLayer on Oct 20, 2022, and marked obsolete on
   // Jul 3, 2025. Once targetName is removed as a synonym, the above code can simply call the
   // PVParams::ioParam() function.
   if (mTargetLayerName.empty()) {
      mParamsIO->ioParam(ioSwitch, "targetName", &mTargetLayerName, false /*warnIfAbsentFlag*/);
      FatalIf(
            !mTargetLayerName.empty(),
            "Probe %s parameter targetName is obsolete. "
            "Use targetLayer for layer probes instead.\n",
            getName_c());
   }
   FatalIf(
         mTargetLayerName.empty(),
         "Probe %s requires the targetLayer string parameter to be set\n",
         getName_c());
}

void TargetLayerComponent::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   ioParam_targetLayer(ioSwitch);
}

} // namespace PV
