#include "ProbeTriggerComponent.hpp"
#include "layers/HyPerLayer.hpp"
#include "observerpattern/ObserverTable.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"

namespace PV {

ProbeTriggerComponent::ProbeTriggerComponent(std::shared_ptr<ParamsIO> paramsIO) {
   initialize(paramsIO);
}

ProbeTriggerComponent::~ProbeTriggerComponent() {}

Response::Status ProbeTriggerComponent::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   if (mTriggerLayerFlag and !mTriggerControl) {
      auto triggerLayer = message->mObjectTable->findObject<HyPerLayer>(mTriggerLayerName);
      FatalIf(
            triggerLayer == nullptr,
            "Probe %s triggerLayer \"%s\" is not a layer in the HyPerCol.\n",
            getName_c(),
            mTriggerLayerName.c_str());
      mTriggerControl = triggerLayer->getComponentByType<LayerUpdateController>();
      FatalIf(
            mTriggerControl == nullptr,
            "Probe %s triggerLayer \"%s\" does not have a LayerUpdateController component.\n",
            getName_c(),
            mTriggerLayerName.c_str());
   }
   return Response::SUCCESS;
}

void ProbeTriggerComponent::initialize(std::shared_ptr<ParamsIO> paramsIO) {
   ProbeComponent::initialize(paramsIO);
}

void ProbeTriggerComponent::ioParam_triggerLayerName(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "triggerLayerName", &mTriggerLayerName, false /*warnIfAbsentFlag*/);
   if (ioSwitch == ParamsIOSwitch::Read) {
      mTriggerLayerFlag = (!mTriggerLayerName.empty());
   }
}

void ProbeTriggerComponent::ioParam_triggerOffset(ParamsIOSwitch ioSwitch) {
   pvAssert(!mParamsIO->presentAndNotBeenRead("triggerLayerName"));
   if (mTriggerLayerFlag) {
      mParamsIO->ioParam(ioSwitch, "triggerOffset", &mTriggerOffset);
      if (mTriggerOffset < 0) {
         Fatal().printf(
               "%s \"%s\" error: TriggerOffset (%f) must be positive\n",
               mParamsIO->getKeyword().c_str(),
               getName_c(),
               mTriggerOffset);
      }
   }
}

void ProbeTriggerComponent::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   ioParam_triggerLayerName(ioSwitch);
   ioParam_triggerOffset(ioSwitch);
}

bool ProbeTriggerComponent::needUpdate(double simTime, double deltaTime) {
   bool needUpdate;
   if (mTriggerControl) {
      needUpdate = mTriggerControl->needUpdate(simTime + mTriggerOffset, deltaTime);
   }
   else {
      needUpdate = true;
   }
   return needUpdate;
}

} // end namespace PV
