#include "ProbeTriggerComponent.hpp"
#include "layers/HyPerLayer.hpp"
#include "observerpattern/ObserverTable.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"

namespace PV {

ProbeTriggerComponent::ProbeTriggerComponent(char const *objName, PVParams *params) {
   initialize(objName, params);
}

ProbeTriggerComponent::~ProbeTriggerComponent() { free(mTriggerLayerName); }

Response::Status ProbeTriggerComponent::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   if (mTriggerLayerFlag and !mTriggerControl) {
      auto triggerLayer = message->mObjectTable->findObject<HyPerLayer>(mTriggerLayerName);
      FatalIf(
            triggerLayer == nullptr,
            "Probe %s triggerLayer \"%s\" is not a layer in the HyPerCol.\n",
            getName_c(),
            mTriggerLayerName);
      mTriggerControl = triggerLayer->getComponentByType<LayerUpdateController>();
      FatalIf(
            mTriggerControl == nullptr,
            "Probe %s triggerLayer \"%s\" does not have a LayerUpdateController component.\n",
            getName_c(),
            mTriggerLayerName);
   }
   return Response::SUCCESS;
}

void ProbeTriggerComponent::initialize(char const *objName, PVParams *params) {
   ProbeComponent::initialize(objName, params);
}

void ProbeTriggerComponent::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   bool warnIfAbsent = false;
   getParams()->ioParamString(
         ioFlag, getName_c(), "triggerLayerName", &mTriggerLayerName, nullptr, warnIfAbsent);
   if (ioFlag == PARAMS_IO_READ) {
      mTriggerLayerFlag = (mTriggerLayerName != nullptr && mTriggerLayerName[0] != '\0');
   }
}

void ProbeTriggerComponent::ioParam_triggerOffset(enum ParamsIOFlag ioFlag) {
   pvAssert(!getParams()->presentAndNotBeenRead(getName_c(), "triggerLayerName"));
   if (mTriggerLayerFlag) {
      getParams()->ioParamValue(
            ioFlag, getName_c(), "triggerOffset", &mTriggerOffset, mTriggerOffset);
      if (mTriggerOffset < 0) {
         Fatal().printf(
               "%s \"%s\" error: TriggerOffset (%f) must be positive\n",
               getParams()->groupKeywordFromName(getName_c()),
               getName_c(),
               mTriggerOffset);
      }
   }
}

void ProbeTriggerComponent::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_triggerLayerName(ioFlag);
   ioParam_triggerOffset(ioFlag);
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
