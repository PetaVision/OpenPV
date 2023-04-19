#include "ProbeTriggerComponent.hpp"
#include <cassert>

namespace PV {

ColumnEnergyProbeTriggerComponent::ColumnEnergyProbeTriggerComponent(char const *objName, PVParams *params) {
   initialize(objName, params);
}
void ColumnEnergyProbeTriggerComponent::initialize(enum ParamsIOFlag ioFlag) {
   ProbeTriggerComponent::initialize(objName, params);
}

void ColumnEnergyProbeTriggerComponent::ioParam_reductionInterval(enum ParamsIOFlag ioFlag) {
   assert(!getParams()->presentAndNotBeenRead(getName(), "triggerLayerName"));
   if (getTriggerLayerName() == nullptr) {
      bool warnIfAbsent = false;
      parameters()->ioParamValue(
            ioFlag, name, "reductionInterval", &mSkipInterval, mSkipInterval, warnIfAbsent);
   }
}

void ColumnEnergyProbeTriggerComponent::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ProbeTriggerComponent::ioParamsFillGroup(ioFlag);
   ioParam_reductionInterval(ioFlag);
}

bool ColumnEnergyProbeTriggerComponent::needUpdate(double simTime, double deltaTime) {
   bool needUpdate;
   if (getTriggerLayerName() != nullptr) {
      needUpdate = ProbeTriggerComponent::needUpdate(simTime, deltaTime);
   }
   else {
      if (mReductionCount
      mReductionCounter--;
   }
   return needUpdate;
}

// namespace PV {
