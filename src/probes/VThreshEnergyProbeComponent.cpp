#include "VThreshEnergyProbeComponent.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "utils/PVLog.hpp"
#include <cassert>
#include <cstdlib>

namespace PV {
VThreshEnergyProbeComponent::VThreshEnergyProbeComponent(char const *objName, PVParams *params) {
   initialize(objName, params);
}

void VThreshEnergyProbeComponent::initialize(char const *objName, PVParams *params) {
   EnergyProbeComponent::initialize(objName, params);
}

Response::Status VThreshEnergyProbeComponent::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   return EnergyProbeComponent::communicateInitInfo(message);
}

void VThreshEnergyProbeComponent::initializeState(HyPerLayer *targetLayer) {
   auto *activityComponent = targetLayer->getComponentByType<ActivityComponent>();
   FatalIf(
         activityComponent == nullptr,
         "Probe %s: targetLayer \"%s\" does not have an activity component.\n",
         getName_c(),
         targetLayer->getName());
   ANNActivityBuffer *activityBuffer = activityComponent->getComponentByType<ANNActivityBuffer>();
   FatalIf(
         activityBuffer == nullptr,
         "Probe %s: targetLayer \"%s\" does not have an ANNActivityBuffer component.\n",
         getName_c(),
         targetLayer->getName());
   FatalIf(
         activityBuffer->usingVerticesListInParams() == true,
         "Probe %s: LCAProbes require target layer \"%s\" to use VThresh etc. "
         "instead of verticesV/verticesA.\n",
         getName_c(),
         targetLayer->getName());
   setCoefficient(activityBuffer->getVThresh());
}

void VThreshEnergyProbeComponent::ioParam_coefficient(enum ParamsIOFlag ioFlag) {
   assert(!getParams()->presentAndNotBeenRead(getName_c(), "energyProbe"));
   // Should call getParams()->handleUnnecessaryParameters() here, if energyProbe is defined.
}

} // namespace PV
