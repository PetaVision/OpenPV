#include "L0NormLCAEnergyProbeComponent.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "utils/PVLog.hpp"
#include <cassert>
#include <cstdlib>

namespace PV {
L0NormLCAEnergyProbeComponent::L0NormLCAEnergyProbeComponent(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults) {
   initialize(params, defaults);
}

void L0NormLCAEnergyProbeComponent::initialize(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   EnergyProbeComponent::initialize(params, defaults);
}

Response::Status L0NormLCAEnergyProbeComponent::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   return EnergyProbeComponent::communicateInitInfo(message);
}

void L0NormLCAEnergyProbeComponent::initializeState(HyPerLayer *targetLayer) {
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
   double VThresh     = static_cast<double>(activityBuffer->getVThresh());
   double coefficient = 0.5 * VThresh * VThresh;
   setCoefficient(coefficient);
}

void L0NormLCAEnergyProbeComponent::ioParam_coefficient(ParamsIOSwitch ioSwitch) {
   assert(!mParamsIO->presentAndNotBeenRead("energyProbe"));
   // Should call mParamsIO->handleUnnecessaryParameters() here, if energyProbe is defined.
}

} // namespace PV
