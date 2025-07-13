#include "EnergyProbeComponent.hpp"
#include "utils/PVLog.hpp"
#include <cassert>
#include <cstdlib>

namespace PV {
EnergyProbeComponent::EnergyProbeComponent(
      std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   initialize(params, defaults);
}

EnergyProbeComponent::~EnergyProbeComponent() {}

void EnergyProbeComponent::initialize(
      std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   ProbeComponent::initialize(params, defaults);
}

Response::Status EnergyProbeComponent::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   if (mEnergyProbeName.empty() or mEnergyProbe != nullptr) {
      return Response::NO_ACTION;
   }

   auto *objectTable = message->mObjectTable;
   mEnergyProbe      = objectTable->findObject<ColumnEnergyProbe>(mEnergyProbeName);
   FatalIf(
         mEnergyProbe == nullptr,
         "Probe %s energyProbe \"%s\" does not exist or is not a ColumnEnergyProbe.\n",
         getName_c(),
         mEnergyProbeName.c_str());
   return Response::SUCCESS;
}

void EnergyProbeComponent::ioParam_coefficient(ParamsIOSwitch ioSwitch) {
   assert(!mParamsIO->presentAndNotBeenRead("energyProbe"));
   if (!mEnergyProbeName.empty()) {
      mParamsIO->ioParam(ioSwitch, "coefficient", &mCoefficient);
   }
}

void EnergyProbeComponent::ioParam_energyProbe(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "energyProbe", &mEnergyProbeName, false /*warnIfAbsentFlag*/);
}

void EnergyProbeComponent::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   ioParam_energyProbe(ioSwitch);
   ioParam_coefficient(ioSwitch);
}

} // namespace PV
