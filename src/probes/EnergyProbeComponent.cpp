#include "EnergyProbeComponent.hpp"
#include "utils/PVLog.hpp"
#include <cassert>
#include <cstdlib>

namespace PV {
EnergyProbeComponent::EnergyProbeComponent(char const *objName, PVParams *params) {
   initialize(objName, params);
}

EnergyProbeComponent::~EnergyProbeComponent() { free(mEnergyProbeName); }

void EnergyProbeComponent::initialize(char const *objName, PVParams *params) {
   ProbeComponent::initialize(objName, params);
}

Response::Status EnergyProbeComponent::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   if (mEnergyProbeName == nullptr or mEnergyProbe != nullptr) {
      return Response::NO_ACTION;
   }

   auto *objectTable = message->mObjectTable;
   mEnergyProbe      = objectTable->findObject<ColumnEnergyProbe>(mEnergyProbeName);
   FatalIf(
         mEnergyProbe == nullptr,
         "Probe %s energyProbe \"%s\" does not exist or is not a ColumnEnergyProbe.\n",
         getName_c(),
         mEnergyProbeName);
   return Response::SUCCESS;
}

void EnergyProbeComponent::ioParam_coefficient(enum ParamsIOFlag ioFlag) {
   assert(!getParams()->presentAndNotBeenRead(getName_c(), "energyProbe"));
   if (mEnergyProbeName and mEnergyProbeName[0]) {
      getParams()->ioParamValue(
            ioFlag, getName_c(), "coefficient", &mCoefficient, mCoefficient, true /*warnIfAbsent*/);
   }
}

void EnergyProbeComponent::ioParam_energyProbe(enum ParamsIOFlag ioFlag) {
   getParams()->ioParamString(
         ioFlag, getName_c(), "energyProbe", &mEnergyProbeName, NULL, false /*warnIfAbsent*/);
}

void EnergyProbeComponent::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_energyProbe(ioFlag);
   ioParam_coefficient(ioFlag);
}

} // namespace PV
