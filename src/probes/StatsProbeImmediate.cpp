/*
 * StatsProbeImmediate.cpp
 */

#include "StatsProbeImmediate.hpp"

#include "columns/Communicator.hpp"

namespace PV {

StatsProbeImmediate::StatsProbeImmediate(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

StatsProbeImmediate::StatsProbeImmediate() {}

StatsProbeImmediate::~StatsProbeImmediate() {}

void StatsProbeImmediate::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   StatsProbe::initialize(paramsIO, comm);
}

void StatsProbeImmediate::ioParam_immediateMPIAssembly(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read) {
      setImmediateMPIAssembly(true);
      mParamsIO->handleUnnecessaryParameter("immediateMPIAssembly", getImmediateMPIAssembly());
   }
}

} // namespace PV
