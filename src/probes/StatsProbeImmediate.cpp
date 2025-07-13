/*
 * StatsProbeImmediate.cpp
 */

#include "StatsProbeImmediate.hpp"

#include "columns/Communicator.hpp"

namespace PV {

StatsProbeImmediate::StatsProbeImmediate(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

StatsProbeImmediate::StatsProbeImmediate() {}

StatsProbeImmediate::~StatsProbeImmediate() {}

void StatsProbeImmediate::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   StatsProbe::initialize(params, defaults, comm);
}

void StatsProbeImmediate::ioParam_immediateMPIAssembly(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read) {
      setImmediateMPIAssembly(true);
      mParamsIO->handleUnnecessaryParameter("immediateMPIAssembly", getImmediateMPIAssembly());
   }
}

} // namespace PV
