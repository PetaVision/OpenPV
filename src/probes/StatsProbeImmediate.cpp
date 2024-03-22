/*
 * StatsProbeImmediate.cpp
 */

#include "StatsProbeImmediate.hpp"

#include "columns/Communicator.hpp"
#include "io/PVParams.hpp"

namespace PV {

StatsProbeImmediate::StatsProbeImmediate(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

StatsProbeImmediate::StatsProbeImmediate() {}

StatsProbeImmediate::~StatsProbeImmediate() {}

void StatsProbeImmediate::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   StatsProbe::initialize(name, params, comm);
}

void StatsProbeImmediate::ioParam_immediateMPIAssembly(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      setImmediateMPIAssembly(true);
      parameters()->handleUnnecessaryParameter(
            getName(), "immediateMPIAssembly", getImmediateMPIAssembly());
   }
}

} // namespace PV
