/*
 * GPUSystemTestProbe.cpp
 * Author: slundquist
 */

#include "GPUSystemTestProbe.hpp"

#include "CheckStatsAllZerosCheckSigma.hpp"
#include <probes/ActivityBufferStatsProbeLocal.hpp>

#include <memory>

namespace PV {
GPUSystemTestProbe::GPUSystemTestProbe(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

GPUSystemTestProbe::~GPUSystemTestProbe() {}

void GPUSystemTestProbe::createProbeCheckStats(char const *name, PVParams *params) {
   mCheckStats = std::make_shared<CheckStatsAllZerosCheckSigma>(name, params);
}

void GPUSystemTestProbe::createProbeLocal(char const *name, PVParams *params) {
   mProbeLocal = std::make_shared<ActivityBufferStatsProbeLocal>(name, params);
}

void GPUSystemTestProbe::initialize(char const *name, PVParams *params, Communicator const *comm) {
   RequireAllZeroActivityProbe::initialize(name, params, comm);
}

} // end namespace PV
