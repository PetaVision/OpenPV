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
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

GPUSystemTestProbe::~GPUSystemTestProbe() {}

void GPUSystemTestProbe::createProbeCheckStats(
      std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   mCheckStats = std::make_shared<CheckStatsAllZerosCheckSigma>(params, defaults);
}

void GPUSystemTestProbe::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   RequireAllZeroActivityProbe::initialize(params, defaults, comm);
}

} // end namespace PV
