/*
 * GPUSystemTestProbe.cpp
 * Author: slundquist
 */

#include "GPUSystemTestProbe.hpp"

#include "CheckStatsAllZerosCheckSigma.hpp"
#include <probes/ActivityBufferStatsProbeLocal.hpp>

#include <memory>

namespace PV {
GPUSystemTestProbe::GPUSystemTestProbe(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

GPUSystemTestProbe::~GPUSystemTestProbe() {}

void GPUSystemTestProbe::createProbeCheckStats(std::shared_ptr<ParamsIO> paramsIO) {
   mCheckStats = std::make_shared<CheckStatsAllZerosCheckSigma>(paramsIO);
}

void GPUSystemTestProbe::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   RequireAllZeroActivityProbe::initialize(paramsIO, comm);
}

} // end namespace PV
