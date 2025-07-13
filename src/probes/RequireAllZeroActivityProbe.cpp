/*
 * RequireAllZeroActivityProbe.cpp
 *
 *  Created on: Mar 26, 2014
 *      Author: pschultz
 */

#include "RequireAllZeroActivityProbe.hpp"

#include "columns/Communicator.hpp"
#include "probes/ActivityBufferStatsProbeLocal.hpp"
#include "probes/ProbeData.hpp"
#include "probes/StatsProbeTypes.hpp"

#include <stdexcept>
#include <string>

namespace PV {

RequireAllZeroActivityProbe::RequireAllZeroActivityProbe(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

RequireAllZeroActivityProbe::RequireAllZeroActivityProbe() {}

RequireAllZeroActivityProbe::~RequireAllZeroActivityProbe() {}

void RequireAllZeroActivityProbe::checkStats() {
   int storedValuesSize = static_cast<int>(mProbeAggregator->getStoredValues().size());
   int backIndex        = storedValuesSize - 1;
   if (backIndex >= 0) {
      ProbeData<LayerStats> const &stats = mProbeAggregator->getStoredValues().getData(backIndex);
      mCheckStats->checkStats(stats);
      if (mCheckStats->foundNonzero()) {
         std::string errorMessage(getDescription());
         errorMessage.append(" found a nonzero value outside of tolerance ");
         errorMessage.append(std::to_string(mProbeLocal->getNnzThreshold()));
      }
   }
}

Response::Status RequireAllZeroActivityProbe::cleanup() {
   mCheckStats->cleanup();
   if (mCheckStats->foundNonzero()) {
      std::string errorMessage(getDescription());
      errorMessage.append(" found nonzero value outside of tolerance ");
      errorMessage.append(std::to_string(mProbeLocal->getNnzThreshold()));
      mStatus = PV_FAILURE;
   }
   return Response::SUCCESS;
}

void RequireAllZeroActivityProbe::createComponents(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   StatsProbeImmediate::createComponents(params, defaults, comm);
   createProbeCheckStats(params, defaults);
}

void RequireAllZeroActivityProbe::createProbeCheckStats(
      std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   mCheckStats = std::make_shared<CheckStatsAllZeros>(params, defaults);
}

void RequireAllZeroActivityProbe::createProbeLocal(
      std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   mProbeLocal = std::make_shared<ActivityBufferStatsProbeLocal>(params, defaults);
}

void RequireAllZeroActivityProbe::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   StatsProbeImmediate::initialize(params, defaults, comm);
}

int RequireAllZeroActivityProbe::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = StatsProbeImmediate::ioParamsFillGroup(ioSwitch);
   mCheckStats->ioParamsFillGroup(ioSwitch, mParamsIO);
   return status;
}

} // namespace PV
