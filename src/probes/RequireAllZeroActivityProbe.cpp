/*
 * RequireAllZeroActivityProbe.cpp
 *
 *  Created on: Mar 26, 2014
 *      Author: pschultz
 */

#include "RequireAllZeroActivityProbe.hpp"

#include "columns/Communicator.hpp"
#include "io/PVParams.hpp"
#include "probes/ActivityBufferStatsProbeLocal.hpp"
#include "probes/ProbeData.hpp"
#include "probes/StatsProbeTypes.hpp"

#include <stdexcept>
#include <string>

namespace PV {

RequireAllZeroActivityProbe::RequireAllZeroActivityProbe(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
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
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   StatsProbeImmediate::createComponents(name, params, comm);
   createProbeCheckStats(name, params);
}

void RequireAllZeroActivityProbe::createProbeCheckStats(char const *name, PVParams *params) {
   mCheckStats = std::make_shared<CheckStatsAllZeros>(name, params);
}

void RequireAllZeroActivityProbe::createProbeLocal(char const *name, PVParams *params) {
   mProbeLocal = std::make_shared<ActivityBufferStatsProbeLocal>(name, params);
}

void RequireAllZeroActivityProbe::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   StatsProbeImmediate::initialize(name, params, comm);
}

int RequireAllZeroActivityProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = StatsProbeImmediate::ioParamsFillGroup(ioFlag);
   mCheckStats->ioParamsFillGroup(ioFlag);
   return status;
}

} // namespace PV
