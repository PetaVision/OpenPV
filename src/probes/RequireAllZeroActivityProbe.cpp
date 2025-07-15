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

RequireAllZeroActivityProbe::RequireAllZeroActivityProbe(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
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

void RequireAllZeroActivityProbe::createComponents(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   StatsProbeImmediate::createComponents(paramsIO, comm);
   createProbeCheckStats(paramsIO);
}

void RequireAllZeroActivityProbe::createProbeCheckStats(std::shared_ptr<ParamsIO> paramsIO) {
   mCheckStats = std::make_shared<CheckStatsAllZeros>(paramsIO);
}

void RequireAllZeroActivityProbe::createProbeLocal(std::shared_ptr<ParamsIO> paramsIO) {
   mProbeLocal = std::make_shared<ActivityBufferStatsProbeLocal>(paramsIO);
}

void RequireAllZeroActivityProbe::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   StatsProbeImmediate::initialize(paramsIO, comm);
}

int RequireAllZeroActivityProbe::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = StatsProbeImmediate::ioParamsFillGroup(ioSwitch);
   mCheckStats->ioParamsFillGroup(ioSwitch, mParamsIO);
   return status;
}

} // namespace PV
