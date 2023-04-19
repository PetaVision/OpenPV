/*
 * ParameterSweepTestProbe.cpp
 *
 *  Created on: Aug 13, 2012
 *      Author: pschultz
 */

#include "ParameterSweepTestProbe.hpp"
#include "include/pv_common.h"
#include "probes/ActivityBufferStatsProbeLocal.hpp"
#include "probes/ProbeData.hpp"
#include "probes/StatsProbeTypes.hpp"
#include <utils/PVLog.hpp>

#include <cmath>
#include <memory>

namespace PV {

ParameterSweepTestProbe::ParameterSweepTestProbe(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

ParameterSweepTestProbe::~ParameterSweepTestProbe() {}

void ParameterSweepTestProbe::checkStats() {
   const int rootProc = 0;
   if (mCommunicator->commRank() != rootProc) {
      return;
   }
   auto const &storedValues           = mProbeAggregator->getStoredValues();
   auto numTimestamps                 = storedValues.size();
   int lastTimestampIndex             = static_cast<int>(numTimestamps) - 1;
   ProbeData<LayerStats> const &stats = storedValues.getData(lastTimestampIndex);
   double simTime                     = stats.getTimestamp();
   int status                         = PV_SUCCESS;
   int nbatch                         = static_cast<int>(stats.size());
   if (simTime >= 3.0) {
      for (int b = 0; b < nbatch; b++) {
         LayerStats const &statsElem = stats.getValue(b);
         FatalIf(std::fabs(mExpectedSum - statsElem.mSum) >= 1e-6, "Test failed.\n");
         FatalIf(std::fabs(mExpectedMin - statsElem.mMin) >= 1e-6f, "Test failed.\n");
         FatalIf(std::fabs(mExpectedMax - statsElem.mMax) >= 1e-6f, "Test failed.\n");
      }
   }
}

void ParameterSweepTestProbe::createProbeLocal(char const *name, PVParams *params) {
   mProbeLocal = std::make_shared<ActivityBufferStatsProbeLocal>(name, params);
}

void ParameterSweepTestProbe::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   StatsProbeImmediate::initialize(name, params, comm);
}

int ParameterSweepTestProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = StatsProbeImmediate::ioParamsFillGroup(ioFlag);
   ioParam_expectedSum(ioFlag);
   ioParam_expectedMin(ioFlag);
   ioParam_expectedMax(ioFlag);
   return status;
}

void ParameterSweepTestProbe::ioParam_expectedSum(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "expectedSum", &mExpectedSum, mExpectedSum);
}
void ParameterSweepTestProbe::ioParam_expectedMin(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "expectedMin", &mExpectedMin, mExpectedMin);
}

void ParameterSweepTestProbe::ioParam_expectedMax(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "expectedMax", &mExpectedMax, mExpectedMax);
}

} /* namespace PV */
