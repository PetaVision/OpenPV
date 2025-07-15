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

ParameterSweepTestProbe::ParameterSweepTestProbe(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
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

void ParameterSweepTestProbe::createProbeLocal(std::shared_ptr<ParamsIO> paramsIO) {
   mProbeLocal = std::make_shared<ActivityBufferStatsProbeLocal>(paramsIO);
}

void ParameterSweepTestProbe::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   StatsProbeImmediate::initialize(paramsIO, comm);
}

int ParameterSweepTestProbe::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = StatsProbeImmediate::ioParamsFillGroup(ioSwitch);
   ioParam_expectedSum(ioSwitch);
   ioParam_expectedMin(ioSwitch);
   ioParam_expectedMax(ioSwitch);
   return status;
}

void ParameterSweepTestProbe::ioParam_expectedSum(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "expectedSum", &mExpectedSum);
}
void ParameterSweepTestProbe::ioParam_expectedMin(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "expectedMin", &mExpectedMin);
}

void ParameterSweepTestProbe::ioParam_expectedMax(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "expectedMax", &mExpectedMax);
}

} /* namespace PV */
