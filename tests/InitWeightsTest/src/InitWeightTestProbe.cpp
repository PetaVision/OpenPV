/*
 * InitWeightTestProbe.cpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#include "InitWeightTestProbe.hpp"
#include <probes/ActivityBufferStatsProbeLocal.hpp>
#include <probes/ProbeData.hpp>
#include <probes/StatsProbeTypes.hpp>
#include <utils/PVLog.hpp>

#include <cstdlib>
#include <memory>

namespace PV {

InitWeightTestProbe::InitWeightTestProbe(
      const char *name,
      PVParams *params,
      Communicator const *comm)
      : StatsProbeImmediate() {
   initialize(name, params, comm);
}

void InitWeightTestProbe::checkStats() {
   const int rootProc = 0;
   if (mCommunicator->commRank() != rootProc) {
      return;
   }
   auto const &storedValues           = mProbeAggregator->getStoredValues();
   auto numTimestamps                 = storedValues.size();
   int lastTimestampIndex             = static_cast<int>(numTimestamps) - 1;
   ProbeData<LayerStats> const &stats = storedValues.getData(lastTimestampIndex);
   int nbatch                         = static_cast<int>(stats.size());
   double simTime                     = stats.getTimestamp();
   if (simTime > 2.0) {
      for (int b = 0; b < nbatch; b++) {
         LayerStats const &statsElem = stats.getValue(b);
         FatalIf(std::abs(statsElem.mMin) >= 0.001f, "Test failed.\n");
         FatalIf(std::abs(statsElem.mMax) >= 0.001f, "Test failed.\n");
         FatalIf(std::abs(statsElem.average()) >= 0.001, "Test failed.\n");
      }
   }
}

void InitWeightTestProbe::createProbeLocal(char const *name, PVParams *params) {
   mProbeLocal = std::make_shared<ActivityBufferStatsProbeLocal>(name, params);
}

void InitWeightTestProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   StatsProbeImmediate::initialize(name, params, comm);
}

} /* namespace PV */
