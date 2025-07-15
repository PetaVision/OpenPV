/*
 * KernelTestProbe.cpp
 *
 *  Created on: Sep 1, 2011
 *      Author: gkenyon
 */

#include "KernelTestProbe.hpp"
#include <probes/ActivityBufferStatsProbeLocal.hpp>
#include <probes/ProbeData.hpp>
#include <probes/StatsProbeTypes.hpp>
#include <utils/PVLog.hpp>

#include <cstdlib>
#include <memory>

namespace PV {

KernelTestProbe::KernelTestProbe(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm)
      : StatsProbeImmediate() {
   initialize(paramsIO, comm);
}

void KernelTestProbe::checkStats() {
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
   if (simTime > 2.0) {
      for (int b = 0; b < nbatch; b++) {
         LayerStats const &statsElem = stats.getValue(b);
         FatalIf(std::abs(statsElem.mMin - 1.00f) >= 0.01f, "Test failed.\n");
         FatalIf(std::abs(statsElem.mMax - 1.00f) >= 0.01f, "Test failed.\n");
         FatalIf(std::abs(statsElem.average() - 1.00) >= 0.01, "Test failed.\n");
      }
   }
}

void KernelTestProbe::createProbeLocal(std::shared_ptr<ParamsIO> paramsIO) {
   mProbeLocal = std::make_shared<ActivityBufferStatsProbeLocal>(paramsIO);
}

void KernelTestProbe::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   StatsProbeImmediate::initialize(paramsIO, comm);
}

} /* namespace PV */
