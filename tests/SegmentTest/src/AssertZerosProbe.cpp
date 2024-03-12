/*
 * AssertZerosProbe.cpp
 * Author: slundquist
 */

#include "AssertZerosProbe.hpp"
#include "components/BasePublisherComponent.hpp"
#include "components/LayerInputBuffer.hpp"
#include "include/PVLayerLoc.hpp"
#include "include/pv_types.h"
#include "layers/HyPerLayer.hpp"
#include "probes/ActivityBufferStatsProbeLocal.hpp"
#include "probes/ProbeData.hpp"
#include "probes/StatsProbeTypes.hpp"
#include "utils/PVLog.hpp"

#include <cmath>
#include <memory>

namespace PV {
AssertZerosProbe::AssertZerosProbe(const char *name, PVParams *params, Communicator const *comm)
      : StatsProbeImmediate() {
   initialize(name, params, comm);
}

// 2 tests: max difference can be 5e-4, max std is 5e-5
void AssertZerosProbe::checkStats() {
   auto *targetLayerInputBuffer = getTargetLayer()->getComponentByType<LayerInputBuffer>();
   auto *targetPublisher        = getTargetLayer()->getComponentByType<BasePublisherComponent>();
   const PVLayerLoc *loc        = getTargetLayer()->getLayerLoc();
   int numExtNeurons            = targetPublisher->getNumExtended() * loc->nbatch;
   int numResNeurons            = targetLayerInputBuffer->getBufferSizeAcrossBatch();
   const float *A               = targetPublisher->getLayerData();
   const float *GSyn_E          = targetLayerInputBuffer->getChannelData(CHANNEL_EXC);
   const float *GSyn_I          = targetLayerInputBuffer->getChannelData(CHANNEL_INH);
   auto const &storedValues     = mProbeAggregator->getStoredValues();
   auto numTimestamps           = storedValues.size();
   int lastTimestampIndex       = static_cast<int>(numTimestamps) - 1;
   ProbeData<LayerStats> const &stats = storedValues.getData(lastTimestampIndex);
   double simTime                     = stats.getTimestamp();

   for (int i = 0; i < numExtNeurons; i++) {
      FatalIf(fabsf(A[i]) >= 5e-4f, "Test failed.\n");
   }

   if (simTime > 0) {
      // Make sure gsyn_e and gsyn_i are not all 0's
      float sum_E = 0;
      float sum_I = 0;
      for (int i = 0; i < numResNeurons; i++) {
         sum_E += GSyn_E[i];
         sum_I += GSyn_I[i];
      }

      FatalIf(sum_E == 0, "Test failed.\n");
      FatalIf(sum_I == 0, "Test failed.\n");
   }

   for (int b = 0; b < loc->nbatch; b++) {
      // For max std of 5e-5
      LayerStats const &statsElem = stats.getValue(b);
      FatalIf(
            statsElem.sigma() > 5e-5,
            "%s: t=%f, batch index %d has sigma %f, greater than allowed tolerance %f.\n",
            getDescription_c(),
            simTime,
            b,
            statsElem.sigma(),
            5e-5);
   }
}

void AssertZerosProbe::createProbeLocal(char const *name, PVParams *params) {
   mProbeLocal = std::make_shared<ActivityBufferStatsProbeLocal>(name, params);
}

void AssertZerosProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   StatsProbeImmediate::initialize(name, params, comm);
}

} // end namespace PV
