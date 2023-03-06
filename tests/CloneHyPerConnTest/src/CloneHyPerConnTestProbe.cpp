/*
 * CloneHyPerConnTestProbe.cpp
 *
 *  Created on: Feb 24, 2012
 *      Author: peteschultz
 */

#include "CloneHyPerConnTestProbe.hpp"

#include <include/pv_common.h>
#include <probes/ProbeData.hpp>
#include <probes/StatsProbeTypes.hpp>
#include <utils/PVLog.hpp>

#include <cstdlib>

namespace PV {

CloneHyPerConnTestProbe::CloneHyPerConnTestProbe(
      const char *name,
      PVParams *params,
      Communicator const *comm)
      : StatsProbeImmediate() {
   initialize(name, params, comm);
}

void CloneHyPerConnTestProbe::checkStats() {
   if (mCommunicator->commRank() != 0) {
      return;
   }
   auto const &storedValues           = mProbeAggregator->getStoredValues();
   auto numTimestamps                 = storedValues.size();
   int lastTimestampIndex             = static_cast<int>(numTimestamps) - 1;
   ProbeData<LayerStats> const &stats = storedValues.getData(lastTimestampIndex);

   double simTime   = stats.getTimestamp();
   int status       = PV_SUCCESS;
   int const nbatch = stats.size();
   for (int b = 0; b < nbatch; ++b) {
      if (simTime > 2.0) {
         LayerStats const &statsElem = stats.getValue(b);
         if (std::abs(statsElem.mMin) >= 1.0e-6f) {
            ErrorLog().printf(
                  "%s: t=%f, batch index %d had minimum %f instead of expected zero.\n",
                  getDescription_c(),
                  simTime,
                  b,
                  (double)statsElem.mMin);
            status = PV_FAILURE;
         }
         if (std::abs(statsElem.mMax) >= 1.0e-6f) {
            ErrorLog().printf(
                  "%s: t=%f, batch index %d had maximum %f instead of expected zero.\n",
                  getDescription_c(),
                  simTime,
                  b,
                  (double)statsElem.mMax);
            status = PV_FAILURE;
         }
         if (std::abs(statsElem.average()) >= 1.0e-6) {
            ErrorLog().printf(
                  "%s: t=%f, batch index %d had average %f instead of expected zero.\n",
                  getDescription_c(),
                  simTime,
                  b,
                  statsElem.average());
            status = PV_FAILURE;
         }
      }
   }
   FatalIf(status != PV_SUCCESS, "Test failed.\n");
}

void CloneHyPerConnTestProbe::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   StatsProbeImmediate::initialize(name, params, comm);
}

} /* namespace PV */
