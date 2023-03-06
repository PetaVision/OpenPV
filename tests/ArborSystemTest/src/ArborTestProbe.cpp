/*
 * ArborTestProbe.cpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#include "ArborTestProbe.hpp"
#include <include/pv_common.h>
#include <probes/ActivityBufferStatsProbeLocal.hpp>
#include <probes/ProbeData.hpp>
#include <probes/StatsProbeTypes.hpp>
#include <utils/PVLog.hpp>

#include <cstdlib>
#include <memory>

namespace PV {

ArborTestProbe::ArborTestProbe(const char *name, PVParams *params, Communicator const *comm)
      : StatsProbeImmediate() {
   initialize(name, params, comm);
}

ArborTestProbe::~ArborTestProbe() {}

void ArborTestProbe::checkStats() {
   if (mCommunicator->commRank() != 0) {
      return;
   }
   auto const &storedValues           = mProbeAggregator->getStoredValues();
   auto numTimestamps                 = storedValues.size();
   int lastTimestampIndex             = static_cast<int>(numTimestamps) - 1;
   ProbeData<LayerStats> const &stats = storedValues.getData(lastTimestampIndex);

   double simTime = stats.getTimestamp();
   int status     = PV_SUCCESS;
   int nbatch     = static_cast<int>(stats.size());
   for (int b = 0; b < nbatch; b++) {
      LayerStats const &statsElem = stats.getValue(b);
      double average              = statsElem.average();
      if (simTime == 1.0 and std::abs(average - 0.25) >= 0.0001) {
         ErrorLog().printf(
               "%s: t=1.0, batch index %d had average %f instead of expected 0.25.\n",
               getDescription_c(),
               b,
               average);
         status = PV_FAILURE;
      }
      else if (simTime == 2.0 and std::abs(average - 0.50) >= 0.0001) {
         ErrorLog().printf(
               "%s: t=2.0, batch index %d had average %f instead of expected 0.50.\n",
               getDescription_c(),
               b,
               average);
         status = PV_FAILURE;
      }
      else if (simTime == 3.0 and std::abs(average - 0.75) >= 0.0001) {
         ErrorLog().printf(
               "%s: t=3.0, batch index %d had average %f instead of expected 0.75.\n",
               getDescription_c(),
               b,
               average);
         status = PV_FAILURE;
      }
      else if (simTime > 3.0) {
         if (std::abs(statsElem.mMin - 1.00f) >= 0.0001f) {
            ErrorLog().printf(
                  "%s: t=%.1f, batch index %d had minimum %f instead of expected 1.00.\n",
                  getDescription_c(),
                  simTime,
                  b,
                  (double)statsElem.mMin);
            status = PV_FAILURE;
         }
         if (std::abs(statsElem.mMax - 1.00f) >= 0.0001f) {
            ErrorLog().printf(
                  "%s: t=%.1f, batch index %d had maximum %f instead of expected 1.00.\n",
                  getDescription_c(),
                  simTime,
                  b,
                  (double)statsElem.mMax);
            status = PV_FAILURE;
         }
         if (std::abs(average - 1.00) >= 0.0001) {
            ErrorLog().printf(
                  "%s: t=%.1f, batch index %d had average %f instead of expected 1.00.\n",
                  getDescription_c(),
                  simTime,
                  b,
                  average);
            status = PV_FAILURE;
         }
      }
   }
   FatalIf(status != PV_SUCCESS, "Test failed.\n");
}

void ArborTestProbe::createProbeLocal(char const *name, PVParams *params) {
   mProbeLocal = std::make_shared<ActivityBufferStatsProbeLocal>(name, params);
}

void ArborTestProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   StatsProbeImmediate::initialize(name, params, comm);
}

} /* namespace PV */
