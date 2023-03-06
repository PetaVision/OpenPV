/*
 * ArborTestForOnesProbe.cpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#include "ArborTestForOnesProbe.hpp"
#include <include/pv_common.h>
#include <probes/StatsProbeTypes.hpp>
#include <utils/PVLog.hpp>

namespace PV {

ArborTestForOnesProbe::ArborTestForOnesProbe(
      const char *name,
      PVParams *params,
      Communicator const *comm)
      : StatsProbeImmediate() {
   initialize(name, params, comm);
}

ArborTestForOnesProbe::~ArborTestForOnesProbe() {}

void ArborTestForOnesProbe::checkStats() {
   int status               = PV_SUCCESS;
   auto const &storedValues = mProbeAggregator->getStoredValues();
   int lastTimestampIndex   = static_cast<int>(storedValues.size()) - 1;
   auto const &stats        = storedValues.getData(lastTimestampIndex);
   double simTime           = stats.getTimestamp();
   if (simTime > 1.0) {
      for (int b = 0; b < stats.size(); ++b) {
         LayerStats const &statsElem = stats.getValue(b);
         if (checkValue(statsElem.mMin, simTime, b, "minimum") != PV_SUCCESS) {
            status = PV_FAILURE;
         }
         if (checkValue(statsElem.mMax, simTime, b, "maximum") != PV_SUCCESS) {
            status = PV_FAILURE;
         }
         float average = static_cast<float>(statsElem.average());
         if (checkValue(average, simTime, b, "average") != PV_SUCCESS) {
            status = PV_FAILURE;
         }
      }
   }
   FatalIf(status != PV_SUCCESS, "Test failed.\n");
}

int ArborTestForOnesProbe::checkValue(
      float value,
      double timestamp,
      int batchIndex,
      char const *desc) {
   if (value <= 0.9999f or value >= 1.0001f) {
      ErrorLog().printf(
            "%s: t=%f, batch index %d had %s value %f instead of 1.\n",
            getDescription_c(),
            timestamp,
            batchIndex,
            desc,
            (double)value);
      return PV_FAILURE;
   }
   return PV_SUCCESS;
}

void ArborTestForOnesProbe::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   StatsProbeImmediate::initialize(name, params, comm);
}

} /* namespace PV */
