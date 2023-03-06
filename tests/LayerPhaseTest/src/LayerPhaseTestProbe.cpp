/*
 * LayerPhaseTestProbe.cpp
 *
 *  Created on: January 27, 2013
 *      Author: garkenyon
 */

#include "LayerPhaseTestProbe.hpp"
#include "include/pv_common.h"
#include <probes/ProbeData.hpp>
#include <probes/StatsProbeTypes.hpp>
#include <probes/VMembraneBufferStatsProbeLocal.hpp>
#include <utils/PVLog.hpp>

#include <cstdlib>
#include <memory>

namespace PV {

LayerPhaseTestProbe::LayerPhaseTestProbe(
      const char *name,
      PVParams *params,
      Communicator const *comm)
      : StatsProbeImmediate() {
   initialize(name, params, comm);
}

void LayerPhaseTestProbe::checkStats() {
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
   if (simTime >= mEquilibriumTime) {
      for (int b = 0; b < nbatch; b++) {
         LayerStats const &statsElem = stats.getValue(b);
         float const tol             = 1e-6f;
         if (std::abs(statsElem.mMin - mEquilibriumValue) >= tol) {
            ErrorLog().printf(
                  "%s, t=%f, b=%d, minimum %f differs from correct value %f\n",
                  getDescription_c(),
                  simTime,
                  b,
                  (double)statsElem.mMin,
                  (double)mEquilibriumValue);
            status = PV_FAILURE;
         }
         if (std::abs(statsElem.mMax - mEquilibriumValue) >= tol) {
            ErrorLog().printf(
                  "%s, t=%f, b=%d, maximum %f differs from correct value %f\n",
                  getDescription_c(),
                  simTime,
                  b,
                  (double)statsElem.mMax,
                  (double)mEquilibriumValue);
            status = PV_FAILURE;
         }
         if (std::abs(statsElem.average() - (double)mEquilibriumValue) >= (double)tol) {
            ErrorLog().printf(
                  "%s, t=%f, b=%d, average value %f differs from correct value %f\n",
                  getDescription_c(),
                  simTime,
                  b,
                  statsElem.average(),
                  (double)mEquilibriumValue);
            status = PV_FAILURE;
         }
      }
   }
   FatalIf(status != PV_SUCCESS, "%s failed.\n", getDescription_c());
}

void LayerPhaseTestProbe::createProbeLocal(char const *name, PVParams *params) {
   mProbeLocal = std::make_shared<VMembraneBufferStatsProbeLocal>(name, params);
}

void LayerPhaseTestProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   StatsProbeImmediate::initialize(name, params, comm);
}

int LayerPhaseTestProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = StatsProbeImmediate::ioParamsFillGroup(ioFlag);
   ioParam_equilibriumValue(ioFlag);
   ioParam_equilibriumTime(ioFlag);
   return status;
}

void LayerPhaseTestProbe::ioParam_equilibriumValue(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag, getName(), "equilibriumValue", &mEquilibriumValue, mEquilibriumValue, true);
}

void LayerPhaseTestProbe::ioParam_equilibriumTime(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag, getName(), "equilibriumTime", &mEquilibriumTime, mEquilibriumTime, true);
}

} /* namespace PV */
