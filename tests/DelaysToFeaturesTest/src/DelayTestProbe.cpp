/*
 * DelayTestProbe.cpp
 *
 *  Created on: October 1, 2013
 *      Author: wchavez
 */

#include "DelayTestProbe.hpp"
#include <include/PVLayerLoc.hpp>
#include <include/pv_common.h>
#include <layers/HyPerLayer.hpp>
#include <probes/ProbeData.hpp>
#include <probes/StatsProbeTypes.hpp>
#include <utils/PVLog.hpp>

#include <cmath>

namespace PV {

DelayTestProbe::DelayTestProbe(const char *name, PVParams *params, Communicator const *comm)
      : StatsProbeImmediate() {
   initialize(name, params, comm);
}

DelayTestProbe::~DelayTestProbe() {}

void DelayTestProbe::checkStats() {
   Communicator const *icComm = mCommunicator;
   int const rootProc         = 0;
   if (icComm->commRank() != rootProc) {
      return;
   }
   auto const &storedValues           = mProbeAggregator->getStoredValues();
   auto numTimestamps                 = storedValues.size();
   int lastTimestampIndex             = static_cast<int>(numTimestamps) - 1;
   ProbeData<LayerStats> const &stats = storedValues.getData(lastTimestampIndex);
   double simTime                     = stats.getTimestamp();

   const PVLayerLoc *loc = getTargetLayer()->getLayerLoc();
   const int rows        = icComm->numCommRows();
   const int cols        = icComm->numCommColumns();

   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   FatalIf(
         loc->nbatch != static_cast<int>(stats.size()),
         "ProbeData<LayerStats> has size %d but the target layer has batch size %d\n",
         static_cast<int>(stats.size()),
         loc->nbatch);

   for (int b = 0; b < loc->nbatch; b++) {
      LayerStats const &statsElem = stats.getValue(b);
      double avgExpected;
      int nnzExpected;
      if (simTime == 0.0) {
         avgExpected = 0.0;
         nnzExpected = (int)std::nearbyint(simTime) * nx * rows * ny * cols;
      }
      else {
         avgExpected = ((simTime - 1.0) / (double)nf);
         nnzExpected = ((int)std::nearbyint(simTime) - 1) * nx * rows * ny * cols;
      }
      double average = statsElem.average();
      FatalIf(
            average != avgExpected,
            "t = %f: Average for batch element %d: expected %f, received %f\n",
            simTime,
            b,
            avgExpected,
            average);
      FatalIf(
            statsElem.mNumNonzero != nnzExpected,
            "t = %f: number of nonzero elements for batch element %d: expected %d, received %d\n",
            simTime,
            b,
            nnzExpected,
            statsElem.mNumNonzero);
   }
}

void DelayTestProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   StatsProbeImmediate::initialize(name, params, comm);
}

} /* namespace PV */
