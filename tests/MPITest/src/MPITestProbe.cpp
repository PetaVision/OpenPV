/*
 * MPITestProbe.cpp
 *
 *  Created on:
 *      Author: garkenyon
 */

#include "MPITestProbe.hpp"
#include "MPITestProbeOutputter.hpp"
#include <columns/Communicator.hpp>
#include <components/LayerGeometry.hpp>
#include <include/PVLayerLoc.h>
#include <io/PVParams.hpp>
#include <probes/ActivityBufferStatsProbeLocal.hpp>
#include <probes/ProbeData.hpp>
#include <probes/StatsProbeImmediate.hpp>
#include <probes/StatsProbeTypes.hpp>
#include <utils/conversions.hpp>

#include <memory>

namespace PV {

MPITestProbe::MPITestProbe(const char *name, PVParams *params, Communicator const *comm)
      : StatsProbeImmediate() {
   initialize(name, params, comm);
}

void MPITestProbe::checkStats() {
   // if many to one connection, each neuron should receive its global x/y/f position
   // if one to many connection, the position of the nearest sending cell is received
   // assume sending layer has scale factor == 1
   auto *layerGeometry = getTargetLayer()->getComponentByType<LayerGeometry>();
   int xScaleLog2      = layerGeometry->getXScale();

   // determine min/max position of receiving layer
   PVLayerLoc const *loc = getTargetLayer()->getLayerLoc();
   int nf                = loc->nf;
   int nxGlobal          = loc->nxGlobal;
   int nyGlobal          = loc->nyGlobal;
   float min_global_xpos = xPosGlobal(0, xScaleLog2, nxGlobal, nyGlobal, nf);
   int kGlobal           = nf * nxGlobal * nyGlobal - 1;
   float max_global_xpos = xPosGlobal(kGlobal, xScaleLog2, nxGlobal, nyGlobal, nf);

   if (xScaleLog2 < 0) {
      float xpos_shift = 0.5f - min_global_xpos;
      min_global_xpos  = 0.5f;
      max_global_xpos -= xpos_shift;
   }
   double ave_global_xpos = (double)(min_global_xpos + max_global_xpos) / 2.0;

   auto const &storedValues           = mProbeAggregator->getStoredValues();
   auto numTimestamps                 = storedValues.size();
   int lastTimestampIndex             = static_cast<int>(numTimestamps) - 1;
   ProbeData<LayerStats> const &stats = storedValues.getData(lastTimestampIndex);
   if (stats.getTimestamp() > 3.0) {
      auto *outputter = dynamic_cast<MPITestProbeOutputter *>(mProbeOutputter.get());
      outputter->printGlobalXPosStats(stats, min_global_xpos, max_global_xpos, ave_global_xpos);
      // MPITestProbeOutputter::printGlobalXPosStats will error out if values are bad.
   }
}

void MPITestProbe::createProbeLocal(char const *name, PVParams *params) {
   mProbeLocal = std::make_shared<ActivityBufferStatsProbeLocal>(name, params);
}

void MPITestProbe::createProbeOutputter(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   mProbeOutputter = std::make_shared<MPITestProbeOutputter>(name, params, comm);
}

void MPITestProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   StatsProbeImmediate::initialize(name, params, comm);
}

} // end namespace PV
