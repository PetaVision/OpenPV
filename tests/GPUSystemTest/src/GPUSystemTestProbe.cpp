/*
 * GPUSystemTestProbe.cpp
 * Author: slundquist
 */

#include "GPUSystemTestProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <string.h>
#include <utils/PVLog.hpp>

namespace PV {
GPUSystemTestProbe::GPUSystemTestProbe(const char *name, HyPerCol *hc) : StatsProbe() {
   initialize_base();
   initialize(name, hc);
}

int GPUSystemTestProbe::initialize_base() { return PV_SUCCESS; }

int GPUSystemTestProbe::initialize(const char *name, HyPerCol *hc) {
   return StatsProbe::initialize(name, hc);
}

void GPUSystemTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) { requireType(BufActivity); }

// 2 tests: max difference can be 5e-4, max std is 5e-5
int GPUSystemTestProbe::outputState(double timed) {
   int status            = StatsProbe::outputState(timed);
   const PVLayerLoc *loc = getTargetLayer()->getLayerLoc();
   int numExtNeurons     = getTargetLayer()->getNumExtendedAllBatches();
   const float *A        = getTargetLayer()->getLayerData();
   float sumsq           = 0;
   for (int i = 0; i < numExtNeurons; i++) {
      FatalIf(!(fabsf(A[i]) < 5e-4f), "Test failed.\n");
   }
   for (int b = 0; b < loc->nbatch; b++) {
      // For max std of 5e-5
      FatalIf(!(sigma[b] <= 5e-5f), "Test failed.\n");
   }

   return status;
}

} // end namespace PV
