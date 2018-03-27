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
Response::Status GPUSystemTestProbe::outputState(double timed) {
   auto status = StatsProbe::outputState(timed);
   if (status != Response::SUCCESS) {
      return status;
   }
   const PVLayerLoc *loc = getTargetLayer()->getLayerLoc();
   int numExtNeurons     = getTargetLayer()->getNumExtendedAllBatches();
   const float *A        = getTargetLayer()->getLayerData();
   float sumsq           = 0;
   for (int i = 0; i < numExtNeurons; i++) {
      float tol = 5.0e-4;
      FatalIf(
            fabsf(A[i]) >= 5e-4f,
            "%s neuron index %d has value %f at time %f; tolerance is %f.\n",
            getTargetLayer()->getDescription_c(),
            i,
            (double)A[i],
            timed,
            (double)tol);
   }
   for (int b = 0; b < loc->nbatch; b++) {
      // For max std of 5.0fe-5
      float tol = 5e-5;
      FatalIf(
            sigma[b] > tol,
            "%s batch element %d has standard deviation %f at time %f; tolerance is %f.\n",
            getTargetLayer()->getDescription_c(),
            b,
            (double)sigma[b],
            timed,
            (double)tol);
   }

   return status;
}

} // end namespace PV
