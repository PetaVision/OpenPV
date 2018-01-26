/*
 * AssertZerosProbe.cpp
 * Author: slundquist
 */

#include "AssertZerosProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <string.h>
#include <utils/PVLog.hpp>

namespace PV {
AssertZerosProbe::AssertZerosProbe(const char *name, HyPerCol *hc) : StatsProbe() {
   initialize_base();
   initialize(name, hc);
}

int AssertZerosProbe::initialize_base() { return PV_SUCCESS; }

int AssertZerosProbe::initialize(const char *name, HyPerCol *hc) {
   return StatsProbe::initialize(name, hc);
}

void AssertZerosProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) { requireType(BufActivity); }

// 2 tests: max difference can be 5e-4, max std is 5e-5
Response::Status AssertZerosProbe::outputState(double timed) {
   auto status = StatsProbe::outputState(timed);
   if (status != Response::SUCCESS) {
      return status;
   }
   const PVLayerLoc *loc = getTargetLayer()->getLayerLoc();
   int numExtNeurons     = getTargetLayer()->getNumExtendedAllBatches();
   int numResNeurons     = getTargetLayer()->getNumNeuronsAllBatches();
   const float *A        = getTargetLayer()->getLayerData();
   const float *GSyn_E   = getTargetLayer()->getChannel(CHANNEL_EXC);
   const float *GSyn_I   = getTargetLayer()->getChannel(CHANNEL_INH);

   // getOutputStream().precision(15);
   float sumsq = 0;
   for (int i = 0; i < numExtNeurons; i++) {
      FatalIf(!(fabsf(A[i]) < 5e-4f), "Test failed.\n");
   }

   if (timed > 0) {
      // Make sure gsyn_e and gsyn_i are not all 0's
      float sum_E = 0;
      float sum_I = 0;
      for (int i = 0; i < numResNeurons; i++) {
         sum_E += GSyn_E[i];
         sum_I += GSyn_I[i];
      }

      FatalIf(!(sum_E != 0), "Test failed.\n");
      FatalIf(!(sum_I != 0), "Test failed.\n");
   }

   for (int b = 0; b < loc->nbatch; b++) {
      // For max std of 5e-5
      FatalIf(!(sigma[b] <= 5e-5f), "Test failed.\n");
   }

   return Response::SUCCESS;
}

} // end namespace PV
