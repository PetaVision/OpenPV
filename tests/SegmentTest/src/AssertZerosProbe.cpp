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
AssertZerosProbe::AssertZerosProbe(const char *name, PVParams *params, Communicator const *comm)
      : StatsProbe() {
   initialize_base();
   initialize(name, params, comm);
}

int AssertZerosProbe::initialize_base() { return PV_SUCCESS; }

void AssertZerosProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   StatsProbe::initialize(name, params, comm);
}

void AssertZerosProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) { requireType(BufActivity); }

// 2 tests: max difference can be 5e-4, max std is 5e-5
Response::Status AssertZerosProbe::outputState(double simTime, double deltaTime) {
   auto status = StatsProbe::outputState(simTime, deltaTime);
   if (status != Response::SUCCESS) {
      return status;
   }
   auto *targetLayerInputBuffer = getTargetLayer()->getComponentByType<LayerInputBuffer>();
   auto *targetPublisher        = getTargetLayer()->getComponentByType<BasePublisherComponent>();
   const PVLayerLoc *loc        = getTargetLayer()->getLayerLoc();
   int numExtNeurons            = targetPublisher->getNumExtended() * loc->nbatch;
   int numResNeurons            = targetLayerInputBuffer->getBufferSizeAcrossBatch();
   const float *A               = targetPublisher->getLayerData();
   const float *GSyn_E          = targetLayerInputBuffer->getChannelData(CHANNEL_EXC);
   const float *GSyn_I          = targetLayerInputBuffer->getChannelData(CHANNEL_INH);

   // getOutputStream().precision(15);
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
      FatalIf(sigma[b] > 5e-5f, "Test failed.\n");
   }

   return Response::SUCCESS;
}

} // end namespace PV
