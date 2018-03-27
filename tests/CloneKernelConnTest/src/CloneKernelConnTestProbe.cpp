/*
 * CloneKernelConnTestProbe.cpp
 *
 *  Created on: Feb 24, 2012
 *      Author: peteschultz
 */

#include "CloneKernelConnTestProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <string.h>
#include <utils/PVLog.hpp>

namespace PV {

CloneKernelConnTestProbe::CloneKernelConnTestProbe(const char *name, HyPerCol *hc) : StatsProbe() {
   initialize_base();
   initialize(name, hc);
}

int CloneKernelConnTestProbe::initialize_base() { return PV_SUCCESS; }

int CloneKernelConnTestProbe::initialize(const char *name, HyPerCol *hc) {
   return StatsProbe::initialize(name, hc);
}

Response::Status CloneKernelConnTestProbe::outputState(double timed) {
   auto status = StatsProbe::outputState(timed);
   if (status != Response::SUCCESS) {
      return status;
   }
   int const rank    = parent->getCommunicator()->commRank();
   int const rcvProc = 0;
   if (rank != rcvProc) {
      return status;
   }
   for (int b = 0; b < parent->getNBatch(); b++) {
      if (timed > 2.0) {
         FatalIf(!(fabsf(fMin[b]) < 1e-6f), "Test failed.\n");
         FatalIf(!(fabsf(fMax[b]) < 1e-6f), "Test failed.\n");
         FatalIf(!(fabsf(avg[b]) < 1e-6f), "Test failed.\n");
      }
   }

   return status;
}

} /* namespace PV */
