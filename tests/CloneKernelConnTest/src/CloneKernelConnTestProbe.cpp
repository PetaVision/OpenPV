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

CloneKernelConnTestProbe::CloneKernelConnTestProbe(const char *probeName, HyPerCol *hc)
      : StatsProbe() {
   initCloneKernelConnTestProbe_base();
   initCloneKernelConnTestProbe(probeName, hc);
}

int CloneKernelConnTestProbe::initCloneKernelConnTestProbe_base() { return PV_SUCCESS; }

int CloneKernelConnTestProbe::initCloneKernelConnTestProbe(const char *probeName, HyPerCol *hc) {
   return initStatsProbe(probeName, hc);
}

int CloneKernelConnTestProbe::outputState(double timed) {
   int status           = StatsProbe::outputState(timed);
   Communicator *icComm = getTargetLayer()->getParent()->getCommunicator();
   const int rcvProc    = 0;
   if (icComm->commRank() != rcvProc) {
      return 0;
   }

   for (int b = 0; b < getParent()->getNBatch(); b++) {
      if (timed > 2.0) {
         FatalIf(!(fabsf(fMin[b]) < 1e-6f), "Test failed.\n");
         FatalIf(!(fabsf(fMax[b]) < 1e-6f), "Test failed.\n");
         FatalIf(!(fabsf(avg[b]) < 1e-6f), "Test failed.\n");
      }
   }

   return status;
}

} /* namespace PV */
