/*
 * CloneHyPerConnTestProbe.cpp
 *
 *  Created on: Feb 24, 2012
 *      Author: peteschultz
 */

#include "CloneHyPerConnTestProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <string.h>
#include <utils/PVLog.hpp>

namespace PV {

CloneHyPerConnTestProbe::CloneHyPerConnTestProbe(const char *probeName, HyPerCol *hc)
      : StatsProbe() {
   initCloneHyPerConnTestProbe_base();
   initCloneHyPerConnTestProbe(probeName, hc);
}

int CloneHyPerConnTestProbe::initCloneHyPerConnTestProbe_base() { return PV_SUCCESS; }

int CloneHyPerConnTestProbe::initCloneHyPerConnTestProbe(const char *probeName, HyPerCol *hc) {
   return initStatsProbe(probeName, hc);
}

int CloneHyPerConnTestProbe::outputState(double timed) {
   int status           = StatsProbe::outputState(timed);
   Communicator *icComm = getTargetLayer()->getParent()->getCommunicator();
   const int rcvProc    = 0;
   if (icComm->commRank() != rcvProc) {
      return 0;
   }
   for (int b = 0; b < getParent()->getNBatch(); b++) {
      if (timed > 2.0) {
         pvErrorIf(!(fabsf(fMin[b]) < 1e-6f), "Test failed.\n");
         pvErrorIf(!(fabsf(fMax[b]) < 1e-6f), "Test failed.\n");
         pvErrorIf(!(fabsf(avg[b]) < 1e-6f), "Test failed.\n");
      }
   }

   return status;
}

} /* namespace PV */
