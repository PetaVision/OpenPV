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

int CloneKernelConnTestProbe::outputState(double timed) {
   int status           = StatsProbe::outputState(timed);
   Communicator *icComm = parent->getCommunicator();
   const int rcvProc    = 0;
   if (icComm->commRank() != rcvProc) {
      return 0;
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
