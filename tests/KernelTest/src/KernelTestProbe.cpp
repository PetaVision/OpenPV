/*
 * KernelTestProbe.cpp
 *
 *  Created on: Sep 1, 2011
 *      Author: gkenyon
 */

#include "KernelTestProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <string.h>
#include <utils/PVLog.hpp>

namespace PV {

KernelTestProbe::KernelTestProbe(const char *probeName, HyPerCol *hc) : StatsProbe() {
   initKernelTestProbe(probeName, hc);
}

int KernelTestProbe::initKernelTestProbe_base() { return PV_SUCCESS; }

int KernelTestProbe::initKernelTestProbe(const char *probeName, HyPerCol *hc) {
   return initStatsProbe(probeName, hc);
}

void KernelTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) { requireType(BufActivity); }

int KernelTestProbe::outputState(double timed) {
   int status           = StatsProbe::outputState(timed);
   Communicator *icComm = getTargetLayer()->getParent()->getCommunicator();
   const int rcvProc    = 0;
   if (icComm->commRank() != rcvProc) {
      return 0;
   }
   for (int b = 0; b < parent->getNBatch(); b++) {
      if (timed > 2.0) {
         FatalIf(!((fMin[b] > 0.99f) && (fMin[b] < 1.010f)), "Test failed.\n");
         FatalIf(!((fMax[b] > 0.99f) && (fMax[b] < 1.010f)), "Test failed.\n");
         FatalIf(!((avg[b] > 0.99f) && (avg[b] < 1.010f)), "Test failed.\n");
      }
   }

   return status;
}

} /* namespace PV */
