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

KernelTestProbe::KernelTestProbe(const char *name, HyPerCol *hc) : StatsProbe() {
   initialize(name, hc);
}

int KernelTestProbe::initialize_base() { return PV_SUCCESS; }

int KernelTestProbe::initialize(const char *name, HyPerCol *hc) {
   return StatsProbe::initialize(name, hc);
}

void KernelTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) { requireType(BufActivity); }

Response::Status KernelTestProbe::outputState(double timestamp) {
   auto status = StatsProbe::outputState(timestamp);
   if (status != Response::SUCCESS) {
      return status;
   }
   Communicator *icComm = parent->getCommunicator();
   const int rcvProc    = 0;
   if (icComm->commRank() != rcvProc) {
      return status;
   }
   for (int b = 0; b < parent->getNBatch(); b++) {
      if (timestamp > 2.0) {
         FatalIf((fMin[b] <= 0.99f) or (fMin[b] >= 1.010f), "Test failed.\n");
         FatalIf((fMax[b] <= 0.99f) or (fMax[b] >= 1.010f), "Test failed.\n");
         FatalIf((avg[b] <= 0.99f) or (avg[b] >= 1.010f), "Test failed.\n");
      }
   }

   return status;
}

} /* namespace PV */
