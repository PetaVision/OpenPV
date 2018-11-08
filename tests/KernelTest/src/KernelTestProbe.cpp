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

KernelTestProbe::KernelTestProbe(const char *name, PVParams *params, Communicator *comm)
      : StatsProbe() {
   initialize(name, params, comm);
}

int KernelTestProbe::initialize_base() { return PV_SUCCESS; }

void KernelTestProbe::initialize(const char *name, PVParams *params, Communicator *comm) {
   StatsProbe::initialize(name, params, comm);
}

void KernelTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) { requireType(BufActivity); }

Response::Status KernelTestProbe::outputState(double simTime, double deltaTime) {
   auto status = StatsProbe::outputState(simTime, deltaTime);
   if (status != Response::SUCCESS) {
      return status;
   }
   Communicator *icComm = mCommunicator;
   const int rcvProc    = 0;
   if (icComm->commRank() != rcvProc) {
      return status;
   }
   for (int b = 0; b < mLocalBatchWidth; b++) {
      if (simTime > 2.0) {
         FatalIf((fMin[b] <= 0.99f) or (fMin[b] >= 1.010f), "Test failed.\n");
         FatalIf((fMax[b] <= 0.99f) or (fMax[b] >= 1.010f), "Test failed.\n");
         FatalIf((avg[b] <= 0.99f) or (avg[b] >= 1.010f), "Test failed.\n");
      }
   }

   return status;
}

} /* namespace PV */
