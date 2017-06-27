/*
 * InitWeightTestProbe.cpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#include "InitWeightTestProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <string.h>
#include <utils/PVLog.hpp>

namespace PV {

InitWeightTestProbe::InitWeightTestProbe(const char *name, HyPerCol *hc) : StatsProbe() {
   initialize(name, hc);
}

int InitWeightTestProbe::initialize_base() { return PV_SUCCESS; }

int InitWeightTestProbe::initialize(const char *name, HyPerCol *hc) {
   return StatsProbe::initialize(name, hc);
}

void InitWeightTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) { requireType(BufActivity); }

int InitWeightTestProbe::outputState(double timed) {
   int status           = StatsProbe::outputState(timed);
   Communicator *icComm = parent->getCommunicator();
   const int rcvProc    = 0;
   if (icComm->commRank() != rcvProc) {
      return 0;
   }
   for (int b = 0; b < parent->getNBatch(); b++) {
      if (timed > 2.0) {
         FatalIf(!((fMin[b] > -0.001f) && (fMin[b] < 0.001f)), "Test failed.\n");
         FatalIf(!((fMax[b] > -0.001f) && (fMax[b] < 0.001f)), "Test failed.\n");
         FatalIf(!((avg[b] > -0.001f) && (avg[b] < 0.001f)), "Test failed.\n");
      }
   }

   return status;
}

} /* namespace PV */
