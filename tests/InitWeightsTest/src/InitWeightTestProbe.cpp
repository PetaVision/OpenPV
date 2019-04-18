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

Response::Status InitWeightTestProbe::outputState(double timed) {
   auto status = StatsProbe::outputState(timed);
   if (status != Response::SUCCESS) {
      return status;
   }
   Communicator *icComm = parent->getCommunicator();
   const int rcvProc    = 0;
   if (icComm->commRank() != rcvProc) {
      return status;
   }
   for (int b = 0; b < parent->getNBatch(); b++) {
      if (timed > 2.0) {
         FatalIf(std::abs(fMin[b]) >= 0.001f, "Test failed.\n");
         FatalIf(std::abs(fMax[b]) >= 0.001f, "Test failed.\n");
         FatalIf(std::abs(avg[b]) >= 0.001f, "Test failed.\n");
      }
   }

   return status;
}

} /* namespace PV */
