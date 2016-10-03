/*
 * ArborTestForOnesProbe.cpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#include "ArborTestForOnesProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <string.h>
#include <utils/PVLog.hpp>

namespace PV {

ArborTestForOnesProbe::ArborTestForOnesProbe(const char *probeName, HyPerCol *hc) : StatsProbe() {
   initArborTestForOnesProbe_base();
   initArborTestForOnesProbe(probeName, hc);
}

ArborTestForOnesProbe::~ArborTestForOnesProbe() {}

int ArborTestForOnesProbe::initArborTestForOnesProbe_base() { return PV_SUCCESS; }

int ArborTestForOnesProbe::initArborTestForOnesProbe(const char *probeName, HyPerCol *hc) {
   return initStatsProbe(probeName, hc);
}

int ArborTestForOnesProbe::outputState(double timed) {
   int status           = StatsProbe::outputState(timed);
   Communicator *icComm = getTargetLayer()->getParent()->getCommunicator();
   const int rcvProc    = 0;
   if (icComm->commRank() != rcvProc) {
      return 0;
   }
   if (timed > 1.0) {
      for (int b = 0; b < getParent()->getNBatch(); b++) {
         pvErrorIf(!((fMin[b] > 0.99f) && (fMin[b] < 1.01f)), "Test failed.\n");
         pvErrorIf(!((fMax[b] > 0.99f) && (fMax[b] < 1.01f)), "Test failed.\n");
         pvErrorIf(!((avg[b] > 0.99f) && (avg[b] < 1.01f)), "Test failed.\n");
      }
   }

   return status;
}

} /* namespace PV */
