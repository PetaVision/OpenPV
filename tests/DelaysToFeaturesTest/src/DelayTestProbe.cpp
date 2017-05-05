/*
 * DelayTestProbe.cpp
 *
 *  Created on: October 1, 2013
 *      Author: wchavez
 */

#include "DelayTestProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <string.h>
#include <utils/PVLog.hpp>

namespace PV {

DelayTestProbe::DelayTestProbe(const char *probeName, HyPerCol *hc) : StatsProbe() {
   initDelayTestProbe_base();
   initDelayTestProbe(probeName, hc);
}

DelayTestProbe::~DelayTestProbe() {}

int DelayTestProbe::initDelayTestProbe_base() { return PV_SUCCESS; }

int DelayTestProbe::initDelayTestProbe(const char *probeName, HyPerCol *hc) {
   return initStatsProbe(probeName, hc);
}

int DelayTestProbe::outputState(double timestamp) {
   int status           = StatsProbe::outputState(timestamp);
   Communicator *icComm = getTargetLayer()->getParent()->getCommunicator();
   const int rcvProc    = 0;
   if (icComm->commRank() != rcvProc) {
      return 0;
   }
   const PVLayerLoc *loc = getTargetLayer()->getLayerLoc();
   const int rows        = getTargetLayer()->getParent()->getCommunicator()->numCommRows();
   const int cols        = getTargetLayer()->getParent()->getCommunicator()->numCommColumns();

   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;

   for (int b = 0; b < loc->nbatch; b++) {
      float avgExpected;
      int nnzExpected;
      if (timestamp == 0) {
         avgExpected = 0.0f;
         nnzExpected = (int)std::nearbyint(timestamp) * nx * rows * ny * cols;
      }
      else {
         avgExpected = (float)((timestamp - 1.0) / nf);
         nnzExpected = ((int)std::nearbyint(timestamp) - 1) * nx * rows * ny * cols;
      }
      FatalIf(
            avg[b] != avgExpected,
            "t = %f: Average for batch element %d: expected %f, received %f\n",
            timestamp,
            b,
            (double)avgExpected,
            (double)avg[b]);
      FatalIf(
            nnz[b] != nnzExpected,
            "t = %f: number of nonzero elements for batch element %d: expected %d, received %d\n",
            timestamp,
            b,
            nnzExpected,
            nnz[b]);
   }
   return status;
}

} /* namespace PV */
