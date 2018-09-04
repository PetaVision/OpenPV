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

DelayTestProbe::DelayTestProbe(const char *name, HyPerCol *hc) : StatsProbe() {
   initialize_base();
   initialize(name, hc);
}

DelayTestProbe::~DelayTestProbe() {}

int DelayTestProbe::initialize_base() { return PV_SUCCESS; }

int DelayTestProbe::initialize(const char *name, HyPerCol *hc) {
   return StatsProbe::initialize(name, hc);
}

Response::Status DelayTestProbe::outputState(double simTime, double deltaTime) {
   auto status = StatsProbe::outputState(simTime, deltaTime);
   if (status != Response::SUCCESS) {
      return status;
   }
   Communicator *icComm = parent->getCommunicator();
   int const rcvProc    = 0;
   if (icComm->commRank() != rcvProc) {
      return status;
   }
   const PVLayerLoc *loc = getTargetLayer()->getLayerLoc();
   const int rows        = icComm->numCommRows();
   const int cols        = icComm->numCommColumns();

   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;

   for (int b = 0; b < loc->nbatch; b++) {
      float avgExpected;
      int nnzExpected;
      if (simTime == 0) {
         avgExpected = 0.0f;
         nnzExpected = (int)std::nearbyint(simTime) * nx * rows * ny * cols;
      }
      else {
         avgExpected = (float)((simTime - 1.0) / nf);
         nnzExpected = ((int)std::nearbyint(simTime) - 1) * nx * rows * ny * cols;
      }
      FatalIf(
            avg[b] != avgExpected,
            "t = %f: Average for batch element %d: expected %f, received %f\n",
            simTime,
            b,
            (double)avgExpected,
            (double)avg[b]);
      FatalIf(
            nnz[b] != nnzExpected,
            "t = %f: number of nonzero elements for batch element %d: expected %d, received %d\n",
            simTime,
            b,
            nnzExpected,
            nnz[b]);
   }
   return status;
}

} /* namespace PV */
