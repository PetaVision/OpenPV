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

InitWeightTestProbe::InitWeightTestProbe(const char *name, PVParams *params, Communicator *comm)
      : StatsProbe() {
   initialize(name, params, comm);
}

int InitWeightTestProbe::initialize_base() { return PV_SUCCESS; }

void InitWeightTestProbe::initialize(const char *name, PVParams *params, Communicator *comm) {
   StatsProbe::initialize(name, params, comm);
}

void InitWeightTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) { requireType(BufActivity); }

Response::Status InitWeightTestProbe::outputState(double simTime, double deltaTime) {
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
         FatalIf(std::abs(fMin[b]) >= 0.001f, "Test failed.\n");
         FatalIf(std::abs(fMax[b]) >= 0.001f, "Test failed.\n");
         FatalIf(std::abs(avg[b]) >= 0.001f, "Test failed.\n");
      }
   }

   return status;
}

} /* namespace PV */
