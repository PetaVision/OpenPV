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

CloneKernelConnTestProbe::CloneKernelConnTestProbe(
      const char *name,
      PVParams *params,
      Communicator *comm)
      : StatsProbe() {
   initialize_base();
   initialize(name, params, comm);
}

int CloneKernelConnTestProbe::initialize_base() { return PV_SUCCESS; }

void CloneKernelConnTestProbe::initialize(const char *name, PVParams *params, Communicator *comm) {
   StatsProbe::initialize(name, params, comm);
}

Response::Status CloneKernelConnTestProbe::outputState(double simTime, double deltaTime) {
   auto status = StatsProbe::outputState(simTime, deltaTime);
   if (status != Response::SUCCESS) {
      return status;
   }
   int const rank    = mCommunicator->commRank();
   int const rcvProc = 0;
   if (rank != rcvProc) {
      return status;
   }
   for (int b = 0; b < mLocalBatchWidth; b++) {
      if (simTime > 2.0) {
         FatalIf(!(fabsf(fMin[b]) < 1e-6f), "Test failed.\n");
         FatalIf(!(fabsf(fMax[b]) < 1e-6f), "Test failed.\n");
         FatalIf(!(fabsf(avg[b]) < 1e-6f), "Test failed.\n");
      }
   }

   return status;
}

} /* namespace PV */
