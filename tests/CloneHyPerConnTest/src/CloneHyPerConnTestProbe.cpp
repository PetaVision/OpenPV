/*
 * CloneHyPerConnTestProbe.cpp
 *
 *  Created on: Feb 24, 2012
 *      Author: peteschultz
 */

#include "CloneHyPerConnTestProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <string.h>
#include <utils/PVLog.hpp>

namespace PV {

CloneHyPerConnTestProbe::CloneHyPerConnTestProbe(const char *name, HyPerCol *hc) : StatsProbe() {
   initialize_base();
   initialize(name, hc);
}

int CloneHyPerConnTestProbe::initialize_base() { return PV_SUCCESS; }

int CloneHyPerConnTestProbe::initialize(const char *name, HyPerCol *hc) {
   return StatsProbe::initialize(name, hc);
}

Response::Status CloneHyPerConnTestProbe::outputState(double simTime, double deltaTime) {
   auto status = StatsProbe::outputState(simTime, deltaTime);
   if (status != Response::SUCCESS) {
      return status;
   }
   int const rank    = parent->getCommunicator()->commRank();
   int const rcvProc = 0;
   if (rank != rcvProc) {
      return status;
   }
   int const nbatch = getTargetLayer()->getLayerLoc()->nbatch;
   for (int b = 0; b < nbatch; b++) {
      if (simTime > 2.0) {
         FatalIf(fabsf(fMin[b]) >= 1e-6f, "Test failed.\n");
         FatalIf(fabsf(fMax[b]) >= 1e-6f, "Test failed.\n");
         FatalIf(fabsf(avg[b]) >= 1e-6f, "Test failed.\n");
      }
   }

   return status;
}

} /* namespace PV */
