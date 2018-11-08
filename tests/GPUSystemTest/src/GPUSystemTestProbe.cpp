/*
 * GPUSystemTestProbe.cpp
 * Author: slundquist
 */

#include "GPUSystemTestProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <string.h>
#include <utils/PVLog.hpp>

namespace PV {
GPUSystemTestProbe::GPUSystemTestProbe(const char *name, PVParams *params, Communicator *comm) {
   initialize_base();
   initialize(name, params, comm);
}

int GPUSystemTestProbe::initialize_base() { return PV_SUCCESS; }

void GPUSystemTestProbe::initialize(const char *name, PVParams *params, Communicator *comm) {
   StatsProbe::initialize(name, params, comm);
}

void GPUSystemTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) { requireType(BufActivity); }

// 2 tests: max difference can be 5e-4, max std is 5e-5
Response::Status GPUSystemTestProbe::outputState(double simTime, double deltaTime) {
   auto status = RequireAllZeroActivityProbe::outputState(simTime, deltaTime);
   if (status != Response::SUCCESS) {
      return status;
   }
   const PVLayerLoc *loc = getTargetLayer()->getLayerLoc();
   int numExtNeurons     = getTargetLayer()->getNumExtendedAllBatches();
   const float *A        = getTargetLayer()->getLayerData();
   float sumsq           = 0;
   float tolSigma        = 5e-5;
   for (int b = 0; b < loc->nbatch; b++) {
      // For max std of 5.0fe-5
      if (sigma[b] > tolSigma) {
         if (!nonzeroFound) {
            nonzeroTime = simTime;
         }
         nonzeroFound = true;
         if (mCommunicator->commRank() == 0) {
            std::stringstream message("");
            message << getDescription_c() << ": Nonzero standard deviation " << simTime
                    << " at time " << nonzeroTime << "; tolerance is " << tolSigma << "\n";
            if (immediateExitOnFailure) {
               Fatal() << message.str();
            }
            else {
               WarnLog() << message.str();
            }
         }
      }
   }

   return status;
}

} // end namespace PV
