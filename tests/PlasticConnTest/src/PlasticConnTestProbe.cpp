/*
 * PlasticConnTestProbe.cpp
 *
 *  Created on:
 *      Author: garkenyon
 */

#include "PlasticConnTestProbe.hpp"
#include <string.h>
#include <utils/PVLog.hpp>

namespace PV {

PlasticConnTestProbe::PlasticConnTestProbe(
      const char *probename,
      PVParams *params,
      Communicator const *comm) {
   initialize(probename, params, comm);
}

void PlasticConnTestProbe::initialize(
      const char *probename,
      PVParams *params,
      Communicator const *comm) {
   errorPresent = false;
   KernelProbe::initialize(probename, params, comm);
}

Response::Status PlasticConnTestProbe::outputState(double simTime, double deltaTime) {
   if (mOutputStreams.empty()) {
      return Response::NO_ACTION;
   }
   output(0).printf("    Time %f, %s:\n", simTime, getTargetConn()->getDescription_c());

   const int nxp       = getPatchSize()->getPatchSizeX();
   const int nyp       = getPatchSize()->getPatchSizeY();
   const int nfp       = getPatchSize()->getPatchSizeF();
   const int patchSize = nxp * nyp * nfp;
   float const *w      = getWeightData() + getKernelIndex() * patchSize;

   FatalIf(
         getOutputPlasticIncr() and getDeltaWeightData() == nullptr,
         "%s: %s has dKernelData(%d,%d) set to null.\n",
         getDescription_c(),
         getTargetConn()->getDescription_c(),
         getKernelIndex(),
         getArbor());
   float const *dw = getDeltaWeightData() + getKernelIndex() * patchSize;

   int status = PV_SUCCESS;
   for (int k = 0; k < patchSize; k++) {
      int x  = kxPos(k, nxp, nyp, nfp);
      int wx = (nxp - 1) / 2 - x; // assumes connection is one-to-one
      if (getOutputWeights()) {
         float wCorrect  = simTime * wx;
         float wObserved = w[k];
         if (fabs(((double)(wObserved - wCorrect)) / simTime) > 1e-4) {
            int y = kyPos(k, nxp, nyp, nfp);
            int f = featureIndex(k, nxp, nyp, nfp);
            output(0).printf(
                  "        index %d (x=%d, y=%d, f=%d: w = %f, should be %f\n",
                  k,
                  x,
                  y,
                  f,
                  (double)wObserved,
                  (double)wCorrect);
            status = PV_FAILURE;
         }
      }
      if (simTime > 0 && getOutputPlasticIncr()) {
         float dwCorrect  = wx;
         float dwObserved = dw[k];
         if (dwObserved != dwCorrect) {
            int y = kyPos(k, nxp, nyp, nfp);
            int f = featureIndex(k, nxp, nyp, nfp);
            output(0).printf(
                  "        index %d (x=%d, y=%d, f=%d: dw = %f, should be %f\n",
                  k,
                  x,
                  y,
                  f,
                  (double)dwObserved,
                  (double)dwCorrect);
            status = PV_FAILURE;
         }
      }
   }
   FatalIf(status != PV_SUCCESS, "%s failed at t=%f.\n", getDescription_c(), simTime);
   if (status == PV_SUCCESS) {
      if (getOutputWeights()) {
         output(0).printf("        All weights are correct.\n");
      }
      if (getOutputPlasticIncr()) {
         output(0).printf("        All plastic increments are correct.\n");
      }
   }
   if (getOutputPatchIndices()) {
      patchIndices();
   }

   return Response::SUCCESS;
}

PlasticConnTestProbe::~PlasticConnTestProbe() {
   if (!mOutputStreams.empty()) {
      if (!errorPresent) {
         output(0).printf("No errors detected\n");
      }
   }
}

} // end of namespace PV block
