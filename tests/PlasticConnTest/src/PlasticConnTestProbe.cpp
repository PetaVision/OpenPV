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

PlasticConnTestProbe::PlasticConnTestProbe(const char *probename, HyPerCol *hc) {
   initialize(probename, hc);
}

int PlasticConnTestProbe::initialize(const char *probename, HyPerCol *hc) {
   errorPresent = false;
   return KernelProbe::initialize(probename, hc);
}

Response::Status PlasticConnTestProbe::outputState(double timed) {
   if (mOutputStreams.empty()) {
      return Response::NO_ACTION;
   }
   output(0).printf("    Time %f, %s:\n", timed, getTargetConn()->getDescription_c());

   const int nxp       = getPatchSize()->getPatchSizeX();
   const int nyp       = getPatchSize()->getPatchSizeY();
   const int nfp       = getPatchSize()->getPatchSizeF();
   const int patchSize = nxp * nyp * nfp;
   float const *w      = getWeightData() + getKernelIndex() * patchSize;

   if (getOutputPlasticIncr() && getDeltaWeightData() == nullptr) {
      Fatal().printf(
            "%s: %s has dKernelData(%d,%d) set to null.\n",
            getDescription_c(),
            getTargetConn()->getDescription_c(),
            getKernelIndex(),
            getArbor());
   }
   float const *dw = getDeltaWeightData() + getKernelIndex() * patchSize;

   int status = PV_SUCCESS;
   for (int k = 0; k < patchSize; k++) {
      int x  = kxPos(k, nxp, nyp, nfp);
      int wx = (nxp - 1) / 2 - x; // assumes connection is one-to-one
      if (getOutputWeights()) {
         float wCorrect  = timed * wx;
         float wObserved = w[k];
         if (k == 0) {
            double q = fabs(((double)(wObserved - wCorrect)) / timed);
            printf(
                  "fabs(%f/%f) = %f\n",
                  ((double)(wObserved - wCorrect)),
                  timed,
                  fabs(((double)(wObserved - wCorrect)) / timed));
         }
         if (fabs(((double)(wObserved - wCorrect)) / timed) > 1e-4) {
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
         }
         // status = PV_FAILURE;
      }
      if (timed > 0 && getOutputPlasticIncr()) {
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
         }
         // status = PV_FAILURE;
      }
   }
   FatalIf(status != PV_SUCCESS, "%s failed at t=%f.\n", getDescription_c(), timed);
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
   Communicator *icComm = parent->getCommunicator();
   if (!mOutputStreams.empty()) {
      if (!errorPresent) {
         output(0).printf("No errors detected\n");
      }
   }
}

} // end of namespace PV block
