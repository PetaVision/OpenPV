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

/**
 * @filename
 * @type
 * @msg
 */
PlasticConnTestProbe::PlasticConnTestProbe(const char *probename, HyPerCol *hc) {
   initialize(probename, hc);
}

int PlasticConnTestProbe::initialize(const char *probename, HyPerCol *hc) {
   errorPresent = false;
   return KernelProbe::initialize(probename, hc);
}
/**
 * @timef
 */
int PlasticConnTestProbe::outputState(double timed) {
   HyPerConn *c = getTargetHyPerConn();
   FatalIf(c == nullptr, "%s has targetConnection set to null.\n");
   if (mOutputStreams.empty()) {
      return PV_SUCCESS;
   }
   output(0).printf("    Time %f, %s:\n", timed, c->getDescription_c());
   const float *w  = c->get_wDataHead(getArbor(), getKernelIndex());
   const float *dw = c->get_dwDataHead(getArbor(), getKernelIndex());
   if (getOutputPlasticIncr() && dw == NULL) {
      Fatal().printf(
            "%s: %s has dKernelData(%d,%d) set to null.\n",
            getDescription_c(),
            c->getDescription_c(),
            getKernelIndex(),
            getArbor());
   }
   int nxp    = c->xPatchSize();
   int nyp    = c->yPatchSize();
   int nfp    = c->fPatchSize();
   int status = PV_SUCCESS;
   for (int k = 0; k < nxp * nyp * nfp; k++) {
      int x  = kxPos(k, nxp, nyp, nfp);
      int wx = (nxp - 1) / 2 - x; // assumes connection is one-to-one
      if (getOutputWeights()) {
         float wCorrect  = timed * wx;
         float wObserved = w[k];
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
      }
      if (timed > 0 && getOutputPlasticIncr() && dw != NULL) {
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
      patchIndices(c);
   }

   return PV_SUCCESS;
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
