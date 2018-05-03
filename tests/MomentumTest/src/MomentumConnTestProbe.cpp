/*
 * MomentumConnTestProbe.cpp
 *
 *  Created on:
 *      Author: slundquist
 */

#include "MomentumConnTestProbe.hpp"
#include <string.h>
#include <utils/PVLog.hpp>

namespace PV {

MomentumConnTestProbe::MomentumConnTestProbe(const char *probename, HyPerCol *hc) {
   initialize(probename, hc);
}

int MomentumConnTestProbe::initialize(const char *probename, HyPerCol *hc) {
   return KernelProbe::initialize(probename, hc);
}

int MomentumConnTestProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   KernelProbe::ioParamsFillGroup(ioFlag);
   ioParam_isViscosity(ioFlag);
   return PV_SUCCESS;
}

void MomentumConnTestProbe::ioParam_isViscosity(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "isViscosity", &isViscosity, 0 /*default value*/);
}

Response::Status MomentumConnTestProbe::outputState(double timed) {
   HyPerConn *c = getTargetHyPerConn();
   FatalIf(c == nullptr, "%s has targetConnection set to null.\n", getDescription_c());
   if (mOutputStreams.empty()) {
      return Response::NO_ACTION;
   }
   output(0).printf("    Time %f, %s:\n", timed, getTargetConn()->getDescription_c());
   const float *w  = c->getWeightsDataHead(getArbor(), getKernelIndex());
   const float *dw = c->getDeltaWeightsDataHead(getArbor(), getKernelIndex());
   if (getOutputPlasticIncr() && dw == NULL) {
      Fatal().printf(
            "%s: %s has dKernelData(%d,%d) set to null.\n",
            getDescription_c(),
            getTargetConn()->getDescription_c(),
            getKernelIndex(),
            getArbor());
   }
   int nxp    = c->getPatchSizeX();
   int nyp    = c->getPatchSizeY();
   int nfp    = c->getPatchSizeF();
   int status = PV_SUCCESS;
   for (int k = 0; k < nxp * nyp * nfp; k++) {
      float wObserved = w[k];
      // Pulse happens at time 3
      float wCorrect;

      if (timed < 3) {
         wCorrect = 0;
      }
      else {
         if (isViscosity) {
            wCorrect = 1;
            for (int i = 0; i < (timed - 3); i++) {
               wCorrect += expf(-(2 * (i + 1)));
            }
         }
         else {
            wCorrect = 2 - powf(2, -(timed - 3));
         }
      }

      if (fabs(((double)(wObserved - wCorrect)) / timed) > 1e-4) {
         int y = kyPos(k, nxp, nyp, nfp);
         int f = featureIndex(k, nxp, nyp, nfp);
         output(0).printf("        w = %f, should be %f\n", (double)wObserved, (double)wCorrect);
         exit(EXIT_FAILURE);
      }
   }

   return Response::SUCCESS;
}

} // end of namespace PV block
