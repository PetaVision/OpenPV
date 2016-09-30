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

/**
 * @filename
 * @type
 * @msg
 */
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

/**
 * @timef
 */
int MomentumConnTestProbe::outputState(double timed) {
   HyPerConn *c         = getTargetHyPerConn();
   Communicator *icComm = c->getParent()->getCommunicator();
   const int rcvProc    = 0;
   if (icComm->commRank() != rcvProc) {
      return PV_SUCCESS;
   }
   pvErrorIf(!(getTargetConn() != NULL), "Test failed.\n");
   outputStream->printf("    Time %f, %s:\n", timed, getTargetConn()->getDescription_c());
   const pvwdata_t *w = c->get_wDataHead(getArbor(), getKernelIndex());
   const pvdata_t *dw = c->get_dwDataHead(getArbor(), getKernelIndex());
   if (getOutputPlasticIncr() && dw == NULL) {
      pvError().printf(
            "%s: %s has dKernelData(%d,%d) set to null.\n",
            getDescription_c(),
            getTargetConn()->getDescription_c(),
            getKernelIndex(),
            getArbor());
   }
   int nxp    = c->xPatchSize();
   int nyp    = c->yPatchSize();
   int nfp    = c->fPatchSize();
   int status = PV_SUCCESS;
   for (int k = 0; k < nxp * nyp * nfp; k++) {
      pvdata_t wObserved = w[k];
      // Pulse happens at time 3
      pvdata_t wCorrect;

      if (timed < 3) {
         wCorrect = 0;
      } else {
         if (isViscosity) {
            wCorrect = 1;
            for (int i = 0; i < (timed - 3); i++) {
               wCorrect += expf(-(2 * (i + 1)));
            }
         } else {
            wCorrect = 2 - powf(2, -(timed - 3));
         }
      }

      if (fabs(((double)(wObserved - wCorrect)) / timed) > 1e-4) {
         int y = kyPos(k, nxp, nyp, nfp);
         int f = featureIndex(k, nxp, nyp, nfp);
         outputStream->printf(
               "        w = %f, should be %f\n", (double)wObserved, (double)wCorrect);
         exit(-1);
      }
   }

   return PV_SUCCESS;
}

} // end of namespace PV block
