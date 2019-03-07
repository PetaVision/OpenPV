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

MomentumConnTestProbe::MomentumConnTestProbe(
      const char *probename,
      PVParams *params,
      Communicator const *comm) {
   initialize(probename, params, comm);
}

void MomentumConnTestProbe::initialize(
      const char *probename,
      PVParams *params,
      Communicator const *comm) {
   KernelProbe::initialize(probename, params, comm);
}

int MomentumConnTestProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   KernelProbe::ioParamsFillGroup(ioFlag);
   ioParam_isViscosity(ioFlag);
   return PV_SUCCESS;
}

void MomentumConnTestProbe::ioParam_isViscosity(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "isViscosity", &isViscosity, 0 /*default value*/);
}

Response::Status MomentumConnTestProbe::outputState(double simTime, double deltaTime) {
   if (mOutputStreams.empty()) {
      return Response::NO_ACTION;
   }
   output(0).printf("    Time %f, %s:\n", simTime, getTargetConn()->getDescription_c());

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
      float wObserved = w[k];
      // Pulse happens at time 3
      float wCorrect;

      if (simTime < 3) {
         wCorrect = 0;
      }
      else {
         wCorrect = 1.0f - std::pow(2.0f, -((float)simTime - 2.0f));
      }

      if (fabs(((double)(wObserved - wCorrect)) / simTime) > 1e-4) {
         int y = kyPos(k, nxp, nyp, nfp);
         int f = featureIndex(k, nxp, nyp, nfp);
         output(0).printf("        w = %f, should be %f\n", (double)wObserved, (double)wCorrect);
         exit(EXIT_FAILURE);
      }
   }

   return Response::SUCCESS;
}

} // end of namespace PV block
