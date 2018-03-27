/*
 * MPITestProbe.cpp
 *
 *  Created on:
 *      Author: garkenyon
 */

#include "MPITestProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <string.h>
#include <utils/PVLog.hpp>

namespace PV {

MPITestProbe::MPITestProbe(const char *name, HyPerCol *hc) : StatsProbe() { initialize(name, hc); }

int MPITestProbe::initialize_base() { return PV_SUCCESS; }

int MPITestProbe::initialize(const char *name, HyPerCol *hc) {
   return StatsProbe::initialize(name, hc);
}

Response::Status MPITestProbe::outputState(double timed) {
   auto status = StatsProbe::outputState(timed);
   if (status != Response::SUCCESS) {
      return status;
   }
   if (mOutputStreams.empty()) {
      return status;
   }
   float tol = 1e-4f;

   // if many to one connection, each neuron should receive its global x/y/f position
   // if one to many connection, the position of the nearest sending cell is received
   // assume sending layer has scale factor == 1
   int xScaleLog2 = getTargetLayer()->getCLayer()->xScale;

   // determine min/max position of receiving layer
   const PVLayerLoc *loc = getTargetLayer()->getLayerLoc();
   int nf                = loc->nf;
   int nxGlobal          = loc->nxGlobal;
   int nyGlobal          = loc->nyGlobal;
   float min_global_xpos = xPosGlobal(0, xScaleLog2, nxGlobal, nyGlobal, nf);
   int kGlobal           = nf * nxGlobal * nyGlobal - 1;
   float max_global_xpos = xPosGlobal(kGlobal, xScaleLog2, nxGlobal, nyGlobal, nf);

   if (xScaleLog2 < 0) {
      float xpos_shift = 0.5f - min_global_xpos;
      min_global_xpos  = 0.5f;
      max_global_xpos -= xpos_shift;
   }
   float ave_global_xpos = (min_global_xpos + max_global_xpos) / 2.0f;

   for (int b = 0; b < (int)mOutputStreams.size(); b++) {
      if (timed > 3.0) {
         output(b) << std::endl;
         output(b).printf(
               "%s min_global_xpos==%f ave_global_xpos==%f max_global_xpos==%f",
               getMessage(),
               (double)min_global_xpos,
               (double)ave_global_xpos,
               (double)max_global_xpos);
         FatalIf(
               (fMin[b] / min_global_xpos <= (1 - tol)) or (fMin[b] / min_global_xpos >= (1 + tol)),
               "%s fMin differs from %f.\n",
               getDescription_c(),
               (double)min_global_xpos);
         FatalIf(
               (fMax[b] / max_global_xpos <= (1 - tol)) or (fMax[b] / max_global_xpos >= (1 + tol)),
               "%s fMax differs from %f.\n",
               getDescription_c(),
               (double)max_global_xpos);
         FatalIf(
               (avg[b] / ave_global_xpos <= (1 - tol)) or (avg[b] / ave_global_xpos >= (1 + tol)),
               "%s average differs from %f.\n",
               getDescription_c(),
               (double)ave_global_xpos);
      }
   }

   return status;
}

} // end namespace PV
