/*
 * receiveFromPostProbe.cpp
 * Author: slundquist
 */

#include "ReceiveFromPostProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <string.h>
#include <utils/PVLog.hpp>

namespace PV {
ReceiveFromPostProbe::ReceiveFromPostProbe(const char *name, HyPerCol *hc) : StatsProbe() {
   initialize_base();
   initialize(name, hc);
}

int ReceiveFromPostProbe::initialize_base() {
   tolerance = (float)1e-3f;
   return PV_SUCCESS;
}

int ReceiveFromPostProbe::initialize(const char *name, HyPerCol *hc) {
   return StatsProbe::initialize(name, hc);
}

int ReceiveFromPostProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = StatsProbe::ioParamsFillGroup(ioFlag);
   ioParam_tolerance(ioFlag);
   return status;
}

void ReceiveFromPostProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) { requireType(BufActivity); }

void ReceiveFromPostProbe::ioParam_tolerance(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, getName(), "tolerance", &tolerance, tolerance);
}

Response::Status ReceiveFromPostProbe::outputState(double timed) {
   auto status = StatsProbe::outputState(timed);
   if (status != Response::SUCCESS) {
      return status;
   }
   const PVLayerLoc *loc = getTargetLayer()->getLayerLoc();
   int numExtNeurons     = getTargetLayer()->getNumExtended();
   const float *A        = getTargetLayer()->getLayerData();
   bool failed           = false;
   for (int i = 0; i < numExtNeurons; i++) {
      if (fabsf(A[i]) != 0) {
         int xpos =
               kxPos(i,
                     loc->nx + loc->halo.lt + loc->halo.rt,
                     loc->ny + loc->halo.dn + loc->halo.up,
                     loc->nf);
         int ypos =
               kyPos(i,
                     loc->nx + loc->halo.lt + loc->halo.rt,
                     loc->ny + loc->halo.dn + loc->halo.up,
                     loc->nf);
         int fpos = featureIndex(
               i,
               loc->nx + loc->halo.lt + loc->halo.rt,
               loc->ny + loc->halo.dn + loc->halo.up,
               loc->nf);
         // InfoLog() << "[" << xpos << "," << ypos << "," << fpos << "] = " << std::fixed << A[i]
         // <<
         // "\n";
      }
      // For roundoff errors
      if (fabsf(A[i]) >= tolerance) {
         ErrorLog().printf(
               "%s %s activity outside of tolerance %f: extended index %d has activity %f\n",
               getMessage(),
               getTargetLayer()->getDescription_c(),
               (double)tolerance,
               i,
               (double)A[i]);
         failed = true;
      }
      if (failed) {
         exit(EXIT_FAILURE);
      }
   }
   return status;
}

} // end namespace PV
