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
ReceiveFromPostProbe::ReceiveFromPostProbe(
      const char *name,
      PVParams *params,
      Communicator const *comm)
      : StatsProbe() {
   initialize_base();
   initialize(name, params, comm);
}

int ReceiveFromPostProbe::initialize_base() {
   tolerance = (float)1e-3f;
   return PV_SUCCESS;
}

void ReceiveFromPostProbe::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   StatsProbe::initialize(name, params, comm);
}

int ReceiveFromPostProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = StatsProbe::ioParamsFillGroup(ioFlag);
   ioParam_tolerance(ioFlag);
   return status;
}

void ReceiveFromPostProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) { requireType(BufActivity); }

void ReceiveFromPostProbe::ioParam_tolerance(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "tolerance", &tolerance, tolerance);
}

Response::Status ReceiveFromPostProbe::outputState(double simTime, double deltaTime) {
   auto status = StatsProbe::outputState(simTime, deltaTime);
   if (status != Response::SUCCESS) {
      return status;
   }
   auto *publisherComponent = getTargetLayer()->getComponentByType<BasePublisherComponent>();
   int numExtNeurons = publisherComponent->getNumExtended();
   const float *A    = publisherComponent->getLayerData();
   bool failed       = false;
   for (int i = 0; i < numExtNeurons; i++) {
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
