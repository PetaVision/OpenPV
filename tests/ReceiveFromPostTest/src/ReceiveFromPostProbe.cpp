/*
 * receiveFromPostProbe.cpp
 * Author: slundquist
 */

#include "ReceiveFromPostProbe.hpp"
#include <columns/Communicator.hpp>
#include <components/BasePublisherComponent.hpp>
#include <io/PVParams.hpp>
#include <layers/HyPerLayer.hpp>
#include <probes/ActivityBufferStatsProbeLocal.hpp>
#include <probes/StatsProbeImmediate.hpp>
#include <utils/PVLog.hpp>

#include <cmath>
#include <memory>

namespace PV {
ReceiveFromPostProbe::ReceiveFromPostProbe(
      const char *name,
      PVParams *params,
      Communicator const *comm)
      : StatsProbeImmediate() {
   initialize(name, params, comm);
}

void ReceiveFromPostProbe::checkStats() {
   auto *publisherComponent     = getTargetLayer()->getComponentByType<BasePublisherComponent>();
   long numExtNeurons           = publisherComponent->getNumExtended();
   int nbatch                   = publisherComponent->getLayerLoc()->nbatch;
   std::vector<float const *> A = publisherComponent->getLayerCube().data;
   bool failed                  = false;
   for (int b = 0; b < nbatch; ++b) {
     for (long i = 0; i < numExtNeurons; i++) {
        // For roundoff errors
        float a = A[b][i];
        if (std::fabs(a) >= mTolerance) {
           ErrorLog().printf(
                 "%s activity outside of tolerance %f: "
                 "batch element %d, extended index %ld has activity %f\n",
                 getDescription_c(),
                 (double)mTolerance,
                 b,
                 i,
                 (double)a);
           failed = true;
        }
        FatalIf(failed, "Test failed.\n");
     }
   }
}

void ReceiveFromPostProbe::createProbeLocal(char const *name, PVParams *params) {
   mProbeLocal = std::make_shared<ActivityBufferStatsProbeLocal>(name, params);
}

void ReceiveFromPostProbe::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   StatsProbeImmediate::initialize(name, params, comm);
}

int ReceiveFromPostProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = StatsProbeImmediate::ioParamsFillGroup(ioFlag);
   ioParam_tolerance(ioFlag);
   return status;
}

void ReceiveFromPostProbe::ioParam_tolerance(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "tolerance", &mTolerance, mTolerance);
}

} // end namespace PV
