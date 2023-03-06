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
   auto *publisherComponent = getTargetLayer()->getComponentByType<BasePublisherComponent>();
   int numExtNeurons        = publisherComponent->getNumExtended();
   const float *A           = publisherComponent->getLayerData();
   bool failed              = false;
   for (int i = 0; i < numExtNeurons; i++) {
      // For roundoff errors
      if (std::fabs(A[i]) >= mTolerance) {
         ErrorLog().printf(
               "%s activity outside of tolerance %f: extended index %d has activity %f\n",
               getDescription_c(),
               (double)mTolerance,
               i,
               (double)A[i]);
         failed = true;
      }
      FatalIf(failed, "Test failed.\n");
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
