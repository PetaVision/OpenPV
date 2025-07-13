/*
 * receiveFromPostProbe.cpp
 * Author: slundquist
 */

#include "ReceiveFromPostProbe.hpp"
#include <columns/Communicator.hpp>
#include <components/BasePublisherComponent.hpp>
#include <params/PVParams.hpp>
#include <layers/HyPerLayer.hpp>
#include <probes/ActivityBufferStatsProbeLocal.hpp>
#include <probes/StatsProbeImmediate.hpp>
#include <utils/PVLog.hpp>

#include <cmath>
#include <memory>

namespace PV {
ReceiveFromPostProbe::ReceiveFromPostProbe(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm)
      : StatsProbeImmediate() {
   initialize(params, defaults, comm);
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

void ReceiveFromPostProbe::createProbeLocal(
      std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   mProbeLocal = std::make_shared<ActivityBufferStatsProbeLocal>(params, defaults);
}

void ReceiveFromPostProbe::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   StatsProbeImmediate::initialize(params, defaults, comm);
}

int ReceiveFromPostProbe::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = StatsProbeImmediate::ioParamsFillGroup(ioSwitch);
   ioParam_tolerance(ioSwitch);
   return status;
}

void ReceiveFromPostProbe::ioParam_tolerance(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "tolerance", &mTolerance);
}

} // end namespace PV
