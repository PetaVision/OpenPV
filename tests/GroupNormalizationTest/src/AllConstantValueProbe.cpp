/*
 * AllConstantValueProbe.cpp
 */

#include "AllConstantValueProbe.hpp"
#include <columns/HyPerCol.hpp>
#include <layers/HyPerLayer.hpp>

namespace PV {

AllConstantValueProbe::AllConstantValueProbe(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

AllConstantValueProbe::AllConstantValueProbe() { initialize_base(); }

int AllConstantValueProbe::initialize_base() {
   correctValue = (float)0;
   return PV_SUCCESS;
}

int AllConstantValueProbe::initialize(char const *name, HyPerCol *hc) {
   return StatsProbe::initialize(name, hc);
}

int AllConstantValueProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = StatsProbe::ioParamsFillGroup(ioFlag);
   ioParam_correctValue(ioFlag);
   return status;
}

void AllConstantValueProbe::ioParam_correctValue(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, getName(), "correctValue", &correctValue, correctValue /*default*/);
}

Response::Status AllConstantValueProbe::outputState(double timestamp) {
   auto status = StatsProbe::outputState(timestamp);
   if (status != Response::SUCCESS) {
      return status;
   }
   if (timestamp <= 0) {
      return status;
   }
   if (!mOutputStreams.empty()) {
      int nbatch = getTargetLayer()->getLayerLoc()->nbatch;
      FatalIf(
            nbatch != (int)mOutputStreams.size(),
            "Number of output streams for %s does not agree with local batch width.\n",
            getDescription_c());
      int globalBatchOffset =
            nbatch * (getMPIBlock()->getStartBatch() + getMPIBlock()->getBatchIndex());
      for (int b = 0; b < nbatch; b++) {
         int globalBatchIndex = globalBatchOffset + b;
         if (fMin[b] < correctValue - nnzThreshold or fMax[b] > correctValue + nnzThreshold) {
            output(b).printf(
                  "     Values outside of tolerance nnzThreshold=%f\n", (double)nnzThreshold);
            ErrorLog().printf(
                  "t=%f: fMin=%f, fMax=%f; values more than nnzThreshold=%g away from correct "
                  "value %f\n",
                  timestamp,
                  (double)fMin[b],
                  (double)fMax[b],
                  (double)nnzThreshold,
                  (double)correctValue);
         }
      }
   }
   return status;
}

AllConstantValueProbe::~AllConstantValueProbe() {}

} // namespace PV
