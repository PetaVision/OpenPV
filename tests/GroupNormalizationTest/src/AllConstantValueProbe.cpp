/*
 * AllConstantValueProbe.cpp
 */

#include "AllConstantValueProbe.hpp"
#include <columns/Communicator.hpp>
#include <include/pv_common.h>
#include <params/PVParams.hpp>
#include <probes/ProbeData.hpp>
#include <probes/StatsProbeImmediate.hpp>
#include <probes/StatsProbeTypes.hpp>
#include <utils/PVLog.hpp>

namespace PV {

AllConstantValueProbe::AllConstantValueProbe(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

AllConstantValueProbe::AllConstantValueProbe() {}

void AllConstantValueProbe::checkStats() {
   auto const &storedValues           = mProbeAggregator->getStoredValues();
   auto numTimestamps                 = storedValues.size();
   int lastTimestampIndex             = static_cast<int>(numTimestamps) - 1;
   ProbeData<LayerStats> const &stats = storedValues.getData(lastTimestampIndex);
   double simTime                     = stats.getTimestamp();
   if (simTime <= 0.0) {
      return;
   }
   float nnzThreshold = mProbeLocal->getNnzThreshold();
   int nbatch         = static_cast<int>(stats.size());
   int status         = PV_SUCCESS;
   for (int b = 0; b < nbatch; b++) {
      LayerStats const &statsElem = stats.getValue(b);
      float fMin                  = statsElem.mMin;
      float fMax                  = statsElem.mMax;
      if (fMin < mCorrectValue - nnzThreshold or fMax > mCorrectValue + nnzThreshold) {
         ErrorLog().printf(
               "t=%f: batch element %b, fMin=%f, fMax=%f; values more than "
               "nnzThreshold=%g away from correct value %f\n",
               simTime,
               b,
               (double)fMin,
               (double)fMax,
               (double)nnzThreshold,
               (double)mCorrectValue);
         status = PV_FAILURE;
      }
   }
   FatalIf(status != PV_SUCCESS, "Probe %s failed.\n", getDescription_c());
}

void AllConstantValueProbe::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   StatsProbeImmediate::initialize(paramsIO, comm);
}

int AllConstantValueProbe::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = StatsProbeImmediate::ioParamsFillGroup(ioSwitch);
   ioParam_correctValue(ioSwitch);
   return status;
}

void AllConstantValueProbe::ioParam_correctValue(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "correctValue", &mCorrectValue);
}

AllConstantValueProbe::~AllConstantValueProbe() {}

} // namespace PV
