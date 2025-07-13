#include "CheckStatsProbe.hpp"
#include <columns/Communicator.hpp>
#include <include/pv_common.h>
#include <params/PVParams.hpp>
#include <layers/HyPerLayer.hpp>
#include <probes/ProbeData.hpp>
#include <probes/StatsProbeImmediate.hpp>
#include <probes/StatsProbeTypes.hpp>
#include <utils/PVLog.hpp>

#include <cstdlib>

void CheckStatsProbe::ioParam_correctMin(PV::ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "correctMin", &mCorrectMin);
}

void CheckStatsProbe::ioParam_correctMax(PV::ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "correctMax", &mCorrectMax);
}

void CheckStatsProbe::ioParam_correctMean(PV::ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "correctMean", &mCorrectMean);
}

void CheckStatsProbe::ioParam_correctStd(PV::ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "correctStd", &mCorrectStd);
}

void CheckStatsProbe::ioParam_tolerance(PV::ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "tolerance", &mTolerance);
}

CheckStatsProbe::CheckStatsProbe(
      std::shared_ptr<PV::ParamGroup> params,
      std::shared_ptr<PV::ParamGroup> defaults,
      PV::Communicator const *comm) {
   initialize(params, defaults, comm);
}

CheckStatsProbe::CheckStatsProbe() {}

CheckStatsProbe::~CheckStatsProbe() {}

void CheckStatsProbe::checkStats() {
   int nbatch = getTargetLayer()->getLayerLoc()->nbatch;
   FatalIf(nbatch != 1, "%s is only written for nbatch = 1.\n", getDescription_c());
   if (mCommunicator->commRank() != 0) {
      return;
   }

   auto const &storedValues                   = mProbeAggregator->getStoredValues();
   auto numTimestamps                         = storedValues.size();
   int lastTimestampIndex                     = static_cast<int>(numTimestamps) - 1;
   PV::ProbeData<PV::LayerStats> const &stats = storedValues.getData(lastTimestampIndex);
   PV::LayerStats const &statsElem            = stats.getValue(0);
   int status                                 = PV_SUCCESS;
   if (std::abs(statsElem.mMin - mCorrectMin) > mTolerance) {
      ErrorLog().printf(
            "%s minimum value %f differs from expected value %f.\n",
            getTargetLayer()->getDescription_c(),
            (double)statsElem.mMin,
            (double)mCorrectMin);
      status = PV_FAILURE;
   }
   if (std::abs(statsElem.mMax - mCorrectMax) > mTolerance) {
      ErrorLog().printf(
            "%s maximum value %f differs from expected value %f.\n",
            getTargetLayer()->getDescription_c(),
            (double)statsElem.mMax,
            (double)mCorrectMax);
      status = PV_FAILURE;
   }
   double average, sigma;
   statsElem.derivedStats(average, sigma);
   if (std::abs(static_cast<float>(average) - mCorrectMean) > mTolerance) {
      ErrorLog().printf(
            "%s mean value value %f differs from expected value %f.\n",
            getTargetLayer()->getDescription_c(),
            average,
            (double)mCorrectMean);
      status = PV_FAILURE;
   }
   if (std::abs(static_cast<float>(sigma) - mCorrectStd) > mTolerance) {
      ErrorLog().printf(
            "%s standard deviation %f differs from expected value %f.\n",
            getTargetLayer()->getDescription_c(),
            sigma,
            (double)mCorrectStd);
      status = PV_FAILURE;
   }
   FatalIf(status != PV_SUCCESS, "%s failed.\n", getDescription_c());
}

void CheckStatsProbe::initialize(
      std::shared_ptr<PV::ParamGroup> params,
      std::shared_ptr<PV::ParamGroup> defaults,
      PV::Communicator const *comm) {
   StatsProbeImmediate::initialize(params, defaults, comm);
}

int CheckStatsProbe::ioParamsFillGroup(PV::ParamsIOSwitch ioSwitch) {
   int status = PV::StatsProbeImmediate::ioParamsFillGroup(ioSwitch);
   ioParam_correctMin(ioSwitch);
   ioParam_correctMax(ioSwitch);
   ioParam_correctMean(ioSwitch);
   ioParam_correctStd(ioSwitch);
   ioParam_tolerance(ioSwitch);
   return status;
}
