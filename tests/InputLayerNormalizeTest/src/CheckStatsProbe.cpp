#include "CheckStatsProbe.hpp"
#include <columns/Communicator.hpp>
#include <include/pv_common.h>
#include <io/PVParams.hpp>
#include <layers/HyPerLayer.hpp>
#include <probes/ProbeData.hpp>
#include <probes/StatsProbeImmediate.hpp>
#include <probes/StatsProbeTypes.hpp>
#include <utils/PVLog.hpp>

#include <cstdlib>

void CheckStatsProbe::ioParam_correctMin(enum PV::ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "correctMin", &mCorrectMin, mCorrectMin);
}

void CheckStatsProbe::ioParam_correctMax(enum PV::ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "correctMax", &mCorrectMax, mCorrectMax);
}

void CheckStatsProbe::ioParam_correctMean(enum PV::ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "correctMean", &mCorrectMean, mCorrectMean);
}

void CheckStatsProbe::ioParam_correctStd(enum PV::ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "correctStd", &mCorrectStd, mCorrectStd);
}

void CheckStatsProbe::ioParam_tolerance(enum PV::ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "tolerance", &mTolerance, mTolerance);
}

CheckStatsProbe::CheckStatsProbe(
      char const *name,
      PV::PVParams *params,
      PV::Communicator const *comm) {
   initialize(name, params, comm);
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
      char const *name,
      PV::PVParams *params,
      PV::Communicator const *comm) {
   StatsProbeImmediate::initialize(name, params, comm);
}

int CheckStatsProbe::ioParamsFillGroup(enum PV::ParamsIOFlag ioFlag) {
   int status = PV::StatsProbeImmediate::ioParamsFillGroup(ioFlag);
   ioParam_correctMin(ioFlag);
   ioParam_correctMax(ioFlag);
   ioParam_correctMean(ioFlag);
   ioParam_correctStd(ioFlag);
   ioParam_tolerance(ioFlag);
   return status;
}
