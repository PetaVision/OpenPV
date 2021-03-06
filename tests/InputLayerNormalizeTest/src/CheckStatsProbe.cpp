#include "CheckStatsProbe.hpp"
#include <layers/HyPerLayer.hpp>

void CheckStatsProbe::ioParam_buffer(enum PV::ParamsIOFlag ioFlag) {
   if (ioFlag == PV::PARAMS_IO_READ) {
      type                  = PV::BufActivity;
      char const *paramName = "buffer";
      PV::PVParams *params  = parameters();
      if (params->stringPresent(name, paramName)) {
         char *paramValue = nullptr;
         parameters()->ioParamString(ioFlag, getName(), paramName, &paramValue, "Activity", false);
         pvAssert(paramValue);
         for (size_t c = 0; paramValue[c]; c++) {
            paramValue[c] = (char)tolower((int)paramValue[c]);
         }
         FatalIf(
               strcmp(paramValue, "a") and strcmp(paramValue, "activity"),
               "%s sets buffer to be \"Activity\".\n");
         free(paramValue);
         paramValue = NULL;
      }
   }
}

void CheckStatsProbe::ioParam_correctMin(enum PV::ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "correctMin", &correctMin, correctMin);
}

void CheckStatsProbe::ioParam_correctMax(enum PV::ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "correctMax", &correctMax, correctMax);
}

void CheckStatsProbe::ioParam_correctMean(enum PV::ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "correctMean", &correctMean, correctMean);
}

void CheckStatsProbe::ioParam_correctStd(enum PV::ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "correctStd", &correctStd, correctStd);
}

void CheckStatsProbe::ioParam_tolerance(enum PV::ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "tolerance", &tolerance, tolerance);
}

CheckStatsProbe::CheckStatsProbe(
      char const *name,
      PV::PVParams *params,
      PV::Communicator const *comm) {
   initialize(name, params, comm);
   initialize_base();
}

CheckStatsProbe::CheckStatsProbe() { initialize_base(); }

CheckStatsProbe::~CheckStatsProbe() {}

int CheckStatsProbe::initialize_base() { return PV_SUCCESS; }

void CheckStatsProbe::initialize(
      char const *name,
      PV::PVParams *params,
      PV::Communicator const *comm) {
   StatsProbe::initialize(name, params, comm);
}

int CheckStatsProbe::ioParamsFillGroup(enum PV::ParamsIOFlag ioFlag) {
   int status = PV::StatsProbe::ioParamsFillGroup(ioFlag);
   ioParam_correctMin(ioFlag);
   ioParam_correctMax(ioFlag);
   ioParam_correctMean(ioFlag);
   ioParam_correctStd(ioFlag);
   ioParam_tolerance(ioFlag);
   return status;
}

PV::Response::Status CheckStatsProbe::outputState(double simTime, double deltaTime) {
   int nbatch = getTargetLayer()->getLayerLoc()->nbatch;
   FatalIf(nbatch != 1, "%s is only written for nbatch = 1.\n", getDescription_c());
   auto status = PV::StatsProbe::outputState(simTime, deltaTime);
   if (status != PV::Response::SUCCESS) {
      return status;
   }
   if (mCommunicator->commRank() != 0) {
      return status;
   }
   FatalIf(
         std::abs(fMin[0] - correctMin) > tolerance,
         "%s minimum value %f differs from expected value %f.\n",
         getTargetLayer()->getDescription_c(),
         (double)fMin[0],
         (double)correctMin);
   FatalIf(
         std::abs(fMax[0] - correctMax) > tolerance,
         "%s maximum value %f differs from expected value %f.\n",
         getTargetLayer()->getDescription_c(),
         (double)fMax[0],
         (double)correctMax);
   FatalIf(
         std::abs(avg[0] - correctMean) > tolerance,
         "%s mean value %f differs from expected value %f.\n",
         getTargetLayer()->getDescription_c(),
         (double)avg[0],
         (double)correctMean);
   FatalIf(
         std::abs(sigma[0] - correctStd) > tolerance,
         "%s standard deviation %f differs from expected value %f.\n",
         getTargetLayer()->getDescription_c(),
         (double)sigma[0],
         (double)correctStd);
   return status;
}
