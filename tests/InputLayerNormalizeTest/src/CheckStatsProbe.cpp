#include "CheckStatsProbe.hpp"
#include <columns/HyPerCol.hpp>
#include <layers/HyPerLayer.hpp>

void CheckStatsProbe::ioParam_buffer(enum PV::ParamsIOFlag ioFlag) {
   if (ioFlag == PV::PARAMS_IO_READ) {
      type                  = PV::BufActivity;
      char const *paramName = "buffer";
      PV::PVParams *params  = parent->parameters();
      if (params->stringPresent(name, paramName)) {
         char *paramValue = nullptr;
         parent->parameters()->ioParamString(
               ioFlag, getName(), paramName, &paramValue, "Activity", false);
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
   parent->parameters()->ioParamValue(ioFlag, getName(), "correctMin", &correctMin, correctMin);
}

void CheckStatsProbe::ioParam_correctMax(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, getName(), "correctMax", &correctMax, correctMax);
}

void CheckStatsProbe::ioParam_correctMean(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, getName(), "correctMean", &correctMean, correctMean);
}

void CheckStatsProbe::ioParam_correctStd(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, getName(), "correctStd", &correctStd, correctStd);
}

void CheckStatsProbe::ioParam_tolerance(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, getName(), "tolerance", &tolerance, tolerance);
}

CheckStatsProbe::CheckStatsProbe(char const *name, PV::HyPerCol *hc) {
   initialize(name, hc);
   initialize_base();
}

CheckStatsProbe::CheckStatsProbe() { initialize_base(); }

CheckStatsProbe::~CheckStatsProbe() {}

int CheckStatsProbe::initialize_base() { return PV_SUCCESS; }

int CheckStatsProbe::initialize(char const *name, PV::HyPerCol *hc) {
   return StatsProbe::initialize(name, hc);
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

PV::Response::Status CheckStatsProbe::outputState(double timestamp) {
   int nbatch = getTargetLayer()->getLayerLoc()->nbatch;
   FatalIf(nbatch != 1, "%s is only written for nbatch = 1.\n", getDescription_c());
   auto status = PV::StatsProbe::outputState(timestamp);
   if (status != PV::Response::SUCCESS) {
      return status;
   }
   PV::Communicator *icComm = parent->getCommunicator();
   if (icComm->commRank() != 0) {
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
