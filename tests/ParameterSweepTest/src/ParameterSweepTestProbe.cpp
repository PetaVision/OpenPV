/*
 * ParameterSweepTestProbe.cpp
 *
 *  Created on: Aug 13, 2012
 *      Author: pschultz
 */

#include "ParameterSweepTestProbe.hpp"

namespace PV {

ParameterSweepTestProbe::ParameterSweepTestProbe(const char *probeName, HyPerCol *hc) {
   initParameterSweepTestProbe(probeName, hc);
}

ParameterSweepTestProbe::~ParameterSweepTestProbe() {}

int ParameterSweepTestProbe::initParameterSweepTestProbe(const char *probeName, HyPerCol *hc) {
   int status = initStatsProbe(probeName, hc);
   return status;
}

int ParameterSweepTestProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = StatsProbe::ioParamsFillGroup(ioFlag);
   ioParam_expectedSum(ioFlag);
   ioParam_expectedMin(ioFlag);
   ioParam_expectedMax(ioFlag);
   return status;
}

void ParameterSweepTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) { requireType(BufActivity); }

void ParameterSweepTestProbe::ioParam_expectedSum(enum ParamsIOFlag ioFlag) {
   getParent()->parameters()->ioParamValue(ioFlag, getName(), "expectedSum", &expectedSum, 0.0);
}
void ParameterSweepTestProbe::ioParam_expectedMin(enum ParamsIOFlag ioFlag) {
   getParent()->parameters()->ioParamValue(ioFlag, getName(), "expectedMin", &expectedMin, 0.0f);
}

void ParameterSweepTestProbe::ioParam_expectedMax(enum ParamsIOFlag ioFlag) {
   getParent()->parameters()->ioParamValue(ioFlag, getName(), "expectedMax", &expectedMax, 0.0f);
}

int ParameterSweepTestProbe::outputState(double timed) {
   int status           = StatsProbe::outputState(timed);
   Communicator *icComm = getTargetLayer()->getParent()->getCommunicator();
   const int rcvProc    = 0;
   if (icComm->commRank() != rcvProc) {
      return 0;
   }
   for (int b = 0; b < parent->getNBatch(); b++) {
      if (timed >= 3.0) {
         FatalIf(!(fabs(expectedSum - sum[b]) < 1e-6), "Test failed.\n");
         FatalIf(!(fabs(expectedMin - fMin[b]) < 1e-6), "Test failed.\n");
         FatalIf(!(fabs(expectedMax - fMax[b]) < 1e-6), "Test failed.\n");
      }
   }
   return status;
}

} /* namespace PV */
