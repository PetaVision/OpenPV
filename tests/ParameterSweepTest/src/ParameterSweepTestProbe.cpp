/*
 * ParameterSweepTestProbe.cpp
 *
 *  Created on: Aug 13, 2012
 *      Author: pschultz
 */

#include "ParameterSweepTestProbe.hpp"

namespace PV {

ParameterSweepTestProbe::ParameterSweepTestProbe(const char *name, HyPerCol *hc) {
   initialize(name, hc);
}

ParameterSweepTestProbe::~ParameterSweepTestProbe() {}

int ParameterSweepTestProbe::initialize(const char *name, HyPerCol *hc) {
   int status = StatsProbe::initialize(name, hc);
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
   parent->parameters()->ioParamValue(ioFlag, getName(), "expectedSum", &expectedSum, 0.0);
}
void ParameterSweepTestProbe::ioParam_expectedMin(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, getName(), "expectedMin", &expectedMin, 0.0f);
}

void ParameterSweepTestProbe::ioParam_expectedMax(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, getName(), "expectedMax", &expectedMax, 0.0f);
}

Response::Status ParameterSweepTestProbe::outputState(double timed) {
   auto status = StatsProbe::outputState(timed);
   if (status != Response::SUCCESS) {
      return status;
   }
   Communicator *icComm = parent->getCommunicator();
   const int rcvProc    = 0;
   if (icComm->commRank() != rcvProc) {
      return status;
   }
   for (int b = 0; b < parent->getNBatch(); b++) {
      if (timed >= 3.0) {
         FatalIf(std::fabs(expectedSum - sum[b]) >= 1e-6, "Test failed.\n");
         FatalIf(std::fabs(expectedMin - fMin[b]) >= 1e-6f, "Test failed.\n");
         FatalIf(std::fabs(expectedMax - fMax[b]) >= 1e-6f, "Test failed.\n");
      }
   }
   return status;
}

} /* namespace PV */
