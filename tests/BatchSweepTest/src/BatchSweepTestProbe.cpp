/*
 * BatchSweepTestProbe.cpp
 *
 *  Created on: Aug 13, 2012
 *      Author: pschultz
 */

#include "BatchSweepTestProbe.hpp"

namespace PV {

BatchSweepTestProbe::BatchSweepTestProbe(const char * probeName, HyPerCol * hc) {
   initBatchSweepTestProbe(probeName, hc);
}

BatchSweepTestProbe::~BatchSweepTestProbe() {
}

int BatchSweepTestProbe::initBatchSweepTestProbe(const char * probeName, HyPerCol * hc) {
   int status = initStatsProbe(probeName, hc);
   return status;
}

int BatchSweepTestProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = StatsProbe::ioParamsFillGroup(ioFlag);
   ioParam_expectedSum(ioFlag);
   ioParam_expectedMin(ioFlag);
   ioParam_expectedMax(ioFlag);
   return status;
}

void BatchSweepTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   requireType(BufActivity);
}

void BatchSweepTestProbe::ioParam_expectedSum(enum ParamsIOFlag ioFlag) {
   getParent()->ioParamValue(ioFlag, getName(), "expectedSum", &expectedSum, 0.0);
}
void BatchSweepTestProbe::ioParam_expectedMin(enum ParamsIOFlag ioFlag) {
   getParent()->ioParamValue(ioFlag, getName(), "expectedMin", &expectedMin, 0.0f);
}

void BatchSweepTestProbe::ioParam_expectedMax(enum ParamsIOFlag ioFlag) {
   getParent()->ioParamValue(ioFlag, getName(), "expectedMax", &expectedMax, 0.0f);
}

int BatchSweepTestProbe::outputState(double timed) {
   int status = StatsProbe::outputState(timed);
   InterColComm * icComm = getTargetLayer()->getParent()->icCommunicator();
   const int rcvProc = 0;
   if( icComm->commRank() != rcvProc ) {
      return 0;
   }
   for(int b = 0; b < parent->getNBatch(); b++){
      if (timed >= 3.0 ) {
         assert(fabs(expectedSum - sum[b])<1e-6);
         assert(fabs(expectedMin - fMin[b])<1e-6);
         assert(fabs(expectedMax - fMax[b])<1e-6);
      }
   }
   return status;
}

BaseObject * createBatchSweepTestProbe(char const * name, HyPerCol * hc) {
   return hc ? new BatchSweepTestProbe(name, hc) : NULL;
}

} /* namespace PV */
