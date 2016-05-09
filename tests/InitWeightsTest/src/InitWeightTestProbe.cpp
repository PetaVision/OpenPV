/*
 * InitWeightTestProbe.cpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#include "InitWeightTestProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <string.h>
#include <assert.h>

namespace PV {

InitWeightTestProbe::InitWeightTestProbe(const char * probeName, HyPerCol * hc)
: StatsProbe()
{
   initInitWeightTestProbe(probeName, hc);
}

int InitWeightTestProbe::initInitWeightTestProbe_base() { return PV_SUCCESS; }

int InitWeightTestProbe::initInitWeightTestProbe(const char * probeName, HyPerCol * hc) {
   return initStatsProbe(probeName, hc);
}

void InitWeightTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   requireType(BufActivity);
}


int InitWeightTestProbe::outputState(double timed)
{
   int status = StatsProbe::outputState(timed);
   InterColComm * icComm = getTargetLayer()->getParent()->icCommunicator();
   const int rcvProc = 0;
   if( icComm->commRank() != rcvProc ) {
      return 0;
   }
   for(int b = 0; b < parent->getNBatch(); b++){
      if(timed>2.0f){
         assert((fMin[b]>-0.001)&&(fMin[b]<0.001));
         assert((fMax[b]>-0.001)&&(fMax[b]<0.001));
         assert((avg[b]>-0.001)&&(avg[b]<0.001));
      }
   }

   return status;
}

BaseObject * createInitWeightTestProbe(char const * name, HyPerCol * hc) {
   return hc ? new InitWeightTestProbe(name, hc) : NULL;
}

} /* namespace PV */
