/*
 * ArborTestForOnesProbe.cpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#include "ArborTestForOnesProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <string.h>
#include <assert.h>

namespace PV {

ArborTestForOnesProbe::ArborTestForOnesProbe(const char * probeName, HyPerCol * hc)
: StatsProbe()
{
   initArborTestForOnesProbe_base();
   initArborTestForOnesProbe(probeName, hc);
}

ArborTestForOnesProbe::~ArborTestForOnesProbe() {}

int ArborTestForOnesProbe::initArborTestForOnesProbe_base() {return PV_SUCCESS;}

int ArborTestForOnesProbe::initArborTestForOnesProbe(const char * probeName, HyPerCol * hc) {
   return initStatsProbe(probeName, hc);
}

int ArborTestForOnesProbe::outputState(double timed)
{
   int status = StatsProbe::outputState(timed);
   InterColComm * icComm = getTargetLayer()->getParent()->icCommunicator();
   const int rcvProc = 0;
   if( icComm->commRank() != rcvProc ) {
      return 0;
   }
   if(timed>1.0f){
      for(int b = 0; b < getParent()->getNBatch(); b++){
         assert((fMin[b]>0.99)&&(fMin[b]<1.01));
         assert((fMax[b]>0.99)&&(fMax[b]<1.01));
         assert((avg[b]>0.99)&&(avg[b]<1.01));
      }
   }

   return status;
}

BaseObject * createArborTestForOnesProbe(char const * name, HyPerCol * hc) {
   return hc ? new ArborTestForOnesProbe(name, hc) : NULL;
}

} /* namespace PV */
