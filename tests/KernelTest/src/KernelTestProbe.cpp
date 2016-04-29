/*
 * KernelTestProbe.cpp
 *
 *  Created on: Sep 1, 2011
 *      Author: gkenyon
 */

#include "KernelTestProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <string.h>
#include <assert.h>

namespace PV {

KernelTestProbe::KernelTestProbe(const char * probeName, HyPerCol * hc)
: StatsProbe()
{
   initKernelTestProbe(probeName, hc);
}

int KernelTestProbe::initKernelTestProbe_base() { return PV_SUCCESS; }

int KernelTestProbe::initKernelTestProbe(const char * probeName, HyPerCol * hc)
{
   return initStatsProbe(probeName, hc);
}

void KernelTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   requireType(BufActivity);
}

int KernelTestProbe::outputState(double timed)
{
   int status = StatsProbe::outputState(timed);
   InterColComm * icComm = getTargetLayer()->getParent()->icCommunicator();
   const int rcvProc = 0;
   if( icComm->commRank() != rcvProc ) {
      return 0;
   }
   for(int b = 0; b < parent->getNBatch(); b++){
      if(timed>2.0f){
         assert((fMin[b]>0.99)&&(fMin[b]<1.010));
         assert((fMax[b]>0.99)&&(fMax[b]<1.010));
         assert((avg[b]>0.99)&&(avg[b]<1.010));
      }
   }

   return status;
}

BaseObject * createKernelTestProbe(char const * name, HyPerCol * hc) {
   return hc ? new KernelTestProbe(name, hc) : NULL;
}

} /* namespace PV */
