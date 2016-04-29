/*
 * DelayTestProbe.cpp
 *
 *  Created on: October 1, 2013
 *      Author: wchavez
 */

#include "DelayTestProbe.hpp"
#include <include/pv_arch.h> 
#include <layers/HyPerLayer.hpp>
#include <string.h>
#include <assert.h>

namespace PV {

DelayTestProbe::DelayTestProbe(const char * probeName, HyPerCol * hc)
: StatsProbe()
{
   initDelayTestProbe_base();
   initDelayTestProbe(probeName, hc);
}

DelayTestProbe::~DelayTestProbe() {}

int DelayTestProbe::initDelayTestProbe_base() { return PV_SUCCESS; }

int DelayTestProbe::initDelayTestProbe(const char * probeName, HyPerCol * hc) {
   return initStatsProbe(probeName, hc);
}

int DelayTestProbe::outputState(double timed)
{
   int status = StatsProbe::outputState(timed);
   InterColComm * icComm = getTargetLayer()->getParent()->icCommunicator();
   const int rcvProc = 0;
   if( icComm->commRank() != rcvProc ) {
      return 0;
   }
   const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();
   const int rows = getTargetLayer()->getParent()->icCommunicator()->numCommRows();
   const int cols = getTargetLayer()->getParent()->icCommunicator()->numCommColumns();

   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;

   for(int b = 0; b < loc->nbatch; b++){
      if (timed==0) {
         assert((avg[b] == (timed)/nf)&&(avg[b] == double(nnz[b])/((nx*rows)*(ny*cols)*nf)));
      }
      else {
         assert((avg[b] == (timed-1)/nf)&&(avg[b] == double(nnz[b])/((nx*rows)*(ny*cols)*nf)));
      }
   }
   return status;
}

BaseObject * createDelayTestProbe(char const * probeName, HyPerCol * hc) {
   return hc ? new DelayTestProbe(probeName, hc): NULL;
}

} /* namespace PV */
