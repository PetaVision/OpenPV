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

DelayTestProbe::DelayTestProbe(const char * filename, HyPerLayer * layer, const char * msg)
: StatsProbe(filename, layer, msg)
{
}

DelayTestProbe::DelayTestProbe(HyPerLayer * layer, const char * msg)
: StatsProbe(layer, msg)
{
}

DelayTestProbe::~DelayTestProbe() {}


int DelayTestProbe::outputState(double timed)
{
	int status = StatsProbe::outputState(timed);
#ifdef PV_USE_MPI
   InterColComm * icComm = getTargetLayer()->getParent()->icCommunicator();
   const int rcvProc = 0;
   if( icComm->commRank() != rcvProc ) {
      return 0;
   }
#endif // PV_USE_MPI
   const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();
   const int rows = getTargetLayer()->getParent()->icCommunicator()->numCommRows();
   const int cols = getTargetLayer()->getParent()->icCommunicator()->numCommColumns();

      int nx = loc->nx;
      int ny = loc->ny;
      int nf = loc->nf;
    
    if (timed==0) {
        assert((avg == (timed)/nf)&&(avg == double(nnz)/((nx*rows)*(ny*cols)*nf)));
    }
    
    else {
        assert((avg == (timed-1)/nf)&&(avg == double(nnz)/((nx*rows)*(ny*cols)*nf)));
    }
      return status;
}


} /* namespace PV */
