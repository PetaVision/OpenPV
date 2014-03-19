/*
 * GPUTestForOnesProbe.cpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#include "GPUTestForOnesProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <string.h>
#include <assert.h>

namespace PV {

GPUTestForOnesProbe::GPUTestForOnesProbe(const char * probeName, HyPerCol * hc)
: StatsProbe()
{
}

GPUTestForOnesProbe::~GPUTestForOnesProbe() {}

int GPUTestForOnesProbe::initGPUTestForOnesProbe_base() { return PV_SUCCESS; }

int GPUTestForOnesProbe::initGPUTestForOnesProbe(const char * probeName, HyPerCol * hc) {
   return initStatsProbe(probeName, hc);
}

int GPUTestForOnesProbe::outputState(double timed)
{
	int status = StatsProbe::outputState(timed);
#ifdef PV_USE_MPI
   InterColComm * icComm = getTargetLayer()->getParent()->icCommunicator();
   const int rcvProc = 0;
   if( icComm->commRank() != rcvProc ) {
      return 0;
   }
#endif // PV_USE_MPI
	if(timed>1.0f){
		assert((fMin>0.99)&&(fMin<1.01));
		assert((fMax>0.99)&&(fMax<1.01));
		assert((avg>0.99)&&(avg<1.01));
	}

	return status;
}


} /* namespace PV */
