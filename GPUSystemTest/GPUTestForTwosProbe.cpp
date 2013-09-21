/*
 * GPUTestForTwosProbe.cpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#include "GPUTestForTwosProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <string.h>
#include <assert.h>

namespace PV {

GPUTestForTwosProbe::GPUTestForTwosProbe(const char * filename, HyPerLayer * layer, const char * msg)
: StatsProbe(filename, layer, msg)
{
}

GPUTestForTwosProbe::GPUTestForTwosProbe(HyPerLayer * layer, const char * msg)
: StatsProbe(layer, msg)
{
}

GPUTestForTwosProbe::~GPUTestForTwosProbe() {}

int GPUTestForTwosProbe::outputState(double timed)
{
	int status = StatsProbe::outputState(timed);
#ifdef PV_USE_MPI
   InterColComm * icComm = getTargetLayer()->getParent()->icCommunicator();
   const int rcvProc = 0;
   if( icComm->commRank() != rcvProc ) {
      return 0;
   }
#endif // PV_USE_MPI
	if(timed>2.0f){
		assert((fMin>1.95)&&(fMin<2.01));
		assert((fMax>1.95)&&(fMax<2.01));
		assert((avg>1.95)&&(avg<2.01));
	}

	return status;
}


} /* namespace PV */
