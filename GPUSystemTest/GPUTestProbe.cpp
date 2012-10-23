/*
 * GPUTestProbe.cpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#include "GPUTestProbe.hpp"
#include "../PetaVision/src/include/pv_arch.h"
#include "../PetaVision/src/layers/HyPerLayer.hpp"
#include <string.h>
#include <assert.h>

namespace PV {

GPUTestProbe::GPUTestProbe(const char * filename, HyPerLayer * layer, const char * msg)
: StatsProbe(filename, layer, msg)
{
}

GPUTestProbe::GPUTestProbe(HyPerLayer * layer, const char * msg)
: StatsProbe(layer, msg)
{
}

GPUTestProbe::~GPUTestProbe() {}


int GPUTestProbe::outputState(double timed)
{
	int status = StatsProbe::outputState(timed);
#ifdef PV_USE_MPI
   InterColComm * icComm = getTargetLayer()->getParent()->icCommunicator();
   const int rcvProc = 0;
   if( icComm->commRank() != rcvProc ) {
      return 0;
   }
#endif // PV_USE_MPI
	if(timed==3.0f){
		assert((avg>0.24)&&(avg<0.29));
	}
	else if(timed==4.0f){
		assert((avg>0.4)&&(avg<0.61));
	}
	else if(timed==5.0f){
		assert((avg>0.74)&&(avg<0.761));
	}
	else if(timed>5.0f){
		assert((fMin>0.97)&&(fMin<1.070));
		assert((fMax>0.97)&&(fMax<1.070));
		assert((avg>0.97)&&(avg<1.070));
	}

	return status;
}


} /* namespace PV */
