/*
 * GPUTestForNinesProbe.cpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#include "GPUTestForNinesProbe.hpp"
#include "../PetaVision/src/include/pv_arch.h"
#include "../PetaVision/src/layers/HyPerLayer.hpp"
#include <string.h>
#include <assert.h>

namespace PV {

GPUTestForNinesProbe::GPUTestForNinesProbe(const char * filename, HyPerLayer * layer, const char * msg)
: StatsProbe(filename, layer, msg)
{
}

GPUTestForNinesProbe::GPUTestForNinesProbe(HyPerLayer * layer, const char * msg)
: StatsProbe(layer, msg)
{
}

GPUTestForNinesProbe::~GPUTestForNinesProbe() {}

int GPUTestForNinesProbe::outputState(float timef)
{
	int status = StatsProbe::outputState(timef);
#ifdef PV_USE_MPI
   InterColComm * icComm = getTargetLayer()->getParent()->icCommunicator();
   const int rcvProc = 0;
   if( icComm->commRank() != rcvProc ) {
      return 0;
   }
#endif // PV_USE_MPI
	if(timef>2.0f){
		assert((fMin>8.99)&&(fMin<9.01));
		assert((fMax>8.99)&&(fMax<9.01));
		assert((avg>8.99)&&(avg<9.01));
	}

	return status;
}


} /* namespace PV */
