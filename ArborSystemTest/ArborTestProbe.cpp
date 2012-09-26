/*
 * ArborTestProbe.cpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#include "ArborTestProbe.hpp"
#include "../PetaVision/src/include/pv_arch.h"
#include "../PetaVision/src/layers/HyPerLayer.hpp"
#include <string.h>
#include <assert.h>

namespace PV {

ArborTestProbe::ArborTestProbe(const char * filename, HyPerLayer * layer, const char * msg)
: StatsProbe(filename, layer, msg)
{
}

ArborTestProbe::ArborTestProbe(HyPerLayer * layer, const char * msg)
: StatsProbe(layer, msg)
{
}

ArborTestProbe::~ArborTestProbe() {}


int ArborTestProbe::outputState(float timef)
{
	int status = StatsProbe::outputState(timef);
#ifdef PV_USE_MPI
   InterColComm * icComm = getTargetLayer()->getParent()->icCommunicator();
   const int rcvProc = 0;
   if( icComm->commRank() != rcvProc ) {
      return 0;
   }
#endif // PV_USE_MPI
	if(timef==3.0f){
		assert((avg>0.2499)&&(avg<0.2501));
	}
	else if(timef==4.0f){
		assert((avg>0.4999)&&(avg<0.5001));
	}
	else if(timef==5.0f){
		assert((avg>0.7499)&&(avg<0.7501));
	}
	else if(timef>5.0f){
		assert((fMin>0.9999)&&(fMin<1.001));
		assert((fMax>0.9999)&&(fMax<1.001));
		assert((avg>0.9999)&&(avg<1.001));
	}

	return status;
}


} /* namespace PV */
