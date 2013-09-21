/*
 * ArborTestProbe.cpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#include "ArborTestProbe.hpp"
#include <include/pv_arch.h> 
#include <layers/HyPerLayer.hpp>
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


int ArborTestProbe::outputState(double timed)
{
	int status = StatsProbe::outputState(timed);
#ifdef PV_USE_MPI
   InterColComm * icComm = getTargetLayer()->getParent()->icCommunicator();
   const int rcvProc = 0;
   if( icComm->commRank() != rcvProc ) {
      return 0;
   }
#endif // PV_USE_MPI
	if(timed==1.0f){
		assert((avg>0.2499)&&(avg<0.2501));
	}
	else if(timed==2.0f){
		assert((avg>0.4999)&&(avg<0.5001));
	}
	else if(timed==3.0f){
		assert((avg>0.7499)&&(avg<0.7501));
	}
	else if(timed>3.0f){
		assert((fMin>0.9999)&&(fMin<1.001));
		assert((fMax>0.9999)&&(fMax<1.001));
		assert((avg>0.9999)&&(avg<1.001));
	}

	return status;
}


} /* namespace PV */
