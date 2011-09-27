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

ArborTestProbe::ArborTestProbe(const char * filename, HyPerCol * hc, const char * msg)
: StatsProbe(filename, hc, msg)
{
}

ArborTestProbe::ArborTestProbe(const char * msg)
: StatsProbe(msg)
{
}


int ArborTestProbe::outputState(float time, HyPerLayer * l)
{
	int status = StatsProbe::outputState(time, l);
#ifdef PV_USE_MPI
   InterColComm * icComm = l->getParent()->icCommunicator();
   const int rcvProc = 0;
   if( icComm->commRank() != rcvProc ) {
      return 0;
   }
#endif // PV_USE_MPI
	if(time>5.0f){
		assert((fMin>0.97)&&(fMin<1.070));
		assert((fMax>0.97)&&(fMax<1.070));
		assert((avg>0.97)&&(avg<1.070));
	}

	return status;
}


} /* namespace PV */
