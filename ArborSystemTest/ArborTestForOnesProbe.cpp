/*
 * ArborTestForOnesProbe.cpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#include "ArborTestForOnesProbe.hpp"
#include "../PetaVision/src/include/pv_arch.h"
#include "../PetaVision/src/layers/HyPerLayer.hpp"
#include <string.h>
#include <assert.h>

namespace PV {

ArborTestForOnesProbe::ArborTestForOnesProbe(const char * filename, HyPerCol * hc, const char * msg)
: StatsProbe(filename, hc, msg)
{
}

ArborTestForOnesProbe::ArborTestForOnesProbe(const char * msg)
: StatsProbe(msg)
{
}


int ArborTestForOnesProbe::outputState(float time, HyPerLayer * l)
{
	int status = StatsProbe::outputState(time, l);
#ifdef PV_USE_MPI
   InterColComm * icComm = l->getParent()->icCommunicator();
   const int rcvProc = 0;
   if( icComm->commRank() != rcvProc ) {
      return 0;
   }
#endif // PV_USE_MPI
	if(time>1.0f){
		assert((fMin>0.99)&&(fMin<1.01));
		assert((fMax>0.99)&&(fMax<1.01));
		assert((avg>0.99)&&(avg<1.01));
	}

	return status;
}


} /* namespace PV */
