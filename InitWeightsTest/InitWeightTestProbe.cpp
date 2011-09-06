/*
 * InitWeightTestProbe.cpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#include "InitWeightTestProbe.hpp"
#include "../PetaVision/src/include/pv_arch.h"
#include "../PetaVision/src/layers/HyPerLayer.hpp"
#include <string.h>
#include <assert.h>

namespace PV {

InitWeightTestProbe::InitWeightTestProbe(const char * filename, HyPerCol * hc, const char * msg)
: StatsProbe(filename, hc, msg)
{
}

InitWeightTestProbe::InitWeightTestProbe(const char * msg)
: StatsProbe(msg)
{
}


int InitWeightTestProbe::outputState(float time, HyPerLayer * l)
{
	int status = StatsProbe::outputState(time, l);
#ifdef PV_USE_MPI
   InterColComm * icComm = l->getParent()->icCommunicator();
   const int rcvProc = 0;
   if( icComm->commRank() != rcvProc ) {
      return 0;
   }
#endif // PV_USE_MPI
	if(time>2.0f){
		assert((fMin>-0.999)&&(fMin<0.001));
		assert((fMax>-0.999)&&(fMax<0.001));
		assert((avg>-0.999)&&(avg<0.001));
	}

	return status;
}


} /* namespace PV */
