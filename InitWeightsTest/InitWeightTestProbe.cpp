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

InitWeightTestProbe::InitWeightTestProbe(const char * filename, HyPerLayer * layer, const char * msg)
: StatsProbe()
{
   initStatsProbe(filename, layer, BufActivity, msg);
}

InitWeightTestProbe::InitWeightTestProbe(HyPerLayer * layer, const char * msg)
: StatsProbe()
{
   initStatsProbe(NULL, layer, BufActivity, msg);
}


int InitWeightTestProbe::outputState(double timed)
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
      assert((fMin>-0.001)&&(fMin<0.001));
      assert((fMax>-0.001)&&(fMax<0.001));
      assert((avg>-0.001)&&(avg<0.001));
   }

   return status;
}


} /* namespace PV */
