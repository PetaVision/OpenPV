/*
 * KernelTestProbe.cpp
 *
 *  Created on: Sep 1, 2011
 *      Author: gkenyon
 */

#include "KernelTestProbe.hpp"
#include "../PetaVision/src/include/pv_arch.h"
#include "../PetaVision/src/layers/HyPerLayer.hpp"
#include <string.h>
#include <assert.h>

namespace PV {

KernelTestProbe::KernelTestProbe(const char * filename, HyPerLayer * layer, const char * msg)
: StatsProbe()
{
   initStatsProbe(filename, layer, BufActivity, msg);
}

KernelTestProbe::KernelTestProbe(HyPerLayer * layer, const char * msg)
: StatsProbe()
{
   initStatsProbe(NULL, layer, BufActivity, msg);
}


int KernelTestProbe::outputState(float timef)
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
      assert((fMin>0.99)&&(fMin<1.010));
      assert((fMax>0.99)&&(fMax<1.010));
      assert((avg>0.99)&&(avg<1.010));
   }

   return status;
}


} /* namespace PV */
