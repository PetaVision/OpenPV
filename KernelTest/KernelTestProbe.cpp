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

KernelTestProbe::KernelTestProbe(const char * filename, HyPerCol * hc, const char * msg)
: StatsProbe(filename, hc, msg)
{
}

KernelTestProbe::KernelTestProbe(const char * msg)
: StatsProbe(msg)
{
}


int KernelTestProbe::outputState(float time, HyPerLayer * l)
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
      assert((fMin>0.99)&&(fMin<1.010));
      assert((fMax>0.99)&&(fMax<1.010));
      assert((avg>0.99)&&(avg<1.010));
   }

   return status;
}


} /* namespace PV */
