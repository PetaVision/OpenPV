/*
 * CloneKernelConnTestProbe.cpp
 *
 *  Created on: Feb 24, 2012
 *      Author: peteschultz
 */

#include "CloneKernelConnTestProbe.hpp"
#include "../PetaVision/src/include/pv_arch.h"
#include "../PetaVision/src/layers/HyPerLayer.hpp"
#include <string.h>
#include <assert.h>

namespace PV {

CloneKernelConnTestProbe::CloneKernelConnTestProbe(const char * filename, HyPerCol * hc, const char * msg)
: StatsProbe(filename, hc, msg)
{
}

CloneKernelConnTestProbe::CloneKernelConnTestProbe(const char * msg)
: StatsProbe(msg)
{
}


int CloneKernelConnTestProbe::outputState(float time, HyPerLayer * l)
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
      assert(fabs(fMin) < 1e-6);
      assert(fabs(fMax) < 1e-6);
      assert(fabs(avg) < 1e-6);
   }

   return status;
}


} /* namespace PV */
