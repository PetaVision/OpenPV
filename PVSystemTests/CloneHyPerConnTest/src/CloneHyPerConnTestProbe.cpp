/*
 * CloneHyPerConnTestProbe.cpp
 *
 *  Created on: Feb 24, 2012
 *      Author: peteschultz
 */

#include "CloneHyPerConnTestProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <string.h>
#include <assert.h>

namespace PV {

CloneHyPerConnTestProbe::CloneHyPerConnTestProbe(const char * probeName, HyPerCol * hc)
: StatsProbe()
{
   initCloneHyPerConnTestProbe_base();
   initCloneHyPerConnTestProbe(probeName, hc);
}

int CloneHyPerConnTestProbe::initCloneHyPerConnTestProbe_base() {return PV_SUCCESS;}

int CloneHyPerConnTestProbe::initCloneHyPerConnTestProbe(const char * probeName, HyPerCol * hc) {
   return initStatsProbe(probeName, hc);
}

int CloneHyPerConnTestProbe::outputState(double timed)
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
      assert(fabs(fMin) < 1e-6);
      assert(fabs(fMax) < 1e-6);
      assert(fabs(avg) < 1e-6);
   }

   return status;
}


} /* namespace PV */
