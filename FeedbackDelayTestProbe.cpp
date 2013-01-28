/*
 * FeedbackDelayTestProbe.cpp
 *
 *  Created on: January 27, 2013
 *      Author: garkenyon
 */

#include "FeedbackDelayTestProbe.hpp"
#include "../PetaVision/src/include/pv_arch.h"
#include "../PetaVision/src/layers/HyPerLayer.hpp"
#include <string.h>
#include <assert.h>

namespace PV {

FeedbackDelayTestProbe::FeedbackDelayTestProbe(const char * filename, HyPerLayer * layer, const char * msg)
: StatsProbe(filename, layer, msg)
{
	toggleOutput = false;
}

FeedbackDelayTestProbe::FeedbackDelayTestProbe(HyPerLayer * layer, const char * msg)
: StatsProbe(layer, msg)
{
	toggleOutput = false;
}


int FeedbackDelayTestProbe::outputState(double timed)
{
   int status = StatsProbe::outputState(timed);
#ifdef PV_USE_MPI
   InterColComm * icComm = getTargetLayer()->getParent()->icCommunicator();
   const int rcvProc = 0;
   if( icComm->commRank() != rcvProc ) {
      return 0;
   }
#endif // PV_USE_MPI
   double tol = 1e-6;
   // assert that output of each layer alternates between 1 and 0 after each update
   if (!toggleOutput) {
      assert(fabs(fMin) > (1-tol));
      assert(fabs(fMax) > (1-tol));
      assert(fabs(avg) > (1-tol));
   }
   else {
	      assert(fabs(fMin) < tol);
	      assert(fabs(fMax) < tol);
	      assert(fabs(avg) < tol);
	   }
   toggleOutput = !toggleOutput;

   return status;
}


} /* namespace PV */
