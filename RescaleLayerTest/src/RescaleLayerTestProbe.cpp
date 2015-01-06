/*
 * RescaleLayerTestProbe.cpp
 *
 *  Created on: Sep 1, 2011
 *      Author: gkenyon
 */

#include "RescaleLayerTestProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <string.h>
#include <assert.h>

namespace PV {

RescaleLayerTestProbe::RescaleLayerTestProbe(const char * probeName, HyPerCol * hc)
: StatsProbe()
{
   initRescaleLayerTestProbe(probeName, hc);
}

int RescaleLayerTestProbe::initRescaleLayerTestProbe_base() { return PV_SUCCESS; }

int RescaleLayerTestProbe::initRescaleLayerTestProbe(const char * probeName, HyPerCol * hc)
{
   return initStatsProbe(probeName, hc);
}

void RescaleLayerTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   requireType(BufActivity);
}

int RescaleLayerTestProbe::outputState(double timed)
{
   int status = StatsProbe::outputState(timed);
#ifdef PV_USE_MPI
   InterColComm * icComm = getTargetLayer()->getParent()->icCommunicator();
   const int rcvProc = 0;
   if( icComm->commRank() != rcvProc ) {
      return 0;
   }
#endif // PV_USE_MPI
   return status;
}


} /* namespace PV */
