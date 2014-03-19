/*
 * receiveFromPostProbe.cpp
 * Author: slundquist
 */

#include "ReceiveFromPostProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <assert.h>
#include <string.h>

namespace PV {
ReceiveFromPostProbe::ReceiveFromPostProbe(const char * probeName, HyPerCol * hc)
   : StatsProbe()
{
   initReceiveFromPostProbe_base();
   initReceiveFromPostProbe(probeName, hc);
}

int ReceiveFromPostProbe::initReceiveFromPostProbe_base() { return PV_SUCCESS; }

int ReceiveFromPostProbe::initReceiveFromPostProbe(const char * probeName, HyPerCol * hc) {
   return initStatsProbe(probeName, hc);
}

void ReceiveFromPostProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   requireType(BufActivity);
}

int ReceiveFromPostProbe::outputState(double timed){
   int status = StatsProbe::outputState(timed);
   // const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();
   int numExtNeurons = getTargetLayer()->getNumExtended();
   const pvdata_t * A = getTargetLayer()->getLayerData();
   for (int i = 0; i < numExtNeurons; i++){
      //std::cout<<A[i]<<"\n";
      //For roundoff errors
      assert(abs(A[i]) < 1e-5);
   }
   return status;
}

}
