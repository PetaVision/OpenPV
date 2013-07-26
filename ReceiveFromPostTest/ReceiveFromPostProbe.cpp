/*
 * receiveFromPostProbe.cpp
 * Author: slundquist
 */

#include "ReceiveFromPostProbe.hpp"
#include "../PetaVision/src/include/pv_arch.h"
#include "../PetaVision/src/layers/HyPerLayer.hpp"
#include <assert.h>
#include <string.h>

namespace PV {
ReceiveFromPostProbe::ReceiveFromPostProbe(const char * filename, HyPerLayer * layer, const char * msg)
   : StatsProbe()
{
   initStatsProbe(filename, layer, BufActivity, msg);
}

ReceiveFromPostProbe::ReceiveFromPostProbe(HyPerLayer * layer, const char * msg)
   : StatsProbe()
{
   initStatsProbe(NULL, layer, BufActivity, msg);
}

int ReceiveFromPostProbe::outputState(double timed){
   int status = StatsProbe::outputState(timed);
   const pvdata_t * actLayer = getTargetLayer()->getLayerData();
   const PVLayerLoc * loc = getTargetLayer()->getLayerLoc(); 
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
