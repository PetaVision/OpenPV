/*
 * identicalBatchProbe.cpp
 * Author: slundquist
 */

#include "identicalBatchProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <utils/PVLog.hpp>
#include <string.h>

namespace PV {
identicalBatchProbe::identicalBatchProbe(const char * probeName, HyPerCol * hc)
   : StatsProbe()
{
   initidenticalBatchProbe_base();
   initidenticalBatchProbe(probeName, hc);
}

int identicalBatchProbe::initidenticalBatchProbe_base() { return PV_SUCCESS; }

int identicalBatchProbe::initidenticalBatchProbe(const char * probeName, HyPerCol * hc) {
   return initStatsProbe(probeName, hc);
}

void identicalBatchProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   requireType(BufActivity);
}

//2 tests: max difference can be 5e-4, max std is 5e-5
int identicalBatchProbe::outputState(double timed){
   int status = StatsProbe::outputState(timed);
   const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();
   const pvdata_t * A = getTargetLayer()->getActivity();
   int numExtNeurons = getTargetLayer()->getNumExtended();
   for (int i = 0; i < numExtNeurons; i++){
      pvdata_t checkVal = A[i];
      for(int b = 0; b < loc->nbatch; b++){
         const pvdata_t * ABatch = A + b * getTargetLayer()->getNumExtended();
         float diff = fabs(checkVal - ABatch[i]);
         if(diff > 1e-4){
            pvError() << "Difference at neuron " << i << ", batch 0: " << checkVal << " batch " << b << ": " << ABatch[i] << "\n";
         }
         pvErrorIf(!(diff <= 1e-4), "Test failed.\n");
      }
   }
   return status;
}

} // end namespace PV
