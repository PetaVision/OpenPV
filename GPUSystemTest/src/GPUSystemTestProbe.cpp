/*
 * receiveFromPostProbe.cpp
 * Author: slundquist
 */

#include "GPUSystemTestProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <assert.h>
#include <string.h>

namespace PV {
GPUSystemTestProbe::GPUSystemTestProbe(const char * probeName, HyPerCol * hc)
   : StatsProbe()
{
   initGPUSystemTestProbe_base();
   initGPUSystemTestProbe(probeName, hc);
}

int GPUSystemTestProbe::initGPUSystemTestProbe_base() { return PV_SUCCESS; }

int GPUSystemTestProbe::initGPUSystemTestProbe(const char * probeName, HyPerCol * hc) {
   return initStatsProbe(probeName, hc);
}

void GPUSystemTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   requireType(BufActivity);
}

int GPUSystemTestProbe::outputState(double timed){
   int status = StatsProbe::outputState(timed);
   const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();
   int numExtNeurons = getTargetLayer()->getNumExtended();
   const pvdata_t * A = getTargetLayer()->getLayerData();
   std::cout.precision(15);
   for (int i = 0; i < numExtNeurons; i++){
      if(fabs(A[i]) != 0){
         int xpos = kxPos(i, loc->nx+loc->halo.lt+loc->halo.rt, loc->ny+loc->halo.dn+loc->halo.up, loc->nf);
         int ypos = kyPos(i, loc->nx+loc->halo.lt+loc->halo.rt, loc->ny+loc->halo.dn+loc->halo.up, loc->nf);
         int fpos = featureIndex(i, loc->nx+loc->halo.lt+loc->halo.rt, loc->ny+loc->halo.dn+loc->halo.up, loc->nf);
         //std::cout << "[" << xpos << "," << ypos << "," << fpos << "] = " << std::fixed << A[i] << "\n";
      }
      //For roundoff errors
      assert(fabs(A[i]) < 1e-4);
   }
   return status;
}

} // end namespace PV
