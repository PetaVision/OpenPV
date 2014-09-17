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
   const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();
   int numExtNeurons = getTargetLayer()->getNumExtended();
   const pvdata_t * A = getTargetLayer()->getLayerData();
   std::cout.precision(15);
   std::cout << "outputing\n";
   for (int i = 0; i < numExtNeurons; i++){
      if(fabs(A[i]) != 0){
         int xpos = kxPos(i, loc->nx+loc->halo.lt+loc->halo.rt, loc->ny+loc->halo.dn+loc->halo.up, loc->nf);
         int ypos = kyPos(i, loc->nx+loc->halo.lt+loc->halo.rt, loc->ny+loc->halo.dn+loc->halo.up, loc->nf);
         int fpos = featureIndex(i, loc->nx+loc->halo.lt+loc->halo.rt, loc->ny+loc->halo.dn+loc->halo.up, loc->nf);
         //std::cout << "[" << xpos << "," << ypos << "," << fpos << "] = " << std::fixed << A[i] << "\n";
      }
      //For roundoff errors
      assert(fabs(A[i]) < 1e-6);
   }
   return status;
}

}
