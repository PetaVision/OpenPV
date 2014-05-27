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
   std::cout << "nx: " << loc->nxGlobal << " ny: " << loc->nyGlobal << " nf: " << loc->nf << " nb: " << loc->nb << "\n";
   for (int i = 0; i < numExtNeurons; i++){
      //if(A[i] >= 1e-5){
      //   int xpos = kxPos(i, loc->nx+2*loc->nb, loc->ny+2*loc->nb, loc->nf);
      //   int ypos = kyPos(i, loc->nx+2*loc->nb, loc->ny+2*loc->nb, loc->nf);
      //   int fpos = featureIndex(i, loc->nx+2*loc->nb, loc->ny+2*loc->nb, loc->nf);
      //   std::cout << "[" << xpos << "," << ypos << "," << fpos << "] = " << A[i] << "\n";
      //}
      //For roundoff errors
      assert(abs(A[i]) < 1e-5);
   }
   return status;
}

}
