/*
 * AssertZerosProbe.cpp
 * Author: slundquist
 */

#include "AssertZerosProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <assert.h>
#include <string.h>

namespace PV {
AssertZerosProbe::AssertZerosProbe(const char * probeName, HyPerCol * hc)
   : StatsProbe()
{
   initAssertZerosProbe_base();
   initAssertZerosProbe(probeName, hc);
}

int AssertZerosProbe::initAssertZerosProbe_base() { return PV_SUCCESS; }

int AssertZerosProbe::initAssertZerosProbe(const char * probeName, HyPerCol * hc) {
   return initStatsProbe(probeName, hc);
}

void AssertZerosProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   requireType(BufActivity);
}

//2 tests: max difference can be 5e-4, max std is 5e-5
int AssertZerosProbe::outputState(double timed){
   int status = StatsProbe::outputState(timed);
   const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();
   int numExtNeurons = getTargetLayer()->getNumExtendedAllBatches();
   int numResNeurons = getTargetLayer()->getNumNeuronsAllBatches();
   const pvdata_t * A = getTargetLayer()->getLayerData();
   const pvdata_t * GSyn_E = getTargetLayer()->getChannel(CHANNEL_EXC);
   const pvdata_t * GSyn_I = getTargetLayer()->getChannel(CHANNEL_INH);


   std::cout.precision(15);
   float sumsq = 0;
   for (int i = 0; i < numExtNeurons; i++){
      //if(fabs(A[i]) != 0){
      //   int xpos = kxPos(i, loc->nx+loc->halo.lt+loc->halo.rt, loc->ny+loc->halo.dn+loc->halo.up, loc->nf);
      //   int ypos = kyPos(i, loc->nx+loc->halo.lt+loc->halo.rt, loc->ny+loc->halo.dn+loc->halo.up, loc->nf);
      //   int fpos = featureIndex(i, loc->nx+loc->halo.lt+loc->halo.rt, loc->ny+loc->halo.dn+loc->halo.up, loc->nf);
      //   std::cout << "[" << xpos << "," << ypos << "," << fpos << "] = " << std::fixed << A[i] << "\n";
      //}
      //For max difference roundoff errors
      assert(fabs(A[i]) < 5e-4);
   }

   if(timed > 0){
      //Make sure gsyn_e and gsyn_i are not all 0's
      float sum_E = 0;
      float sum_I = 0;
      for (int i = 0; i < numResNeurons; i++){
         sum_E += GSyn_E[i];
         sum_I += GSyn_I[i];
      }

      assert(sum_E != 0);
      assert(sum_I != 0);
   }

   for(int b = 0; b < loc->nbatch; b++){
      //For max std of 5e-5
      assert(sigma[b] <= 5e-5);
   }

   return status;
}

BaseObject * createAssertZerosProbe(char const * name, HyPerCol * hc) {
   return hc ? new AssertZerosProbe(name, hc) : NULL;
}

} // end namespace PV
