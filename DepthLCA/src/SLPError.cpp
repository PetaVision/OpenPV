
/*
 * SLPError.cpp
 *
 *  Created on: Mar 21, 2014
 *      Author: slundquist 
 */

#include "SLPError.hpp"

namespace PV {

SLPError::SLPError(const char * name, HyPerCol * hc):ANNLayer(name, hc){
}

int SLPError::initialize_base() {
   classError = false;
   return PV_SUCCESS;
}  // end of BatchConn::initialize_base()

int SLPError::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_classError(ioFlag);
   return status;
}

void SLPError::ioParam_classError(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "classError", &classError, classError);
}

int SLPError::updateState(double time, double dt)
{
   const PVLayerLoc * loc = getLayerLoc();

   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   //Reset pointer of gSynHead to point to the inhib channel
   pvdata_t * GSynExt = getChannel(CHANNEL_EXC);
   pvdata_t * GSynInh = getChannel(CHANNEL_INH);

   pvdata_t * A = getCLayer()->activity->data;
   pvdata_t * V = getV();

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
   for(int ni = 0; ni < num_neurons; ni++){
      int next = kIndexExtended(ni, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
      //expected - actual only if expected isn't 0
      if(GSynExt[ni] == 0){
         A[next] = 0;
      }
      else{
         if(classError){
            if(GSynExt[ni] > 0 && GSynInh[ni] > 1){
               A[next] = 0;
            }
            if(GSynExt[ni] < 0 && GSynInh[ni] < 0){
               A[next] = 0;
            }
         }
         else{
            A[next] = GSynExt[ni] - GSynInh[ni];
         }
      }
   }
   return PV_SUCCESS;
}

} /* namespace PV */
