
/*
 * BinaryThresh.cpp
 *
 *  Created on: Mar 21, 2014
 *      Author: slundquist 
 */

#include "BinaryThresh.hpp"

namespace PV {

BinaryThresh::BinaryThresh(const char * name, HyPerCol * hc):ANNLayer(name, hc){
}

int BinaryThresh::updateState(double time, double dt)
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
      //Activity is either 0 or 1 based on if it's active
      A[next] = GSynExt[ni] == 0 ? 0 : 1;
   }
   return PV_SUCCESS;
}

} /* namespace PV */
