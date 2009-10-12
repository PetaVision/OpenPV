/*
 * Simple.cpp
 *
 *  Created on: Oct 10, 2009
 *      Author: travel
 */

#include "Simple.hpp"

namespace PV {

Simple::Simple(const char* name, HyPerCol * hc)
      : HyPerLayer(name, hc)
{
   initialize(TypeSimple);
}

int Simple::recvSynapticInput(HyPerConn* conn, PVLayerCube* activity, int neighbor)
{
   HyPerLayer::recvSynapticInput(conn,activity,neighbor);
   // just copy input to V?  What about size differences?  Convolve?
   return 0;
}


int Simple::reconstruct(HyPerConn * conn, PVLayerCube * cube)
{
   // TODO - implement
   printf("[%d]: Simple::reconstruct: to layer %d from %d\n",
          clayer->columnId, clayer->layerId, conn->preSynapticLayer()->clayer->layerId);
   return 0;
}

int Simple::updateState(float time, float dt)
{
   pvdata_t * phi = clayer->phi[CHANNEL_EXC];

   const float nx = clayer->loc.nx;
   const float ny = clayer->loc.ny;
   const float nf = clayer->numFeatures;
   const float marginWidth = clayer->loc.nPad;

   for (int k = 0; k < clayer->numNeurons; k++) {
      int kex = kIndexExtended(k, nx, ny, nf, marginWidth);
      clayer->V[k] = phi[k];
      clayer->activity->data[kex] = phi[k];
      phi[k] = 0.0;     // reset accumulation buffer
   }

   return 0;
}

} // namespace PV
