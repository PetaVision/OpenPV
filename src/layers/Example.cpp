/*
 * Example.cpp
 *
 *  Created on: Oct 19, 2008
 *      Author: rasmussn
 */

#include "Example.hpp"
#include <stdio.h>

namespace PV {

Example::Example(const char * name, HyPerCol * hc) : HyPerLayer(name, hc)
{
   initialize(TypeGeneric);
}

int Example::recvSynapticInput(HyPerConn * conn, PVLayerCube * activity, int neighbor)
{
   pv_debug_info("[%d]: Example::recvSynapticInput: to layer %d from %d, neighbor %d",
                 clayer->columnId, clayer->layerId, conn->preSynapticLayer()->clayer->layerId, neighbor);

   // use implementation in base class
   HyPerLayer::recvSynapticInput(conn, activity, neighbor);

   return 0;
}

int Example::updateState(float time, float dt)
{
   pv_debug_info("[%d]: Example::updateState:", clayer->columnId);

   // just copy accumulation buffer to membrane potential
   // and activity buffer (nonspiking)

   pvdata_t * phi = clayer->phi[CHANNEL_EXC];
   pvdata_t * activity = clayer->activity->data;

   const float nx = clayer->loc.nx;
   const float ny = clayer->loc.ny;
   const float nf = clayer->numFeatures;
   const float marginWidth = clayer->loc.nPad;

   // make sure activity in border is zero
   //
   // TODO - set numActive and active list?
   int numActive = 0;
   for (int k = 0; k < clayer->numExtended; k++) {
      activity[k] = 0.0;
   }

   for (int k = 0; k < clayer->numNeurons; k++) {
      int kex = kIndexExtended(k, nx, ny, nf, marginWidth);
      clayer->V[k] = phi[k];
      activity[kex] = phi[k];
      phi[k] = 0.0;     // reset accumulation buffer
   }

   return 0;
}

int Example::initFinish(int colId, int colRow, int colCol)
{
   pv_debug_info("[%d]: Example::initFinish: colId=%d colRow=%d, colCol=%d",
                 clayer->columnId, colId, colRow, colCol);
   return 0;
}

int Example::setParams(int numParams, float* params)
{
   pv_debug_info("[%d]: Example::setParams: numParams=%d", clayer->columnId, numParams);
   return 0;
}

int Example::outputState(float time)
{
   pv_debug_info("[%d]: Example::outputState:", clayer->columnId);

   // use implementation in base class
   HyPerLayer::outputState(time);

   return 0;
}

} // namespace PV
