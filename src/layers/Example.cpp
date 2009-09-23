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

   for (int k = 0; k < clayer->numNeurons; k++) {
#ifdef EXTEND_BORDER_INDEX
      int kPhi = kIndexExtended(k, clayer->loc.nx, clayer->loc.ny, clayer->numFeatures,
                                   clayer->loc.nPad);
#else
      int kPhi = k;
#endif
      clayer->V[k] = phi[kPhi];
      clayer->activity->data[k] = phi[kPhi];
      phi[kPhi] = 0.0;     // reset accumulation buffer
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
