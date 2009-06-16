/*
 * SimplePoggio.cpp
 *
 *  Created on: Nov 20, 2008
 *      Author: bjt
 */

#include "SimplePoggio.hpp"

namespace PV {

SimplePoggio::SimplePoggio(const char * name, HyPerCol * hc) : HyPerLayer(name, hc)
{
}

int SimplePoggio::recvSynapticInput(HyPerConn * conn, PVLayerCube * activity, int neighbor)
{
   HyPerLayer * lPre = conn->preSynapticLayer();

   pv_debug_info("[%d]: SimplePoggio::recvSynapticInput: layer %d from %d)",
                 clayer->columnId, lPre->clayer->layerId, clayer->layerId);

   // use implementation in base class
   HyPerLayer::recvSynapticInput(conn, activity, neighbor);

   return 0;
}

int SimplePoggio::updateState(float time, float dt)
{
   pv_debug_info("[%d]: SimplePoggio::updateState:", clayer->columnId);

   // just copy accumulation buffer to membrane potential
   // and activity buffer (nonspiking)

   pvdata_t * phi = clayer->phi[0];

   for (int k = 0; k < clayer->numNeurons; k++) {
#ifdef EXTEND_BORDER_INDEX
      int kPhi = kIndexExtended(k, clayer->loc.nx, clayer->loc.ny, clayer->numFeatures,
                                   clayer->numBorder);
#else
      int kPhi = k;
#endif
      clayer->V[k] = phi[kPhi];
      clayer->activity->data[k] = phi[kPhi];
      phi[kPhi] = 0.0;     // reset accumulation buffer
   }

   return 0;
}

int SimplePoggio::initFinish(int colId, int colRow, int colCol)
{
   pv_debug_info("[%d]: SimplePoggio::initFinish: colId=%d colRow=%d, colCol=%d",
                 clayer->columnId, colId, colRow, colCol);
   return 0;
}

int SimplePoggio::setParams(int numParams, float* params)
{
   pv_debug_info("[%d]: SimplePoggio::setParams: numParams=%d", clayer->columnId, numParams);
   return 0;
}

int SimplePoggio::outputState(float time)
{
   pv_debug_info("[%d]: SimplePoggio::outputState:", clayer->columnId);

   // use implementation in base class
   HyPerLayer::outputState(time);

   return 0;
}

}
