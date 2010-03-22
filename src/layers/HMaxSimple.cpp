/*
 * HMaxSimple.cpp
 *
 *  Created on: Nov 20, 2008
 *      Author: bjt
 */

#include "HMaxSimple.hpp"

namespace PV {

HMaxSimple::HMaxSimple(const char * name, HyPerCol * hc) : HyPerLayer(name, hc)
{
   initialize(TypeGeneric);
}

int HMaxSimple::recvSynapticInput(HyPerConn * conn, PVLayerCube * activity, int neighbor)
{
   pv_debug_info("[%d]: HMaxSimple::recvSynapticInput: layer %d from %d)",
                 clayer->columnId,
                 conn->preSynapticLayer()->clayer->layerId, clayer->layerId);

   // use implementation in base class
   HyPerLayer::recvSynapticInput(conn, activity, neighbor);

   return 0;
}

int HMaxSimple::updateState(float time, float dt)
{
   pv_debug_info("[%d]: HMaxSimple::updateState:", clayer->columnId);

   // just copy accumulation buffer to membrane potential
   // and activity buffer (nonspiking)

   pvdata_t * phi = clayer->phi[0];
   pvdata_t * activity = clayer->activity->data;

   // make sure activity in border is zero
   //
   // TODO - set numActive and active list?
   int numActive = 0;
   for (int k = 0; k < clayer->numExtended; k++) {
      activity[k] = 0.0;
   }

   for (int k = 0; k < clayer->numNeurons; k++) {
      int kPhi = kIndexExtended(k, clayer->loc.nx, clayer->loc.ny, clayer->numFeatures,
                                   clayer->loc.nPad);
      clayer->V[k] = phi[kPhi];
      activity[k] = phi[kPhi];
      phi[kPhi] = 0.0;     // reset accumulation buffer
   }

   return 0;
}

int HMaxSimple::initFinish(int colId, int colRow, int colCol)
{
   pv_debug_info("[%d]: HMaxSimple::initFinish: colId=%d colRow=%d, colCol=%d",
                 clayer->columnId, colId, colRow, colCol);
   return 0;
}

int HMaxSimple::setParams(int numParams, float* params)
{
   pv_debug_info("[%d]: HMaxSimple::setParams: numParams=%d", clayer->columnId, numParams);
   return 0;
}

int HMaxSimple::outputState(float time)
{
   pv_debug_info("[%d]: HMaxSimple::outputState:", clayer->columnId);

   // use implementation in base class
   HyPerLayer::outputState(time);

   return 0;
}

}
