/*
 * HMaxSimple.cpp
 *
 *  Created on: Nov 20, 2008
 *      Author: bjt
 */

#include "HMaxSimple.hpp"
#include "../connections/KernelConn.hpp"

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
   //HyPerLayer::recvSynapticInput(conn, activity, neighbor);

   PVLayer * lPre  = conn->preSynapticLayer() ->clayer;
   PVLayer * lPost = conn->postSynapticLayer()->clayer;

   const PVLayerLoc locPost = lPost->loc;

   const int nxPost = locPost.nx;
   const int nyPost = locPost.ny;
   const int nfPost = lPost->numFeatures;

   const int nxPreEx = lPre->loc.nx;
   const int nyPreEx = lPre->loc.ny;
   const int nfPre   = lPre->numFeatures;
   const int nPadPre = lPre->loc.nPad;

   const int aStrideX = nfPre;
   const int aStrideY = nfPre * (nxPreEx + 2*nPadPre);

   KernelConn * kConn = dynamic_cast<KernelConn*>(conn);

#ifdef FORTRAN
   int iKernel = 0;  // TODO - fix for multiple orientations
   PVPatch * weights = kConn->getKernelPatch(iKernel);

   pvdata_t * w = weights->data;
   pvdata_t * V = lPost->V;

   const int nxp = weights->nx;
   const int nyp = weights->ny;

   call convolve(nx)


#endif

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
