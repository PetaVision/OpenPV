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

#ifdef PV_USE_OPENCL
int Simple::initializeThreadData()
{
   int status = CL_SUCCESS;
#ifdef IMPLEMENT_ME
   // map layer buffers so that layer data can be initialized
   //
   pvdata_t * V = (pvdata_t *)   clBuffers.V->map(CL_MAP_WRITE);
   pvdata_t * Vth = (pvdata_t *) clBuffers.Vth->map(CL_MAP_WRITE);

   // initialize layer data
   //
   for (int k = 0; k < clayer->numNeurons; k++){
      V[k] = V_REST;
   }

   for (int k = 0; k < clayer->numNeurons; k++){
      Vth[k] = VTH_REST;
   }

   clBuffers.V->unmap(V);
   clBuffers.Vth->unmap(Vth);
#endif
   return status;
}

int Simple::initializeThreadKernels()
{
   int status = CL_SUCCESS;

   // create kernels
   //
   updatestate_kernel = parent->getCLDevice()->createKernel("LIF_updatestate.cl", "update_state");

   int argid = 0;
   status |= updatestate_kernel->setKernelArg(argid++, clBuffers.V);
   status |= updatestate_kernel->setKernelArg(argid++, clBuffers.G_E);
   status |= updatestate_kernel->setKernelArg(argid++, clBuffers.G_I);
   status |= updatestate_kernel->setKernelArg(argid++, clBuffers.G_IB);
   status |= updatestate_kernel->setKernelArg(argid++, clBuffers.phi);
   status |= updatestate_kernel->setKernelArg(argid++, clBuffers.activity);
   status |= updatestate_kernel->setKernelArg(argid++, clayer->loc.nx);
   status |= updatestate_kernel->setKernelArg(argid++, clayer->loc.ny);
   status |= updatestate_kernel->setKernelArg(argid++, clayer->numFeatures);
   status |= updatestate_kernel->setKernelArg(argid++, clayer->loc.nPad);

   return status;
}
#endif

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
