/*
 * Example.cpp
 *
 *  Created on: Oct 19, 2008
 *      Author: rasmussn
 */

#include "Example.hpp"
#include <stdio.h>

namespace PV {

Example::Example(const char * name, HyPerCol * hc)
{
   numChannels = 1;
   initialize(name, hc);
}

#ifdef PV_USE_OPENCL
int Example::initializeThreadBuffers(const char * kernelName)
{
   return 0;
}

int Example::initializeThreadKernels(const char * kernelName)
{
   return 0;
}
#endif

int Example::updateState(double time, double dt)
{
   pv_debug_info("[%d]: Example::updateState:", clayer->columnId);

   // just copy accumulation buffer to membrane potential
   // and activity buffer (nonspiking)

   pvdata_t * phi = getChannel(CHANNEL_EXC);
   pvdata_t * activity = clayer->activity->data;

   const int nx = clayer->loc.nx;
   const int ny = clayer->loc.ny;
   const int nf = clayer->loc.nf;
   const PVHalo * halo = &clayer->loc.halo;

   // make sure activity in border is zero
   //
   // TODO - set numActive and active list?
   for (int k = 0; k < clayer->numExtended; k++) {
      activity[k] = 0.0;
   }

   for (int k = 0; k < clayer->numNeurons; k++) {
      int kex = kIndexExtended(k, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
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

int Example::outputState(double timef, bool last)
{
   pv_debug_info("[%d]: Example::outputState:", clayer->columnId);

   // use implementation in base class
   HyPerLayer::outputState(timef);

   return 0;
}

} // namespace PV
