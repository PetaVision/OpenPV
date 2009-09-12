/*
 * LGN.cpp
 *
 *  Created on: Jul 30, 2008
 *
 */

#include "../include/pv_common.h"
#include "LGN.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

namespace PV {

LGN::LGN(const char * name, HyPerCol * hc) : HyPerLayer(name, hc)
{
}

int LGN::recvSynapticInput(HyPerConn * conn, PVLayerCube * activity, int neighbor)
{
   HyPerLayer* lPre = conn->preSynapticLayer();

   pv_debug_info("[%d]: LGN::recvSynapticInput: layer %d from %d)",
                 clayer->columnId, lPre->clayer->layerId, clayer->layerId);

   // use implementation in base class
   return HyPerLayer::recvSynapticInput(conn, activity, neighbor);
}

int LGN::updateState(float time, float dt)
{
   pv_debug_info("[%d]: LGN::updateState:", clayer->columnId);

   // just copy accumulation buffer to membrane potential
   // and activity buffer (nonspiking)

   pvdata_t * PhiExc = clayer->phi[CHANNEL_EXC];
   pvdata_t * PhiInh = clayer->phi[CHANNEL_INH];

   for (int k = 0; k < clayer->numNeurons; k++) {
#ifdef EXTEND_BORDER_INDEX
      int kPhi = kIndexExtended(k, clayer->loc.nx, clayer->loc.ny, clayer->numFeatures,
                                   clayer->loc.nPad);
#else
      int kPhi = k;
#endif
      clayer->V[k] = PhiExc[kPhi];
      clayer->activity->data[k] = PhiExc[kPhi] - PhiInh[kPhi];
      //if (k > 64*2*32 && k < 2*64*32+128) {
      //   printf("PhiExc[%d]=%f, PhiInh[%d]=%f\n", k, PhiExc[kPhi], kPhi, PhiInh[kPhi]);
      //}
      if (clayer->activity->data[k] < 0.0) {
         clayer->activity->data[k] = 0.0;
      }
      PhiExc[kPhi] = 0.0;     // reset accumulation buffers
      PhiInh[kPhi] = 0.0;
   }
   return 0;
}


int LGN::initFinish(int colId, int colRow, int colCol)
{
   pv_debug_info("[%d]: Example::initFinish: colId=%d colRow=%d, colCol=%d\n", clayer->columnId,
          colId, colRow, colCol);
   return 0;
}

int LGN::setParams(int numParams, float* params)
{
   pv_debug_info("[%d]: Example::setParams: numParams=%d\n", clayer->columnId, numParams);
   return 0;
}

int LGN::outputState(float time)
{
   pv_debug_info("[%d]: LGN::outputState:\n", clayer->columnId);
   return HyPerLayer::outputState(time);
}


}

#ifdef __cplusplus
extern "C" {
#endif

// TODO - no longer needed
#ifdef DELETE
static int PV_LGN_kNN(float x0, float y0, int rx, int ry, int k[])
{
   int i, j, i1, j1, t, imin, imax, jmin, jmax, x, y;
   float p, q, xrem, yrem;

   // TODO - get these parameters from the layer
   int Nx = NX;
   int Ny = NY;
   float dx = DX;
   float dy = DY;

   xrem = fmod(x0, dx); /* remainder of x / dx */
   yrem = fmod(y0, dy); /* remainder of y / dy */

   p = x0 - xrem; /* i, dx */
   q = y0 - yrem; /* j, dy */

   i1 = (int) (p / dx); /* scaling to receiver's grid */;
   j1 = (int) (q / dy);

   imin = -(rx - 1);
   jmin = -(ry - 1);
   imax = rx - 1;
   jmax = ry - 1;

   imax = (i1 + imax > Nx - 1) ? Nx - 1 - i1 : imax;
   imin = (i1 + imin < 0) ? -(i1) : imin;

   jmax = (j1 + jmax > Ny - 1) ? Ny - 1 - j1 : jmax;
   jmin = (j1 + jmin < 0) ? -(j1) : jmin;

   t = 0;
   for (i = imin; i <= imax; i++) /* x loop */
   {
      x = i1 + i;
      for (j = jmin; j <= jmax; j++) /* y loop */
      {
         y = j1 + j;

         k[t] = x + Nx * (y);
         // printf("loop=%ld\t\t k=%ld\t x=%ld\t y=%ld\t\n",t,k[t],x,y);
         t++;
      }
   }
   // printf("\nNumber of neighbors = t=%ld \n\n\n",t);
   return 0;
}
#endif

// TODO - move this to default PV_updateState()?
int PV_LGN_updateState(PVLayer * l)
{
   // TODO make this a parameter of the layer?
   int spiking = 0;
   // TODO get this somewhere
   float V_th = 1.0;

   pvdata_t * phi = l->phi[0];
   float * V   = l->V;
   pvdata_t * activity = l->activity->data;

   // TODO fix these to include

   if (spiking) {
      for (int k = 0; k < l->numNeurons; k++) {
#ifdef EXTEND_BORDER_INDEX
         int kPhi = kIndexExtended(k, l->loc.nx, l->loc.ny, l->numFeatures, l->loc.nPad);
#else
         int kPhi = k;
#endif
         activity[k] = (phi[kPhi] > V_th) ? 1.0 : 0.0;
         V[k] = (phi[kPhi] > V_th) ? 0.0 : V[k];
         phi[kPhi] = V[k]; // TODO -figure out if needs to be 0.0
      }
   }
   else {
      for (int k = 0; k < l->numNeurons; k++) {
#ifdef EXTEND_BORDER_INDEX
         int kPhi = kIndexExtended(k, l->loc.nx, l->loc.ny, l->numFeatures, l->loc.nPad);
#else
         int kPhi = k;
#endif
         activity[k] = phi[kPhi];
         V[k] = phi[kPhi];
         phi[kPhi] = 0.0;     // reset accumulation buffer
      }
   }

   return 0;
}

#ifdef __cplusplus
}
#endif
