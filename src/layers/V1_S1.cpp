/*
 * V1_S1.cpp
 *
 *  Created on: Aug 4, 2008
 *      Author: dcoates
 */

#include "V1_S1.hpp"
#include "../connections/PVConnection.h"
#include "../io/io.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

namespace PV {

V1_S1::V1_S1(const char * name, HyPerCol * hc)
{
   PVParams * params = hc->parameters();
   no = params->value(name, "no");

   // base constructor stuff
   setParent(hc);
   init(name, TypeGeneric);
   hc->addLayer(this);

   clayer->Vth[0] = 0.6;

   subWeights = (pvdata_t *) malloc(no * clayer->numFeatures * sizeof(pvdata_t));

   const int nf = clayer->numFeatures;
   for (int k = 0; k < no*nf; k++) {
      subWeights[k] = 0.0;
   }


#define VERTICAL
#ifdef VERTICAL
   // temporary vertical orientation
   subWeights[2]  = 0.5;
   subWeights[8]  = 0.5;
   subWeights[10] = 1.0;
   subWeights[1+16] = 0.5;
   subWeights[4+16] = 0.5; // I think this is 4+16
   subWeights[5+16] = 1.0;
   subWeights[2+32]  = 0.5;
   subWeights[8+32]  = 0.5;
   subWeights[10+32] = 1.0;
   subWeights[1+48] = 0.5;
   subWeights[4+48] = 0.5; // I think this is 4+48
   subWeights[5+48] = 1.0;
#endif

#ifdef HORIZONTAL
   // temporary horizontal orientation
   subWeights[4]  = 0.5;
   subWeights[8]  = 0.5;
   subWeights[12] = 1.0;
   subWeights[4+16]  = 0.5;
   subWeights[8+16]  = 0.5;
   subWeights[12+16] = 1.0;
   subWeights[1+32]  = 0.5;
   subWeights[2+32]  = 0.5;
   subWeights[3+32] = 1.0;
   subWeights[1+48] = 0.5;
   subWeights[2+48] = 0.5;
   subWeights[3+48] = 1.0;
#endif

   float sum = 0.0;
   for (int k = 0; k < nf; k++) {
      sum += subWeights[k];
   }
   for (int k = 0; k < nf; k++) {
      subWeights[k] = subWeights[k]/sum;
   }
}

int V1_S1::columnWillAddLayer(InterColComm * comm, int layerId)
{
   setLayerId(layerId);

   // complete initialization now that we have a parent and a communicator
   // WARNING - this must be done before addPublisher is called
   int id = parent->columnId();
   initGlobal(id, comm->commRow(id), comm->commColumn(id),
                  comm->numCommRows(), comm->numCommColumns());

   // shrink activity to represent reduced feature count after subunits
   clayer->activity->numItems = no * clayer->activity->loc.nx * clayer->activity->loc.ny;
   clayer->activity->size = clayer->activity->numItems * sizeof(pvdata_t);

   const float nx = clayer->loc.nx;
   const float ny = clayer->loc.ny;
   // calculate maximum size of a border cube
   // const int maxBorderSize = pvcube_size(nx*ny*no);

// TODO - calculate the maxBorderSize correctly
   comm->addPublisher(this, clayer->activity->size, clayer->activity->size, MAX_F_DELAY);

   return 0;
}

int V1_S1::updateState(float time, float dt)
{
   float data[4*16*16];

   static int hasData = 0;

   pvdata_t * phi = clayer->phi[0];
   const float nx = clayer->loc.nx;
   const float ny = clayer->loc.ny;
   const int   nf = clayer->numFeatures;

   pv_debug_info("[%d]: V1_S1::updateState:", clayer->columnId);

   // sum over subunits with weights to get membrane potential

   for (int f = 0; f < no; f++) {
      for (int kx = 0; kx < nx; kx++) {
         for (int ky = 0; ky < ny; ky++) {
            float tmp = 0.0;
            for (int kf = 0; kf < nf; kf++) {
               int k = kIndex(kx, ky, kf, nx, ny, nf);
               int kPhi = kIndexExtended(k, nx, ny, nf, clayer->numBorder);
               if (phi[kPhi] > 0.0) {
                  tmp += subWeights[kf]*phi[kPhi];
//                  printf("subWeight[%d] = %f %f\n", kf, subWeights[kf], phi[kPhi]);
               }
            }
            int k  = kIndex(kx, ky, f, nx, ny, 4);
            clayer->V[k] = tmp;
            if (clayer->V[k] > clayer->Vth[0]) {
               clayer->activity->data[k] = tmp;
               data[k] = tmp;
               printf("activity[%d] = %f \n", k, tmp);
            }
            else {
               clayer->activity->data[k] = 0.0;
            }
         }
      }
   }

   for (int k = 0; k < clayer->numNeurons; k++) {
      int kPhi = kIndexExtended(k, clayer->loc.nx, clayer->loc.ny, clayer->numFeatures,
                                clayer->numBorder);
      phi[kPhi] = 0.0;     // reset accumulation buffer
   }

   if (hasData) {
      clayer->activity->numItems = nx*ny*4;
      clayer->activity->size     = nx*ny*4 * sizeof(float);
      pv_tiff_write_cube("data.tif", clayer->activity, nx, ny, 4);
   }
   hasData = 1;

   return 0;
}

} // namespace PV
