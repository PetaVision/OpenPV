/*
 * PVLayer.c
 *
 *  Created on: Nov 18, 2008
 *      Author: Craig Rasmussen
 */

#include "PVLayer.h"
#include "../io/io.h"
#include "../include/default_params.h"
#include "../utils/pv_random.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

///////////////////////////////////////////////////////
// pvlayer C interface implementation
//

PVLayer * pvlayer_new(const PVLayerLoc loc, int xScale, int yScale, int numChannels)
{
   PVLayer * l = (PVLayer *) calloc(sizeof(PVLayer), sizeof(char));
   assert(l != NULL);
   pvlayer_init(l, loc, xScale, yScale, numChannels);
   return l;
}

/**
 * Initialize layer data.  This originally did not allocate layer data.  However,
 * now the size of the layer (nx,ny) is known at initialization time so allocation
 * has been moved here (functionality of initGlobal) and integrated with OpenCL
 * buffers.
 */
int pvlayer_init(PVLayer * l, PVLayerLoc loc, int xScale, int yScale, int numChannels)
{
   int k;
   const int nx = loc.nx;
   const int ny = loc.ny;
   const int nf = loc.nf;

   const int numNeurons  = nx * ny * nf;
   const int numExtended = (nx + 2*loc.nb) * (ny + 2*loc.nb) * nf;

   l->columnId = 0;

   l->layerId = -1; // the HyPerCol will set this
   l->numDelayLevels = 1; // HyPerConns will increase this as necessary by calling pre-synaptic layer's increaseDelayLevels

   l->loc = loc;
   l->numNeurons  = numNeurons;
   l->numExtended = numExtended;

   l->xScale = xScale;
   l->yScale = yScale;

   l->dx = powf(2.0f, (float) xScale);
   l->dy = powf(2.0f, (float) yScale);

   l->xOrigin = 0.5 + l->loc.kx0 * l->dx;
   l->yOrigin = 0.5 + l->loc.ky0 * l->dy;

   l->params = NULL;

   l->numActive = 0;
   l->activeFP  = NULL;

   l->activity = pvcube_new(&l->loc, numExtended);
   l->prevActivity = (float *) calloc(numExtended, sizeof(float));

   l->V = (pvdata_t *) calloc(numNeurons, sizeof(pvdata_t));
   assert(l->V != NULL);

   l->activeIndices = (unsigned int *) calloc(l->numNeurons, sizeof(unsigned int));
   assert(l->activeIndices != NULL);

   // initialize prevActivity (other buffers allocated in HyPerLayer::initialize_base() )
   //
   for (k = 0; k < numExtended; k++) {
      l->prevActivity[k] = -10*REFACTORY_PERIOD;  // allow neuron to fire at time t==0
   }

   return PV_SUCCESS;
}

int pvlayer_finalize(PVLayer * l)
{
   pvcube_delete(l->activity);

   if (l->activeFP != NULL) fclose(l->activeFP);

   free(l->prevActivity);
   free(l->activeIndices);
   free(l->V);
   free(l);

   return 0;
}

float pvlayer_getWeight(float x0, float x, float r, float sigma)
{
   float dx = x - x0;
   return expf(0.5 * dx * dx / (sigma * sigma));
}

int pvlayer_copyUpdate(PVLayer* l) {
   int k;
   pvdata_t * activity = l->activity->data;
   float* V = l->V;

   // copy from the V buffer to the activity buffer
   for (k = 0; k < l->numNeurons; k++) {
      activity[k] = V[k];
   }
   return 0;
}

///////////////////////////////////////////////////////
// pvpatch interface implementation
//

PVPatch * pvpatch_new(int nx, int ny, int nf)
{
   int sf = 1;
   int sx = nf;
   int sy = sx * nx;

   PVPatch * p = (PVPatch *) malloc(sizeof(PVPatch));
   assert(p != NULL);

   pvdata_t * data = NULL;

   pvpatch_init(p, nx, ny, nf, sx, sy, sf, data);

   return p;
}

int pvpatch_delete(PVPatch* p)
{
   free(p);
   return 0;
}

PVPatch * pvpatch_inplace_new(int nx, int ny, int nf)
{
   int sf = 1;
   int sx = nf;
   int sy = sx * nx;

   size_t dataSize = nx * ny * nf * sizeof(pvdata_t);
   PVPatch * p = (PVPatch *) calloc(sizeof(PVPatch) + dataSize, sizeof(char));
   assert(p != NULL);

   pvdata_t * data = (pvdata_t *) ((char*) p + sizeof(PVPatch));

   pvpatch_init(p, nx, ny, nf, sx, sy, sf, data);

   return p;
}

int pvpatch_inplace_delete(PVPatch* p)
{
   free(p);
   return 0;
}

#ifdef COMPRESS_PHI
void pvpatch_accumulate(int nk, float* restrict v, float a, float* restrict w,
                        float* restrict m)
{
   const float scale = 33.3;
   const float inv_scale = 1.0/scale;
   const float shift = 2.0;
   int k;

   for (k = 0; k < nk; k++) {
            v[k] = (((shift + scale*v[k]) + a*w[k]*m[k])
                  - shift) * inv_scale;
      // without mask
      //      v[k] = (((shift + scale*v[k]) + a*w[k])
      //                  - shift) * inv_scale;
   }
}
#else
int pvpatch_accumulate(int nk, float* RESTRICT v, float a, float* RESTRICT w)
{
   int k;
   int err = 0;
   for (k = 0; k < nk; k++) {
      v[k] = v[k] + a*w[k];
   }
   return err;
}
#endif

int pvpatch_accumulate_stochastic(int nk, float* RESTRICT v, float a, float* RESTRICT w)
{
   int k;
   long along = (long) (a*pv_random_max());
   int err = 0;
   for (k = 0; k < nk; k++) {
      v[k] = v[k] + (pv_random()<along)*w[k];
   }
   return err;
}


///////////////////////////////////////////////////////
// pvcube interface implementation
//

PVLayerCube * pvcube_init(PVLayerCube * cube, PVLayerLoc * loc, int numItems)
{
   cube->size = pvcube_size(numItems);
   cube->numItems = numItems;
   cube->loc = *loc;
   pvcube_setAddr(cube);
   return cube;
}

PVLayerCube * pvcube_new(PVLayerLoc * loc, int numItems)
{
   PVLayerCube * cube = (PVLayerCube*) calloc(pvcube_size(numItems), sizeof(char));
   assert(cube !=NULL);
   pvcube_init(cube, loc, numItems);
   return cube;
}

size_t pvcube_size(int numItems)
{
   size_t size = LAYER_CUBE_HEADER_SIZE;
   assert(size == EXPECTED_CUBE_HEADER_SIZE); // depends on PV_ARCH_64 setting
   return size + numItems*sizeof(float);
}

int pvcube_delete(PVLayerCube * cube)
{
   free(cube);
   return 0;
}

int pvcube_setAddr(PVLayerCube * cube)
{
   cube->data = (pvdata_t *) ((char*) cube + LAYER_CUBE_HEADER_SIZE);
   return 0;
}

#ifdef __cplusplus
}
#endif
