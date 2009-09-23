/*
 * PVLayer.c
 *
 *  Created on: Nov 18, 2008
 *      Author: rasmussn
 */

#include "PVLayer.h"
#include "../io/io.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

PVLayer* pvlayer_new(int xScale, int yScale,
                     int nx, int ny, int numFeatures, int nBorder)
{
   PVLayer* l = (PVLayer*) malloc(sizeof(PVLayer));
   pvlayer_init(l, xScale, yScale, nx, ny, numFeatures, nBorder);
   return l;
}

int pvlayer_init(PVLayer* l, int xScale, int yScale,
                 int nx, int ny, int numFeatures, int nBorder)
{
   l->layerId = -1; // the hypercolumn will set this
   l->numActive = 0;
   l->numFeatures = numFeatures;
   l->numDelayLevels = MAX_F_DELAY;

   l->loc.nxGlobal = nx;
   l->loc.nyGlobal = ny;

   l->loc.nx = nx;
   l->loc.ny = ny;

   l->loc.nPad   = nBorder;
   l->loc.nBands = 1;

   l->xScale = xScale;
   l->yScale = yScale;

   l->dx = powf(2.0f, (float) xScale);
   l->dy = powf(2.0f, (float) yScale);

   l->numNeurons = (int) l->loc.nx * (int) l->loc.ny * l->numFeatures;

   l->loc.kx0 = 0.0;
   l->loc.ky0 = 0.0;

   l->xOrigin = 0.5;
   l->yOrigin = 0.5;

   l->columnId = 0;

   l->writeIdx = 0; // which one currently writing to
   // (readers are delayed, relative to this)

   l->numParams = 0;
   l->params = NULL;

   l->initFunc = NULL;
   l->updateFunc = NULL;

   l->activity = NULL;

   // TODO - check for out of memory conditions
   l->activeIndices = (int *) calloc(l->numNeurons, sizeof(int));

   l->numPhis = NUM_CHANNELS;

   return 0;
}

/**
 * Finish initialization with global parameters (multiple hypercolumns)
 */
int pvlayer_initGlobal(PVLayer* l, int colId, int colRow, int colCol, int nRows, int nCols)
{
   int m;
   int ntotal = l->numNeurons;

   // extendNum includes regular size plus border
   // hypothesis: should not in include border because handled by border regions
#ifdef EXTEND_BORDER_INDEX
   size_t extendNum = (l->loc.nx + 2.0*l->loc.nPad) * (l->loc.ny + 2.0*l->loc.nPad);
   extendNum *= l->numFeatures;
#else
   size_t extendNum = l->loc.nx * l->loc.ny * l->numFeatures;
#endif

   l->columnId = colId;

////////////////
// Assume loc parameters are already set as HyPerCol comes up
//
//   l->loc.nx = l->loc.nx / nCols;
//   l->loc.ny = l->loc.ny / nRows;
//   l->loc.kx0 = l->loc.nx * colCol;
//   l->loc.ky0 = l->loc.ny * colRow;

   l->numNeurons = (int) l->loc.nx * (int) l->loc.ny * l->numFeatures;
   if (l->numNeurons * nCols * nRows != ntotal) {
      printf("[%d]: WARNING: pvlayer_initFinish: uneven layout of neurons (nx,ny) = (%d,%d)\n",
             colId, (int)l->loc.nx, (int)l->loc.ny);
   }

   l->xOrigin = 0.5 + l->loc.kx0 * l->dx;
   l->yOrigin = 0.5 + l->loc.ky0 * l->dy;

   l->activity = pvcube_new(&l->loc, l->numNeurons);

   // make a G (variable conductance) for each phi
   l->G   = (pvdata_t **) malloc(sizeof(pvdata_t *) * l->numPhis);
   l->phi = (pvdata_t **) malloc(sizeof(pvdata_t *) * l->numPhis);

   assert(l->G   != NULL);
   assert(l->phi != NULL);

   l->G[0]   = (pvdata_t *) calloc(extendNum*l->numPhis, sizeof(pvdata_t));
   l->phi[0] = (pvdata_t *) calloc(extendNum*l->numPhis, sizeof(pvdata_t));

   assert(l->G[0]   != NULL);
   assert(l->phi[0] != NULL);

   for (m = 1; m < l->numPhis; m++) {
      l->G[m]   = l->G[0]   + m * extendNum;
      l->phi[m] = l->phi[0] + m * extendNum;
   }

   l->V    = (pvdata_t *) calloc(l->numNeurons, sizeof(pvdata_t));
   l->Vth  = (pvdata_t *) calloc(l->numNeurons, sizeof(pvdata_t));

   assert(l->V   != NULL);
   assert(l->Vth != NULL);

   l->G_E  = l->G[PHI_EXC];
   l->G_I  = l->G[PHI_INH];
   l->G_IB = l->G[PHI_INHB];

   return 0;
}

int pvlayer_initFinish(PVLayer * l)
{  int err = 0;

   if (l->initFunc) {
      err = l->initFunc(l);
   }
   return err;
}

int pvlayer_finalize(PVLayer * l)
{
   free(l->G[0]);
   free(l->G);

   free(l->phi[0]);
   free(l->phi);

   pvcube_delete(l->activity);

   free(l->activeIndices);

   free(l->V);
   free(l->Vth);

   free(l->params);

   free(l);

   return 0;
}

#if 0
inline int pvlayer_getPos2(PVLayer * l, int idx, float * x, float * y)
{
   int kx = (idx / l->nfeatures) % l->nx;
   int ky = (idx / (l->nx * l->nfeatures)) % l->ny;
   *x = l->xOrigin + l->xMpiOrigin + kx*l->dx;
   *y = l->yOrigin + l->yMpiOrigin + ky*l->dy;

   return 0;
}
#endif //0
float pvlayer_getWeight(float x0, float x, float r, float sigma)
{
   float dx = x - x0;
   return expf(0.5 * dx * dx / (sigma * sigma));
}

int pvlayer_setParams(PVLayer * l, int numParams, size_t sizeParams, void * params)
{
   // check for existing parameters
   if (l->numParams != 0) {
      assert(l->numParams == numParams);
      // TODO - for now must assume sizeParams are the same, FIX THIS
   }
   else {
      l->numParams = numParams;
      l->params = (float *) malloc(sizeParams);
   }

   assert(l->params != NULL);
   memcpy(l->params, params, sizeParams);

   return 0;
}

int pvlayer_getParams(PVLayer * l, int * numParams, float ** params)
{
   // Give the caller our buffer
   *numParams = l->numParams;
   *params = l->params;
   return 0;
}

int pvlayer_setFuncs(PVLayer * l, void * initFunc, void * updateFunc)
{
   l->initFunc = (int(*)(PVLayer *l)) (initFunc);
   l->updateFunc = (int(*)(PVLayer *l)) (updateFunc);
   return 0;
}

// Default implementation -- output some stats and activity files
int pvlayer_outputState(PVLayer *l)
{
   char str[16];
   static int append = 0;
   int cid = l->columnId;

   const int nx = l->loc.nx;
   const int ny = l->loc.ny;
   const int nf = l->numFeatures;

   // Print avg, max/min, etc of f.
   // Probably DO want to use writeIdx, since that's what was updated in this timestep.
   sprintf(str, "[%d]:L%1.1d: f:", cid, l->layerId);
   printStats(l->activity->data, l->numNeurons, str);

   // Output spike events and V
   sprintf(str, "[%d]:f%1.1d", cid, l->layerId);
   pv_dump_sparse(str, append, l->activity->data, nx, ny, nf);
   sprintf(str, "[%d]:V%1.1d", cid, l->layerId);
   pv_dump(str, append, l->V, nx, ny, nf);

   // append to dump file after original open
   append = 1;

   return 0;
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

int pvpatch_update_plasticity_incr(int nk, float * RESTRICT p,
                                   float aPre, float decay, float ltpAmp)
{
   int k;
   for (k = 0; k < nk; k++) {
      p[k] = decay * p[k] + ltpAmp * aPre;
   }
   return 0;
}

int pvpatch_update_weights(int nk, float * RESTRICT w, float * RESTRICT m, float * RESTRICT p,
                           float aPre, float * RESTRICT aPost, float dWMax, float wMax)
{
   int k;
   for (k = 0; k < nk; k++) {
       w[k] += dWMax * (aPre * m[k] + aPost[k] * p[k]);
       w[k] = w[k] < 0    ? 0    : w[k];
       w[k] = w[k] > wMax ? wMax : w[k];
   }
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
//      if (w[k] > 0.0) {
//         printf("  w[%d] = %f %f %f %p\n", k, w[k], v[k], a, &v[k]);
//         err = -1;
//      }
   }
   return err;
}
#endif

#ifdef __cplusplus
}
#endif
