/*
 * PVLayer.c
 *
 *  Created on: Nov 18, 2008
 *      Author: Craig Rasmussen
 */

#include "PVLayer.h"
#include "../io/io.h"
#include "../include/default_params.h"
#include "../utils/cl_random.h"
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

//PVLayer * pvlayer_new(const PVLayerLoc loc, int xScale, int yScale)
//{
//   PVLayer * l = (PVLayer *) calloc(sizeof(PVLayer), sizeof(char));
//   assert(l != NULL);
//   pvlayer_init(l, loc, xScale, yScale);
//   return l;
//}

///**
// * Initialize layer data.  This originally did not allocate layer data.  However,
// * now the size of the layer (nx,ny) is known at initialization time so allocation
// * has been moved here (functionality of initGlobal) and integrated with OpenCL
// * buffers.
// */
//int pvlayer_init(PVLayer * l, PVLayerLoc loc, int xScale, int yScale)
//{
//   int k;
//   const int nx = loc.nx;
//   const int ny = loc.ny;
//   const int nf = loc.nf;
//
//   const int numNeurons  = nx * ny * nf;
//   const int numExtended = (nx + 2*loc.nb) * (ny + 2*loc.nb) * nf;
//
//   // l->numDelayLevels = 1; // HyPerConns will increase this as necessary by calling pre-synaptic layer's increaseDelayLevels
//   // numDelayLevels now a HyPerLayer member variable
//
//   l->loc = loc;
//   l->numNeurons  = numNeurons;
//   l->numExtended = numExtended;
//
//   l->xScale = xScale;
//   l->yScale = yScale;
//
//   l->dx = powf(2.0f, (float) xScale);
//   l->dy = powf(2.0f, (float) yScale);
//
//   l->xOrigin = 0.5 + l->loc.kx0 * l->dx;
//   l->yOrigin = 0.5 + l->loc.ky0 * l->dy;
//
//   l->params = NULL;
//
//   l->numActive = 0;
//   l->activeFP  = NULL;
//
//   l->activity = pvcube_new(&l->loc, numExtended);
//   l->prevActivity = (float *) calloc(numExtended, sizeof(float));
//
//   l->V = (pvdata_t *) calloc(numNeurons, sizeof(pvdata_t));
//   assert(l->V != NULL);
//
//   l->activeIndices = (unsigned int *) calloc(l->numNeurons, sizeof(unsigned int));
//   assert(l->activeIndices != NULL);
//
//   // initialize prevActivity (other buffers allocated in HyPerLayer::initialize_base() )
//   //
//   for (k = 0; k < numExtended; k++) {
//      l->prevActivity[k] = -10*REFRACTORY_PERIOD;  // allow neuron to fire at time t==0
//   }
//
//   return PV_SUCCESS;
//}

//int pvlayer_finalize(PVLayer * l)
//{
//   pvcube_delete(l->activity);
//
//   if (l->activeFP != NULL) {
//      fclose(l->activeFP->fp); // Can't call fileio.cpp routines from a .c file, so have to replicate PV_fclose
//      free(l->activeFP->name);
//      free(l->activeFP);
//      l->activeFP = NULL;
//   }
//
//   free(l->prevActivity);
//   free(l->activeIndices);
//   free(l->V);
//   free(l);
//
//   return 0;
//}

//float pvlayer_getWeight(float x0, float x, float r, float sigma)
//{
//   float dx = x - x0;
//   return expf(0.5 * dx * dx / (sigma * sigma));
//}

//int pvlayer_copyUpdate(PVLayer* l) {
//   int k;
//   pvdata_t * activity = l->activity->data;
//   float* V = l->V;
//
//   // copy from the V buffer to the activity buffer
//   for (k = 0; k < l->numNeurons; k++) {
//      activity[k] = V[k];
//   }
//   return 0;
//}

///////////////////////////////////////////////////////
// pvpatch interface implementation
//

// PVPatch * pvpatch_new(int nx, int ny, int nf)
PVPatch * pvpatch_new(int nx, int ny)
{
   // int sf = 1;
   // int sx = nf;
   // int sy = sx * nx;

   PVPatch * p = (PVPatch *) malloc(sizeof(PVPatch));
   assert(p != NULL);

   // pvdata_t * data = NULL;

   pvpatch_init(p, nx, ny); // pvpatch_init(p, nx, ny, nf, sx, sy, sf, data);

   return p;
}

int pvpatch_delete(PVPatch* p)
{
   free(p);
   return 0;
}

#ifdef OBSOLETE // Marked obsolete Feb. 27, 2012.  New refactoring for weights means that patches are never created with the data adjacent to the patch structure.
PVPatch * pvpatch_inplace_new(int nx, int ny, int nf)
{
   // int sf = 1;
   // int sx = nf;
   // int sy = sx * nx;

   // size_t dataSize = nx * ny * nf * sizeof(pvdata_t);
   PVPatch * p = (PVPatch *) calloc(sizeof(PVPatch));
   // PVPatch * p = (PVPatch *) calloc(sizeof(PVPatch) + dataSize, sizeof(char));
   assert(p != NULL);

   // pvdata_t * data = (pvdata_t *) ((char*) p + sizeof(PVPatch));

   pvpatch_init(p, nx, ny); // pvpatch_init(p, nx, ny, nf, sx, sy, sf, data);

   return p;
}
#endif // OBSOLETE

#ifdef OBSOLETE // Marked obsolete Feb. 27, 2012.  Duplicates createPatches
pvdata_t * pvpatches_new(PVPatch ** patches, int nx, int ny, int nf, int nPatches) {
   //int sf = 1;
   int sx = nf;
   int sy = sx * nx;
   int sp = sy*ny;

   size_t patchSize = nx * ny * nf * sizeof(pvdata_t);
   size_t dataSize = nPatches * patchSize;
   pvdata_t * dataPatches = (pvdata_t *) calloc(dataSize, sizeof(char));
   assert(dataPatches != NULL);

   //PVPatch ** patches;
   for (int k = 0; k < nPatches; k++) {
      patches[k] = pvpatch_new(nx, ny); // patches[k] = pvpatch_inplace_new_sepdata(nx, ny, nf, &dataPatches[k*sp]);
   }

   return dataPatches;
}
#endif // OBSOLETE


#ifdef OBSOLETE // Marked obsolete Feb. 27, 2012.  New refactoring for weights means that patches are never created with the data adjacent to the patch structure.
PVPatch * pvpatch_inplace_new_sepdata(int nx, int ny, int nf, pvdata_t * data)
{
   // int sf = 1;
   // int sx = nf;
   // int sy = sx * nx;

   //size_t dataSize = nx * ny * nf * sizeof(pvdata_t);
   PVPatch * p = (PVPatch *) malloc(sizeof(PVPatch)); //calloc(sizeof(PVPatch) + sizeof(pvdata_t*), sizeof(char));
   assert(p != NULL);

   //pvdata_t * data = (pvdata_t *) ((char*) p + sizeof(PVPatch));

   pvpatch_init(p, nx, ny); // pvpatch_init(p, nx, ny, nf, sx, sy, sf, data);

   return p;
}
#endif // OBSOLETE

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
int pvpatch_accumulate(int nk, float* RESTRICT v, float a, float* RESTRICT w, void * auxPtr)
{
   int k;
   int err = 0;
   for (k = 0; k < nk; k++) {
      v[k] = v[k] + a*w[k];
   }
   return err;
}
#endif

int pvpatch_accumulate_from_post(int nk, float * RESTRICT v, float * RESTRICT a, float * RESTRICT w, float dt_factor, void * auxPtr) {
   int status = 0;
   int k;
   float dv = 0.0f;
   for (k = 0; k < nk; k++) {
      dv = dv + a[k]*w[k];
   }
   *v = *v + dt_factor*dv;
   return status;
}

int pvpatch_accumulate2(int nk, float* RESTRICT v, float a, float* RESTRICT w, float* RESTRICT m)
{
   int k;
   int err = 0;
   for (k = 0; k < nk; k++) {
      v[k] = v[k] + a*w[k]*m[k];
   }
   return err;
}

#ifdef OBSOLETE // Marked obsolete Aug 21, 2013.  Use cl_random instead of pv_random
int pvpatch_accumulate_stochastic(int nk, float* RESTRICT v, float a, float* RESTRICT w, void * auxPtr)
{
   int k;
   long along = (long) (a*pv_random_max());
   int err = 0;
   for (k = 0; k < nk; k++) {
      v[k] = v[k] + (pv_random()<along)*w[k];
   }
   return err;
}

int pvpatch_accumulate_stochastic_from_post(int nk, float * RESTRICT v, float * RESTRICT a, float * RESTRICT w, float dt_factor, void * auxPtr) {
   int status = 0;
   int k;
   float dv = 0.0f;
   for (k = 0; k < nk; k++) {
      long along = (long) (a[k]*pv_random_max());
      dv = pv_random()<along ? dv + a[k]*w[k] : 0.0f;
   }
   *v = *v + dt_factor*dv;
   return status;
}
#else
int pvpatch_accumulate_stochastic(int nk, float* RESTRICT v, float a, float* RESTRICT w, void * auxPtr)
{
   uint4 * rngArray = (uint4 *) auxPtr;
   long along = (long) (a*cl_random_max());
   int err = 0;
   int k;
   for (k = 0; k < nk; k++) {
      uint4 * rng = &rngArray[k];
      *rng = cl_random_get(*rng);
      v[k] = v[k] + (rng->s0 < along)*w[k];
   }
   return err;
}

int pvpatch_accumulate_stochastic_from_post(int nk, float * RESTRICT v, float * RESTRICT a, float * RESTRICT w, float dt_factor, void * auxPtr) {
   int status = 0;
   uint4 * rng = (uint4 *) auxPtr;
   int k;
   float dv = 0.0f;
   for (k = 0; k < nk; k++) {
      *rng = cl_random_get(*rng);
      double p = (double) rng->s0/cl_random_max(); // 0.0 < p < 1.0
      dv += (p<a[k])*w[k];
   }
   *v = *v + dt_factor*dv;
   return status;
}
#endif // OBSOLETE

#ifdef OBSOLETE // Marked obsolete Aug 19, 2013.  Nobody calls pvpatch_max and whatever WTACompressedLayer was, it's not in the code now.
// Used by WTACompressedLayer
int pvpatch_max(int nk, float * RESTRICT v, float a, float * RESTRICT w, int feature, int * RESTRICT maxloc) {
   int k;
   int err = 0;
   for (k = 0; k < nk; k++) {
      float prod = a*w[k];
      if (prod!=0 && v[k] == prod) {
         err = 1;
      }
      if (v[k] < prod) {
         v[k] = prod;
         maxloc[k] = feature;
      }
   }
   return err;
}
#endif

int pvpatch_max_pooling(int nk, float* RESTRICT v, float a, float* RESTRICT w, void * auxPtr)
{
  int k;
  int err = 0;
  for (k = 0; k < nk; k++) {
     v[k] = v[k] > a ? v[k] : a;
  }
  return err;
}

int pvpatch_max_pooling_from_post(int nk, float * RESTRICT v, float * RESTRICT a, float * RESTRICT w, float dt_factor, void * auxPtr) {
   int status = 0;
   int k;
   float vmax = *v;
   for (k = 0; k < nk; k++) {
      vmax = vmax > a[k] ? vmax : a[k];
   }
   *v = vmax;
   return status;
}
  

///////////////////////////////////////////////////////
// pvcube interface implementation
//

PVLayerCube * pvcube_init(PVLayerCube * cube, const PVLayerLoc * loc, int numItems)
{
   cube->size = pvcube_size(numItems);
   cube->numItems = numItems;
   cube->loc = *loc;
   pvcube_setAddr(cube);
   return cube;
}

PVLayerCube * pvcube_new(const PVLayerLoc * loc, int numItems)
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
