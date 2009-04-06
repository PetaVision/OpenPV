/*
 * weight_cache.h
 *
 *  Created on: Aug 27, 2008
 *      Author: dcoates
 */

#ifndef WEIGHT_CACHE_H_
#define WEIGHT_CACHE_H_

#include "PVConnection.h"

#ifdef __cplusplus
extern "C" {
#endif

int PV_weightCache_init(PVConnection *con);
int PV_weightCache_finalize(PVConnection *con);
int PV_weightCache_getNumPreKernels(PVConnection *con);
int PV_weightCache_getNumPostKernels(PVConnection *con);

#define SENTINEL -1
#define MIN_CACHE_WEIGHT 0.05

// get returns "1" if successfully got the weight from  the cache
static inline int posToIdx(PVConnection *con, float *prePos, float *postPos)
{
   float dx, dy, d2;
   int idx;

   dx = (prePos[DIMX] - postPos[DIMX]);
   dy = (prePos[DIMY] - postPos[DIMY]);
   dx = fabs(dx) > NX / 2 ? -(dx / fabs(dx)) * (NX - fabs(dx)) : dx; // PBCs
   dy = fabs(dy) > NY / 2 ? -(dy / fabs(dy)) * (NY - fabs(dy)) : dy;

   d2 = dx * dx + dy * dy;
   if (d2 > con->r2) return -1;

   // Translate the range from -NX/2->NX/2 to 0->NX
   dx += NX / 2;
   dy += NY / 2;

   // Commented out: now we keep everything in the cache
   //if ((dx >= PV_WEIGHT_CACHE_MAX_R) || (dy >= PV_WEIGHT_CACHE_MAX_R))
   //return -2;

   idx = (int) (dy * con->yStride + dx * con->xStride + postPos[DIMO]);

   return (int) idx;
}

// get position of an indexed neuron relative to input position
static inline int idxToPos(PVConnection *con, int idx, float *ref, float *pos)
{
   float dx, dy, posX, posY;

   pos[DIMO] = idx % con->post->numFeatures;
   idx -= (int) pos[DIMO];

   dx = (float) idx / (con->xStride);
   dx = fmod(dx, NX);
   idx -= (int) (dx * con->xStride);

   dy = (float) idx / (con->yStride);

   dx -= NX / 2;
   dy -= NY / 2;

   posX = ref[DIMX] - dx;
   posY = ref[DIMY] - dy;

   if (posX < 0) posX = NX + posX;
   if (posX >= NX) posX = posX - NX;

   if (posY < 0) posY = NY + posY;
   if (posY >= NY) posY = posY - NY;

   pos[DIMX] = posX;
   pos[DIMY] = posY;

   return 0;
}

static inline int PV_weightCache_get(PVConnection *con, float *prePos,
      int preKernelIndex, float *postPos, float *weight)
{
   int idx;
   *weight = 0.0;

   if ((idx = posToIdx(con, prePos, postPos)) < 0) return 0;

   if (con->weights[preKernelIndex][idx] != SENTINEL) {
      *weight = con->weights[preKernelIndex][idx];
      return 1;
   }

   return 0;
}

static inline int PV_weightCache_getPostNeurons(PVConnection *con, float* prePos,
      int preKernelIndex, int* num)
{
   *num = con->numPostSynapses[preKernelIndex];
   return 0;
}

static inline int PV_weightCache_getPostByIndex(PVConnection *con, float* prePos,
      int preKernelIndex, int postIndex, float* weight, float *pos, int* postKernelIndex)
{
   *weight
         = con->weights[preKernelIndex][con->postCacheIndices[preKernelIndex][postIndex]];
   idxToPos(con, con->postCacheIndices[preKernelIndex][postIndex], prePos, pos);
   //*postKernelIndex = pos[DIMO];
   *postKernelIndex
         = con->postKernels[preKernelIndex][con->postCacheIndices[preKernelIndex][postIndex]];
   return 0;
}

// Determine which 'kernel' the presynaptic neuron corresponds to.
// If the pre layer has <= density of the postsynaptic layer, the presynaptic
// feature uniquely identifies the kernel, due to positional invariance.
// If there are more neurons in the presynaptic layer, need to store a kernel
// for all possible 'sublattices,' with each orientation at each fractional location.
static inline int PV_weightCache_getPreKernelIndex(PVConnection *con, float *prePos,
      int doPosTranslation)
{
   int kernelIndex = (int) prePos[DIMO];
   if (con->pre->loc.dx < con->post->loc.dx) {
      // If pre is more dense than post, need to do more work...

      // Calc how many presynaptic kernels
      int densityRatio = (int) (con->post->loc.dx / con->pre->loc.dx);
      float pixelmod = con->post->loc.dx;

      // Get the x,y index into the kernel array
      int xval = (int) (fmod(prePos[DIMX], pixelmod) / con->pre->loc.dx - 0.5);
      int yval = (int) (fmod(prePos[DIMY], pixelmod) / con->pre->loc.dy - 0.5);
      // xval and yval should be integers now
      kernelIndex += (yval * densityRatio + xval) * con->pre->numFeatures;

      // Now translate into the post-synaptic cell coordinate system
      // Move this into the center of the lattice, on top of a post cell.
      if (doPosTranslation) {
         // First term moves to 0,0 in this region.
         // Second term moves to center of post cell
         prePos[DIMY] -= (yval + 0.5) * con->pre->loc.dy - 0.5 * con->post->loc.dy;
         prePos[DIMX] -= (xval + 0.5) * con->pre->loc.dx - 0.5 * con->post->loc.dx;
      }
   }
   return kernelIndex;
}

// Like for pre. Need to get the position to generate indices for
// normalization.
static inline int PV_weightCache_getPostKernelIndex(PVConnection *con, float *pos,
      int doPosTranslation)
{
   int kernelIndex = (int) pos[DIMO];
   if (con->pre->loc.dx > con->post->loc.dx) {
      // Calc how many presynaptic kernels
      int densityRatio = (int) (con->pre->loc.dx / con->post->loc.dx);
      float pixelmod = con->pre->loc.dx;

      // Get the x,y index into the kernel array
      int xval = (int) (fmod(pos[DIMX], pixelmod) / con->post->loc.dx - 0.5);
      int yval = (int) (fmod(pos[DIMY], pixelmod) / con->post->loc.dy - 0.5);
      // xval and yval should be integers now
      kernelIndex += (yval * densityRatio + xval) * con->post->numFeatures;

      if (doPosTranslation) {
         // First term moves to 0,0 in this region.
         // Second term moves to center of post cell
         pos[DIMY] -= (yval + 0.5) * con->post->loc.dy - 0.5 * con->pre->loc.dy;
         pos[DIMX] -= (xval + 0.5) * con->post->loc.dx - 0.5 * con->pre->loc.dx;
      }
   }
   return kernelIndex;
}

// For a given kernel index, return the position near 0,0 that is representative
// of this location. 'Real' neuron position shouldn't matter much, just need a
// good example for normalization calculation.
static inline int PV_weightCache_getKernelPos(PVConnection *con, int kernelIndex,
      float *pos)
{
   int densityRatio = (int) (con->post->loc.dx / con->pre->loc.dx);
   pos[DIMO] = (float) (kernelIndex % con->pre->numFeatures);
   if (con->pre->loc.dx < con->post->loc.dx) {
      int xval = kernelIndex / con->pre->numFeatures % densityRatio;
      pos[DIMX] = (xval + 0.5) * con->pre->loc.dx;
      int yval = kernelIndex / (con->pre->numFeatures * densityRatio);
      pos[DIMY] = (yval + 0.5) * con->pre->loc.dy;

   }
   else {
      pos[DIMX] = 0.5;
      pos[DIMY] = 0.5;
   }

   return kernelIndex;
}

static inline int PV_weightCache_getPostKernelPos(PVConnection *con, int kernelIndex,
      float *pos)
{
   int densityRatio = (int) (con->pre->loc.dx / con->post->loc.dx);
   pos[DIMO] = (float) (kernelIndex % con->post->numFeatures);
   if (con->post->loc.dx < con->pre->loc.dx) {
      int xval = kernelIndex / con->post->numFeatures % densityRatio;
      pos[DIMX] = (xval + 0.5) * con->post->loc.dx;
      int yval = kernelIndex / (con->post->numFeatures * densityRatio);
      pos[DIMY] = (yval + 0.5) * con->post->loc.dy;

   }
   else {
      pos[DIMX] = 0.5;
      pos[DIMY] = 0.5;
   }

   return kernelIndex;
}

static inline int PV_weightCache_set(PVConnection *con, float *prePos,
      int preKernelIndex, float *postPos, float weight)
{
   int idx;

   if ((idx = posToIdx(con, prePos, postPos)) < 0)
   // Out of range: doesn't belong in cache.
   return 0;

   if (weight < MIN_CACHE_WEIGHT) return 0;

   con->postCacheIndices[preKernelIndex][con->numPostSynapses[preKernelIndex]] = idx;
   con->numPostSynapses[preKernelIndex]++;

   con->weights[preKernelIndex][idx] = weight;
   con->postKernels[preKernelIndex][idx] = PV_weightCache_getPostKernelIndex(con,
         postPos, 0);
   return 1;
}

#ifdef __cplusplus
}
#endif

#endif /* WEIGHT_CACHE_H_ */
