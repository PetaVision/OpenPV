/*
 * KernelConn.cpp
 *
 *  Created on: Aug 6, 2009
 *      Author: gkenyon
 */

#include "KernelConn.hpp"
#include <assert.h>
#include "../io/io.h"

namespace PV {

KernelConn::KernelConn()
{
   initialize_base();
}

KernelConn::KernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, int channel)
{
   initialize_base();
   initialize(name, hc, pre, post, channel);
}

KernelConn::KernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post)
{
   initialize_base();
   initialize(name, hc, pre, post, channel, NULL); // use default channel
}

// provide filename or set to NULL
KernelConn::KernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, int channel, const char * filename)
{
   initialize_base();
   initialize(name, hc, pre, post, channel, filename);
}

int KernelConn::initialize_base()
{
   kernelPatches = NULL;
   return HyPerConn::initialize_base();
}

PVPatch ** KernelConn::allocWeights(PVPatch ** patches, int nPatches, int nxPatch,
      int nyPatch, int nfPatch)
{
   const int arbor = 0;
   int numKernelPatches = numDataPatches(arbor);

   PVPatch ** kernel_patches = (PVPatch**) calloc(sizeof(PVPatch*), numKernelPatches);
   assert(kernel_patches != NULL);

   for (int kernelIndex = 0; kernelIndex < numKernelPatches; kernelIndex++) {
      kernel_patches[kernelIndex] = pvpatch_inplace_new(nxPatch, nyPatch, nfPatch);
      assert(kernel_patches[kernelIndex] != NULL );
   }
   for (int patchIndex = 0; patchIndex < nPatches; patchIndex++) {
      patches[patchIndex] = pvpatch_new(nxPatch, nyPatch, nfPatch);
   }
   for (int patchIndex = 0; patchIndex < nPatches; patchIndex++) {
      int kernelIndex = patchIndexToKernelIndex(patchIndex);
      patches[patchIndex]->data = kernel_patches[kernelIndex]->data;
   }
   return kernel_patches;
}

/*TODO  createWeights currently breaks in this subclass if called more than once,
 * fix interface by adding extra dataPatches argument to overloaded method
 * so asserts are unnecessary
 */
PVPatch ** KernelConn::createWeights(PVPatch ** patches, int nPatches, int nxPatch,
      int nyPatch, int nfPatch)
{
   assert(numAxonalArborLists == 1);

   assert(patches == NULL);

   patches = (PVPatch**) calloc(sizeof(PVPatch*), nPatches);
   assert(patches != NULL);

   assert(kernelPatches == NULL);
   kernelPatches = allocWeights(patches, nPatches, nxPatch, nyPatch, nfPatch);

   return patches;
}

int KernelConn::deleteWeights()
{
   const int arbor = 0;

   for (int k = 0; k < numDataPatches(arbor); k++) {
      pvpatch_inplace_delete(kernelPatches[k]);
   }
   free(kernelPatches);

   return HyPerConn::deleteWeights();
}

PVPatch ** KernelConn::initializeWeights(PVPatch ** patches, int numPatches,
      const char * filename)
{
   int arbor = 0;
   int numKernelPatches = numDataPatches(arbor);
   HyPerConn::initializeWeights(kernelPatches, numKernelPatches, filename);
   return wPatches[arbor];
}

PVPatch ** KernelConn::readWeights(PVPatch ** patches, int numPatches,
      const char * filename)
{
   HyPerConn::readWeights(patches, numPatches, filename);
   return HyPerConn::normalizeWeights(patches, numPatches);
}

int KernelConn::numDataPatches(int arbor)
{
   int xScaleFac = (post->clayer->xScale > pre->clayer->xScale) ? pow(2,
         post->clayer->xScale - pre->clayer->xScale) : 1;
   int yScaleFac = (post->clayer->yScale > pre->clayer->yScale) ? pow(2,
         post->clayer->yScale - pre->clayer->yScale) : 1;
   int numKernelPatches = pre->clayer->numFeatures * xScaleFac * yScaleFac;
   return numKernelPatches;
}

int KernelConn::writeWeights(float time, bool last)
{
   const int arbor = 0;
   const int numPatches = numDataPatches(arbor);
   return HyPerConn::writeWeights(kernelPatches, numPatches, NULL, time, last);
}


int KernelConn::kernelIndexToPatchIndex(int kernelIndex){
   int patchIndex;
   int xScaleFac = (post->clayer->xScale > pre->clayer->xScale) ? pow(2,
         post->clayer->xScale - pre->clayer->xScale) : 1;
   int yScaleFac = (post->clayer->yScale > pre->clayer->yScale) ? pow(2,
         post->clayer->yScale - pre->clayer->yScale) : 1;
   int nfPre = pre->clayer->numFeatures;
   int kxPre = kxPos( kernelIndex, xScaleFac, yScaleFac, nfPre);
   int kyPre = kyPos( kernelIndex, xScaleFac, yScaleFac, nfPre);
   int kfPre = featureIndex( kernelIndex, xScaleFac, yScaleFac, nfPre);
   int nxPre = pre->clayer->loc.nx;
   int nyPre = pre->clayer->loc.ny;
   patchIndex = kIndex( kxPre,  kyPre,  kfPre,  nxPre,  nyPre,  nfPre);
   return patchIndex;
}

// many to one mapping from weight patches to kernels
int KernelConn::patchIndexToKernelIndex(int patchIndex){
   int kernelIndex;
   int nxPre = pre->clayer->loc.nx;
   int nyPre = pre->clayer->loc.ny;
   int nfPre = pre->clayer->numFeatures;
   int kxPre = kxPos( patchIndex, nxPre, nyPre, nfPre);
   int kyPre = kyPos( patchIndex, nxPre, nyPre, nfPre);
   int kfPre = featureIndex( patchIndex, nxPre, nyPre, nfPre);
   int xScaleFac = (post->clayer->xScale > pre->clayer->xScale) ? pow(2,
         post->clayer->xScale - pre->clayer->xScale) : 1;
   int yScaleFac = (post->clayer->yScale > pre->clayer->yScale) ? pow(2,
         post->clayer->yScale - pre->clayer->yScale) : 1;
   kxPre = kxPre % xScaleFac;
   kyPre = kyPre % yScaleFac;
   kernelIndex = kIndex( kxPre,  kyPre,  kfPre,  xScaleFac,  yScaleFac,  nfPre);
   return kernelIndex;
}


} // namespace PV

