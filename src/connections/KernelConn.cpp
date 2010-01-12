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
   printf("KernelConn::KernelConn: running default constructor\n");
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

   for (int k = 0; k < numKernelPatches; k++) {
      kernel_patches[k] = pvpatch_inplace_new(nxPatch, nyPatch, nfPatch);
   }
   for (int k = 0; k < nPatches; k++) {
      patches[k] = pvpatch_new(nxPatch, nyPatch, nfPatch);
   }
   int xScaleFac = (post->clayer->xScale > pre->clayer->xScale) ? pow(2,
         post->clayer->xScale - pre->clayer->xScale) : 1;
   int yScaleFac = (post->clayer->yScale > pre->clayer->yScale) ? pow(2,
         post->clayer->yScale - pre->clayer->yScale) : 1;
   int nxPre = pre->clayer->loc.nx;
   int nyPre = pre->clayer->loc.ny;
   int nfPre = pre->clayer->numFeatures;
   for (int k = 0; k < nPatches; k++) {
      int kxPre = kxPos( k, nxPre, nyPre, nfPre);
      int kyPre = kyPos( k, nxPre, nyPre, nfPre);
      int kfPre = featureIndex( k, nxPre, nyPre, nfPre);
      kxPre = kxPre % xScaleFac;
      kyPre = kyPre % yScaleFac;
      int kprime = kIndex( kxPre,  kyPre,  kfPre,  nxPre,  nyPre,  nfPre);
      patches[k]->data = kernel_patches[kprime]->data;
   }
   return kernel_patches;
}

PVPatch ** KernelConn::createWeights(PVPatch ** patches, int nPatches, int nxPatch,
      int nyPatch, int nfPatch)
{
   // could create only a single patch with following call
   //   return createPatches(numAxonalArborLists, nxp, nyp, nfp);

   assert(numAxonalArborLists == 1);

   // TODO IMPORTANT ################# free memory in patches as well
   // GTK: call delete weights?
   if (patches != NULL) {
      free(patches);
   }

   patches = (PVPatch**) calloc(sizeof(PVPatch*), nPatches);
   assert(patches != NULL);

   // TODO - allocate space for them all at once (inplace)
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
   HyPerConn::initializeWeights(kernelPatches, numDataPatches(arbor), filename);
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
   int numKernelPatches = ((int) this->nfp) * xScaleFac * yScaleFac;
   return numKernelPatches;
}

// k is ignored, writes all weights
int KernelConn::writeWeights(float time, bool last)
{
   const int arbor = 0;
   const int numPatches = numDataPatches(arbor);
   return HyPerConn::writeWeights(kernelPatches, numPatches, NULL, time, last);

}

} // namespace PV

