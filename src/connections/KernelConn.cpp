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



int KernelConn::initialize_base(){
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
   for (int k = 0; k < nPatches; k++) {
      patches[k]->data = kernel_patches[k % numKernelPatches]->data;
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

int KernelConn::numDataPatches(int arbor)
{
   int numKernelPatches = (int) this->nfp;
   return numKernelPatches;
}


// k is ignored, writes all weights
int KernelConn::writeWeights(const char * filename, float time)
{
   int status = 0;
   char name[PV_PATH_MAX];

   if (filename == NULL) {
      if (time == FINAL_TIME) {
         snprintf(name, PV_PATH_MAX-1, "w%d_last", getConnectionId());
      }
      else {
         snprintf(name, PV_PATH_MAX-1, "w%d", getConnectionId());
      }
   }
   else {
      snprintf(name, PV_PATH_MAX-1, "%s", filename);
   }

   const int arbor = 0;
   const int append = 0;
   int numPatches = numDataPatches(arbor);
   status = pv_write_patches(name, append, (int) nxp, (int) nyp, (int) nfp,
                             minWeight(), maxWeight(), numPatches, kernelPatches);
   assert(status == 0);

   return status;

}

} // namespace PV

