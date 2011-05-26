/* CloneKernelConn.cpp
 *
 * Created on: May 23, 2011
 *     Author: peteschultz
 */

#include "CloneKernelConn.hpp"

namespace PV {

CloneKernelConn::CloneKernelConn() {
   initialize_base();
}

CloneKernelConn::CloneKernelConn(const char * name, HyPerCol * hc,
      HyPerLayer * pre, HyPerLayer * post, ChannelType channel,
      KernelConn * originalConn) {
   initialize_base();
   initialize(name, hc, pre, post, channel, originalConn);
}

CloneKernelConn::~CloneKernelConn() {
   for( int k=0; k<numAxonalArborLists; k++ ) {
      axonalArborList[k] = NULL;
   }
}

int CloneKernelConn::initialize_base() {
   originalConn = NULL;
   return PV_SUCCESS;
}

int CloneKernelConn::initialize(const char * name, HyPerCol * hc,
      HyPerLayer * pre, HyPerLayer * post, ChannelType channel,
      KernelConn * originalConn) {
   this->originalConn = originalConn;
   return HyPerConn::initialize(name, hc, pre, post, channel, NULL);
}

int CloneKernelConn::setPatchSize(const char * filename) {
   assert(filename == NULL);
   nxp = originalConn->xPatchSize();
   nyp = originalConn->yPatchSize();
   nfp = originalConn->fPatchSize();
   int xScalePre = pre->getXScale();
   int xScalePost = post->getXScale();
   int status = checkPatchSize(nxp, xScalePre, xScalePost, 'x');
   if( status == PV_SUCCESS) {
      int yScalePre = pre->getYScale();
      int yScalePost = post->getYScale();
      status = checkPatchSize(nyp, yScalePre, yScalePost, 'y');
   }
   return status;
}

PVPatch ** CloneKernelConn::createWeights(PVPatch ** patches,
      int nPatches, int nxPatch, int nyPatch, int nfPatch) {
   assert(numAxonalArborLists == 1);
   assert(patches == NULL);
   assert(nPatches == originalConn->numWeightPatches(0));
   kernelPatches = (PVPatch**) calloc(sizeof(PVPatch*), nPatches);
   if(kernelPatches == NULL) {
      fprintf(stderr, "Group \"%s\": Unable to allocate memory for kernelPatches.\n", name);
      exit(EXIT_FAILURE);
   }
   for(int k=0; k<nPatches; k++) {
      kernelPatches[k] = originalConn->getKernelPatch(k);
   }
   return kernelPatches;
}

PVPatch ** CloneKernelConn::createWeights(PVPatch ** patches) {
   const int arbor = 0;
   int nPatches = originalConn->numWeightPatches(arbor);

   return createWeights(patches, nPatches, nxp, nyp, nfp);
}

int CloneKernelConn::createAxonalArbors() {
   for( int k=0; k<numAxonalArborLists; k++ ) {
      axonalArborList[k] = originalConn->axonalArbor(0,k);
   }
   return PV_SUCCESS;
}

PVPatch ** CloneKernelConn::initializeWeights(PVPatch ** patches,
      int numPatches, const char * filename) {
   return patches;
   // nothing to be done as the patches point to originalConn's space.
}

} // end namespace PV
