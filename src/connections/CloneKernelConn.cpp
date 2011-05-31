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

PVPatch ** CloneKernelConn::allocWeights(PVPatch ** patches, int nPatches,
      int nxPatch, int nyPatch, int nfPatch) {

   const int arbor = 0;
   int numKernelPatches = numDataPatches(arbor);
   assert( numKernelPatches == originalConn->numDataPatches(arbor) );
   assert(kernelPatches == NULL);
   kernelPatches = (PVPatch**) calloc(sizeof(PVPatch*), numKernelPatches);
   assert(kernelPatches != NULL);
   for (int kernelIndex = 0; kernelIndex < numKernelPatches; kernelIndex++) {
      kernelPatches[kernelIndex] = originalConn->getKernelPatch(kernelIndex);
   }
   for (int patchIndex = 0; patchIndex < nPatches; patchIndex++) {
      patches[patchIndex] = pvpatch_new(nxPatch, nyPatch, nfPatch);
      int kernelIndex = this->patchIndexToKernelIndex(patchIndex);
      patches[patchIndex]->data = kernelPatches[kernelIndex]->data;
   }
   return patches;
}

PVPatch ** CloneKernelConn::initializeWeights(PVPatch ** patches,
      int numPatches, const char * filename) {
   return patches;
   // nothing to be done as the weight patches point to originalConn's space.
}

int CloneKernelConn::deleteWeights() {
   free(kernelPatches);  // don't delete kernelPatches[k] as it belongs to originalConn
   return HyPerConn::deleteWeights();
}

} // end namespace PV
