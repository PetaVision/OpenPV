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
   for( int k=0; k<numberOfAxonalArborLists(); k++ ) {
      setArbor(NULL, k);
      //axonalArborList[k] = NULL;
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

int CloneKernelConn::initNormalize() {
   normalize_flag = false;
   return PV_SUCCESS;
}

PVPatch ** CloneKernelConn::allocWeights(PVPatch ** patches, int nPatches,
      int nxPatch, int nyPatch, int nfPatch, int axonId) {

   //const int arbor = 0;
   int numKernelPatches = numDataPatches();
   assert( numKernelPatches == originalConn->numDataPatches() );
   //assert(kernelPatches == NULL);
   //kernelPatches = (PVPatch**) calloc(sizeof(PVPatch*), numKernelPatches);
   //assert(kernelPatches != NULL);
   PVPatch** newKernelPatch = (PVPatch**) calloc(sizeof(PVPatch*), numKernelPatches);
   assert(newKernelPatch != NULL);
   setKernelPatches(newKernelPatch, axonId);
   for (int kernelIndex = 0; kernelIndex < numKernelPatches; kernelIndex++) {
      setKernelPatch(originalConn->getKernelPatch(axonId, kernelIndex), axonId, kernelIndex);
      //kernelPatches[kernelIndex] = originalConn->getKernelPatch(kernelIndex);
   }
   for (int patchIndex = 0; patchIndex < nPatches; patchIndex++) {
      patches[patchIndex] = pvpatch_new(nxPatch, nyPatch, nfPatch);
      int kernelIndex = this->patchIndexToKernelIndex(patchIndex);
      patches[patchIndex]->data = getKernelPatch(axonId, kernelIndex)->data;
   }
   return patches;
}

PVPatch *** CloneKernelConn::initializeWeights(PVPatch *** patches,
      int numPatches, const char * filename) {
   return patches;
   // nothing to be done as the weight patches point to originalConn's space.
}
int CloneKernelConn::setWPatches(PVPatch ** patches, int arborId) {
   return HyPerConn::setWPatches(patches, arborId);
}
int CloneKernelConn::setdWPatches(PVPatch ** patches, int arborId) {
   return HyPerConn::setWPatches(patches, arborId);
}

int CloneKernelConn::deleteWeights() {
   //free(kernelPatches);  // don't delete kernelPatches[k] as it belongs to originalConn
   free(getAllKernelPatches());
   return HyPerConn::deleteWeights();
}

} // end namespace PV
