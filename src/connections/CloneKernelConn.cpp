/* CloneKernelConn.cpp
 *
 * Created on: May 23, 2011
 *     Author: peteschultz
 */

#include "CloneKernelConn.hpp"

namespace PV {

CloneKernelConn::CloneKernelConn(){
   initialize_base();
}

CloneKernelConn::CloneKernelConn(const char * name, HyPerCol * hc,
      HyPerLayer * pre, HyPerLayer * post, ChannelType channel,
      KernelConn * originalConn) {
   initialize_base();
   initialize(name, hc, pre, post, channel, originalConn);
}

int CloneKernelConn::initialize_base() {
   originalConn = NULL;
   return PV_SUCCESS;
}

int CloneKernelConn::initialize(const char * name, HyPerCol * hc,
      HyPerLayer * pre, HyPerLayer * post, ChannelType channel,
      KernelConn * originalConn) {
   this->originalConn = originalConn;
   InitCloneKernelWeights * weightInit = new InitCloneKernelWeights();
   assert(weightInit != NULL);
   int status = HyPerConn::initialize(name, hc, pre, post, channel, NULL, weightInit);
   delete weightInit;
   return status;
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

int CloneKernelConn::constructWeights(const char * filename) {
   PVPatch *** patches = (PVPatch ***) malloc(numberOfAxonalArborLists()*sizeof(PVPatch **));
   if( patches==NULL ){ constructWeightsOutOfMemory(); abort(); }
   set_wPatches(patches);

   pvdata_t *** patchstart = (pvdata_t ***) malloc( numberOfAxonalArborLists()*sizeof(pvdata_t **) );
   if( patchstart==NULL ){ constructWeightsOutOfMemory(); abort(); }
   setGSynPatchStart(patchstart);
   pvdata_t ** gSynPatchStartBuffer = (pvdata_t **) malloc(
         (this->shrinkPatches_flag ? numAxonalArborLists : 1)
               * preSynapticLayer()->getNumExtended() * sizeof(pvdata_t *));
   if (gSynPatchStartBuffer == NULL) { constructWeightsOutOfMemory(); abort(); }

   size_t ** postoffset = (size_t **) malloc( numberOfAxonalArborLists()*sizeof(size_t **) );
   if( postoffset==NULL ){ constructWeightsOutOfMemory(); abort(); }
   setAPostOffset(postoffset);
   size_t * aPostOffsetBuffer = (size_t *) malloc(
         (this->shrinkPatches_flag ? numAxonalArborLists : 1)
               * preSynapticLayer()->getNumExtended() * sizeof(size_t));
   if( aPostOffsetBuffer == NULL ) { constructWeightsOutOfMemory(); abort(); }

   int * delayptr = (int *) malloc(numAxonalArborLists * sizeof(int));
   if( delayptr == NULL ) { constructWeightsOutOfMemory(); abort(); }
   setDelays(delayptr);

   pvdata_t ** datastart;
   datastart = (pvdata_t **) malloc(numAxonalArborLists * sizeof(pvdata_t *));
   if( datastart == NULL ) { constructWeightsOutOfMemory(); abort(); }
   set_wDataStart(datastart);

   datastart = (pvdata_t **) malloc(numAxonalArborLists * sizeof(pvdata_t *));
   if( datastart == NULL ) { constructWeightsOutOfMemory(); abort(); }
   set_dwDataStart(datastart);

   set_kernelPatches(originalConn->getAllKernelPatches());

   int arborstep = this->shrinkPatches_flag * preSynapticLayer()->getNumExtended();
   for (int arbor = 0; arbor < numberOfAxonalArborLists(); arbor++) {
      get_wPatches()[arbor] = originalConn->weights(arbor);

      patchstart[arbor] = gSynPatchStartBuffer;
      postoffset[arbor] = aPostOffsetBuffer;
      for( int kex=0; kex<numWeightPatches(); kex++ ) {
         patchstart[arbor][kex] = originalConn->getGSynPatchStart(kex,arbor);
         postoffset[arbor][kex] = originalConn->getAPostOffset(kex,arbor);
      }
      gSynPatchStartBuffer += arborstep;
      aPostOffsetBuffer += arborstep;

      delayptr[arbor] = originalConn->getDelay(arbor);
      set_wDataStart(arbor, originalConn->get_wDataStart(arbor));
      set_dwDataStart(arbor, originalConn->get_wDataStart(arbor));
   }

   dKernelPatches = NULL;

   initShrinkPatches();
   // Don't call shrinkPatches() since the original connection will have already shrunk patches

   return PV_SUCCESS;
}

void CloneKernelConn::constructWeightsOutOfMemory() {
   connOutOfMemory("CloneKernelConn::constructWeightsOutOfMemory()");
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

int CloneKernelConn::initShrinkPatches() {
   shrinkPatches_flag = originalConn->getShrinkPatches_flag();
   return PV_SUCCESS;
}

int CloneKernelConn::setParams(PVParams * params) {
   numAxonalArborLists = originalConn->numberOfAxonalArborLists();
   plasticityFlag = false; // CloneKernelConn updates automatically, since it's done using pointer magic.
   stochasticReleaseFlag = params->value(name, "stochasticReleaseFlag", 0.0f, true);
   writeCompressedWeights = params->value(name, "writeCompressedWeights", 0.0f, true);
   return PV_SUCCESS;
}

int CloneKernelConn::updateState(float time, float dt) {
   lastUpdateTime = originalConn->getLastUpdateTime();
   return PV_SUCCESS;
}

int CloneKernelConn::deleteWeights() {
   // Have to make sure not to free memory belonging to originalConn.
   // Set pointers that point into originalConn to NULL so that free() has no effect.
   for( int arbor=0; arbor<numberOfAxonalArborLists(); arbor++) {
      PVPatch *** p = get_wPatches();
      p[arbor] = NULL;
      set_wDataStart(arbor,NULL);
      set_dwDataStart(arbor,NULL);
   }
   set_kernelPatches(NULL);
   return HyPerConn::deleteWeights();
}

} // end namespace PV
