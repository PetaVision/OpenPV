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
   int status = PV_SUCCESS;

   if( status == PV_SUCCESS ) status = initShrinkPatches();

   if( status == PV_SUCCESS ) status = createArbors();

   if( status == PV_SUCCESS ) status = setPatchSize(NULL);
   if( status == PV_SUCCESS ) status = setPatchStrides();

   for( int arbor=0; arbor<numberOfAxonalArborLists(); arbor++) {
      get_wDataStart()[arbor] = originalConn->get_wDataStart(arbor);
      get_wPatches()[arbor] = originalConn->weights(arbor);
      // this->setKernelPatches(originalConn->getKernelPatches(arbor),arbor);
      if( status == PV_SUCCESS )
         status = createAxonalArbors(arbor); // sets gSynPatchStart[arbor][*] and aPostOffset[arbor][*]
      if( status != PV_SUCCESS ) break;
   }

   // Don't call initPlasticityPatches since plasticityFlag is always false.
   // Don't call shrinkPatches() since the original connection will have already shrunk patches
   return status;
}

void CloneKernelConn::constructWeightsOutOfMemory() {
   connOutOfMemory("CloneKernelConn::constructWeightsOutOfMemory()");
}

int CloneKernelConn::createAxonalArbors(int arborId) {
   int numPatches = numWeightPatches();
   for( int kex = 0; kex < numPatches; kex++ ) {
      // kex is in extended frame, this makes transformations more difficult
      int kl, offset, nxPatch, nyPatch, dx, dy;
      calcPatchSize(arborId, kex, &kl, &offset, &nxPatch, &nyPatch, &dx, &dy);
      pvdata_t * gSyn = post->getChannel(channel) + kl;
      getGSynPatchStart()[arborId][kex] = gSyn;
      getAPostOffset()[arborId][kex] = offset;
      // Don't call pvpatch_adjust because weight patches point to the
      // original conn's weight patches, which were already shrunk.
   }
   return PV_SUCCESS;
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
   // Set pointers that point into originalConn to NULL so that free() has no effect
   // when KernelConn::deleteWeights or HyPerConn::deleteWeights is called
   for(int arbor=0; arbor<numberOfAxonalArborLists(); arbor++) {
      get_wPatches()[arbor] = NULL;
      set_wDataStart(arbor,NULL);
   }
   // set_kernelPatches(NULL);

   return 0; // HyPerConn::deleteWeights(); // HyPerConn destructor calls HyPerConn::deleteWeights()
}

CloneKernelConn::~CloneKernelConn() {
   deleteWeights();
}

} // end namespace PV
