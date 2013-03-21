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
      HyPerLayer * pre, HyPerLayer * post,
      KernelConn * originalConn) {
   initialize_base();
   initialize(name, hc, pre, post, originalConn);
}

int CloneKernelConn::initialize_base() {
   originalConn = NULL;
   return PV_SUCCESS;
}

int CloneKernelConn::initialize(const char * name, HyPerCol * hc,
      HyPerLayer * pre, HyPerLayer * post,
      KernelConn * originalConn) {
   // Presynaptic layers of the CloneKernelConn and its original conn must have the same size, or the patches won't line up with each other.
   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const PVLayerLoc * origPreLoc = originalConn->preSynapticLayer()->getLayerLoc();
   if (preLoc->nx != origPreLoc->nx || preLoc->ny != origPreLoc->ny || preLoc->nf != origPreLoc->nf || preLoc->nb != origPreLoc->nb ) {
      if (hc->icCommunicator()->commRank()==0) {
         fprintf(stderr, "CloneKernelConn::initialize error: CloneKernelConn \"%s\" and KernelConn \"%s\" must have presynaptic layers with the same geometry (including margin width).\n", name, originalConn->getName());
         fprintf(stderr, "{nx=%d, ny=%d, nf=%d, nb=%d} versus {nx=%d, ny=%d, nf=%d, nb=%d}\n",
                 preLoc->nx, preLoc->ny, preLoc->nf, preLoc->nb, origPreLoc->nx, origPreLoc->ny, origPreLoc->nf, origPreLoc->nb);
      }
      abort();
   }
   this->originalConn = originalConn;
   InitCloneKernelWeights * weightInit = new InitCloneKernelWeights();
   assert(weightInit != NULL);
   int status = HyPerConn::initialize(name, hc, pre, post, NULL, weightInit);
   //why doesn't clonekernelconn call kernelconn's initialize???
   //kernelconns need this and the GPU stuff...
   initPatchToDataLUT();
   delete weightInit;
   return status;
}

int CloneKernelConn::setPatchSize(const char * filename) {
   // nxp, nyp, nfp were set by the read-methods called by HyPerConn::setParams
   assert(filename == NULL);
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

   if( status == PV_SUCCESS ) readShrinkPatches(parent->parameters());

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
   int numPatches = getNumWeightPatches();
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

PVPatch *** CloneKernelConn::initializeWeights(PVPatch *** patches, pvdata_t ** dataStart,
      int numPatches, const char * filename) {
   return patches;
   // nothing to be done as the weight patches point to originalConn's space.
}

void CloneKernelConn::readShrinkPatches(PVParams * params) {
   assert(originalConn);
   shrinkPatches_flag = originalConn->getShrinkPatches_flag();
}

int CloneKernelConn::setParams(PVParams * params) {
   return KernelConn::setParams(params);
}

void CloneKernelConn::readNumAxonalArborLists(PVParams * params) {
   assert(originalConn);
   numAxonalArborLists = originalConn->numberOfAxonalArborLists();
}

void CloneKernelConn::readPlasticityFlag(PVParams * params) {
   plasticityFlag = false; // CloneKernelConn updates automatically, since it's done using pointer magic.
}

int CloneKernelConn::readPatchSize(PVParams * params) {
   nxp = originalConn->xPatchSize();
   nyp = originalConn->yPatchSize();
   return PV_SUCCESS;
}

int CloneKernelConn::readNfp(PVParams * params) {
   nfp = originalConn->fPatchSize();
   return PV_SUCCESS;
}

int CloneKernelConn::updateState(double time, double dt) {
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
