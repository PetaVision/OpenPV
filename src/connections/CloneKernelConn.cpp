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
      const char * pre_layer_name, const char * post_layer_name,
      const char * original_kernelconn_name) {
   initialize_base();
   initialize(name, hc, pre_layer_name, post_layer_name, original_kernelconn_name);
}

int CloneKernelConn::initialize_base() {
   originalConn = NULL;
   return PV_SUCCESS;
}

int CloneKernelConn::initialize(const char * name, HyPerCol * hc,
      const char * pre_layer_name, const char * post_layer_name,
      const char * original_kernelconn_name) {
   InitCloneKernelWeights * weightInit = new InitCloneKernelWeights();
   assert(weightInit != NULL);
   int status = KernelConn::initialize(name, hc, pre_layer_name, post_layer_name, NULL, weightInit);
   if (original_kernelconn_name==NULL) {
      fprintf(stderr, "CloneKernelConn \"%s\" error in rank %d process: originalConnName must be set.\n",
            name, hc->columnId());
      abort();
   }
   originalConnName = strdup(original_kernelconn_name);
   if (originalConnName == NULL) {
      fprintf(stderr, "CloneKernelConn \"%s\" error in rank %d process: unable to allocate memory for originalConnName \"%s\": %s\n",
            name, hc->columnId(), original_kernelconn_name, strerror(errno));
      abort();
   }
   return status;
}

//int CloneKernelConn::setPatchSize(const char * filename) {
//   // nxp, nyp, nfp were set by the read-methods called by HyPerConn::setParams
//   assert(filename == NULL);
//   int xScalePre = pre->getXScale();
//   int xScalePost = post->getXScale();
//   int status = checkPatchSize(nxp, xScalePre, xScalePost, 'x');
//   if( status == PV_SUCCESS) {
//      int yScalePre = pre->getYScale();
//      int yScalePost = post->getYScale();
//      status = checkPatchSize(nyp, yScalePre, yScalePost, 'y');
//   }
//   return status;
//}

int CloneKernelConn::initNormalize() {
   normalize_flag = false;
   return PV_SUCCESS;
}

int CloneKernelConn::constructWeights(const char * filename) {
   int status = PV_SUCCESS;

   // CloneKernelConn::readShrinkPatches does nothing; shrinkPatches_flag is set in communicateInitInfo()
   // if( status == PV_SUCCESS ) readShrinkPatches(parent->parameters());

   // if( status == PV_SUCCESS ) status = setPatchSize(NULL);
   if( status == PV_SUCCESS ) status = setPatchStrides();

   wPatches = this->originalConn->get_wPatches();
   wDataStart = this->originalConn->get_wDataStart();
   gSynPatchStart = this->originalConn->getGSynPatchStart();
   aPostOffset = this->originalConn->getAPostOffset();
   dwDataStart = this->originalConn->get_dwDataStart();

//   for( int arbor=0; arbor<numberOfAxonalArborLists(); arbor++) {
//      get_wDataStart()[arbor] = originalConn->get_wDataStart(arbor);
//      get_wPatches()[arbor] = originalConn->weights(arbor);
//      // this->setKernelPatches(originalConn->getKernelPatches(arbor),arbor);
//      if( status == PV_SUCCESS )
//         status = createAxonalArbors(arbor); // sets gSynPatchStart[arbor][*] and aPostOffset[arbor][*]
//      if( status != PV_SUCCESS ) break;
//   }

   // Don't call initPlasticityPatches since plasticityFlag is always false.
   // Don't call shrinkPatches() since the original connection will have already shrunk patches
   return status;
}

void CloneKernelConn::constructWeightsOutOfMemory() {
   connOutOfMemory("CloneKernelConn::constructWeightsOutOfMemory()");
}

int CloneKernelConn::createAxonalArbors(int arborId) {
//   int numPatches = getNumWeightPatches();
//   for( int kex = 0; kex < numPatches; kex++ ) {
//      // kex is in extended frame, this makes transformations more difficult
//      int kl, offset, nxPatch, nyPatch, dx, dy;
//      calcPatchSize(arborId, kex, &kl, &offset, &nxPatch, &nyPatch, &dx, &dy);
//      pvdata_t * gSyn = post->getChannel(channel) + kl;
//      getGSynPatchStart()[arborId][kex] = gSyn;
//      getAPostOffset()[arborId][kex] = offset;
//      // Don't call pvpatch_adjust because weight patches point to the
//      // original conn's weight patches, which were already shrunk.
//   }
   return PV_SUCCESS;
}

PVPatch *** CloneKernelConn::initializeWeights(PVPatch *** patches, pvdata_t ** dataStart,
      int numPatches, const char * filename) {
   return patches;
   // nothing to be done as the weight patches point to originalConn's space.
}

void CloneKernelConn::readShrinkPatches(PVParams * params) {
   // During the communication phase, shrinkPatches_flag will be copied from originalConn
}

int CloneKernelConn::setParams(PVParams * params) {
   return KernelConn::setParams(params);
}

void CloneKernelConn::readNumAxonalArbors(PVParams * params) {
   // During the communication phase, numAxonalArbors will be copied from originalConn
}

void CloneKernelConn::readPlasticityFlag(PVParams * params) {
   plasticityFlag = false; // CloneKernelConn updates automatically, since it's done using pointer magic.
}

int CloneKernelConn::readPatchSize(PVParams * params) {
   // During the communication phase, nxp, nyp, nxpShrunken, nypShrunken will be copied from originalConn
   return PV_SUCCESS;
}

int CloneKernelConn::readNfp(PVParams * params) {
   // During the communication phase, nfp will be copied from originalConn
   return PV_SUCCESS;
}

int CloneKernelConn::communicateInitInfo() {
   // Need to set originalConn before calling KernelConn::communicate, since KernelConn::communicate calls setPatchSize, which needs originalConn.
   HyPerConn * origHyPerConn = parent->getConnFromName(originalConnName);
   if (origHyPerConn == NULL) {
      fprintf(stderr, "CloneKernelConn \"%s\" error in rank %d process: originalConnName \"%s\" is not a connection in the column.\n",
            name, parent->columnId(), originalConnName);
   }
   originalConn = dynamic_cast<KernelConn *>(origHyPerConn);
   if (originalConn == NULL) {
      fprintf(stderr, "CloneKernelConn \"%s\" error in rank %d process: originalConnName \"%s\" must be a KernelConn or a KernelConn-derived class.\n",
            name, parent->columnId(), originalConnName);
   }

   numAxonalArborLists = originalConn->numberOfAxonalArborLists();
   int status = KernelConn::communicateInitInfo();
   if (status != PV_SUCCESS) return status;

   // Presynaptic layers of the CloneKernelConn and its original conn must have the same size, or the patches won't line up with each other.
   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const PVLayerLoc * origPreLoc = originalConn->preSynapticLayer()->getLayerLoc();

   if (preLoc->nx != origPreLoc->nx || preLoc->ny != origPreLoc->ny || preLoc->nf != origPreLoc->nf || preLoc->nb != origPreLoc->nb ) {
      if (parent->icCommunicator()->commRank()==0) {
         fprintf(stderr, "CloneKernelConn \"%s\" error in rank %d process: CloneKernelConn and originalConn \"%s\" must have presynaptic layers with the same geometry (including margin width).\n",
               name, parent->columnId(), originalConn->getName());
         fprintf(stderr, "{nx=%d, ny=%d, nf=%d, nb=%d} versus {nx=%d, ny=%d, nf=%d, nb=%d}\n",
                 preLoc->nx, preLoc->ny, preLoc->nf, preLoc->nb, origPreLoc->nx, origPreLoc->ny, origPreLoc->nf, origPreLoc->nb);
      }
      abort();
   }

   //Redudent read in case it's a clone of a clone
   numAxonalArborLists = originalConn->numberOfAxonalArborLists();
   shrinkPatches_flag = originalConn->getShrinkPatches_flag();

   return status;
}

int CloneKernelConn::setPatchSize() {
   assert(originalConn);
   nxp = originalConn->xPatchSize();
   nyp = originalConn->yPatchSize();
   nxpShrunken = originalConn->getNxpShrunken();
   nypShrunken = originalConn->getNypShrunken();
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
	   wPatches = NULL;
	   wDataStart = NULL;
	   gSynPatchStart = NULL;
	   aPostOffset = NULL;
	   dwDataStart = NULL;
//   for(int arbor=0; arbor<numberOfAxonalArborLists(); arbor++) {
//      get_wPatches()[arbor] = NULL;
//      set_wDataStart(arbor,NULL);
//   }
   // set_kernelPatches(NULL);

   return 0; // HyPerConn::deleteWeights(); // HyPerConn destructor calls HyPerConn::deleteWeights()
}

CloneKernelConn::~CloneKernelConn() {
   deleteWeights();
}

} // end namespace PV
