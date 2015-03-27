/*
 * TransposePoolingConn.cpp *
 *  Created on: March 25, 2015
 *      Author: slundquist
 */

#include "TransposePoolingConn.hpp"

namespace PV {

TransposePoolingConn::TransposePoolingConn() {
   initialize_base();
}  // TransposePoolingConn::~TransposePoolingConn()

TransposePoolingConn::TransposePoolingConn(const char * name, HyPerCol * hc) {
   initialize_base();
   int status = initialize(name, hc);
}

TransposePoolingConn::~TransposePoolingConn() {
   free(originalConnName); originalConnName = NULL;
   deleteWeights();
   if(needPost){
      postConn = NULL;
   }
   //Transpose conn doesn't allocate postToPreActivity
   if(this->postToPreActivity){
      postToPreActivity = NULL;
   }
}  // TransposePoolingConn::~TransposePoolingConn()

int TransposePoolingConn::initialize_base() {
   plasticityFlag = false; // Default value; override in params
   weightUpdatePeriod = 1;   // Default value; override in params
   weightUpdateTime = 0;
   // TransposePoolingConn::initialize_base() gets called after
   // HyPerConn::initialize_base() so these default values override
   // those in HyPerConn::initialize_base().
   // TransposePoolingConn::initialize_base() gets called before
   // HyPerConn::initialize(), so these values still get overridden
   // by the params file values.

   originalConnName = NULL;
   originalConn = NULL;
   needFinalize = true;
   return PV_SUCCESS;
}  // TransposePoolingConn::initialize_base()

int TransposePoolingConn::initialize(const char * name, HyPerCol * hc) {
   int status = PV_SUCCESS;
   if (status == PV_SUCCESS) status = HyPerConn::initialize(name, hc, NULL, NULL);
   return status;
}

int TransposePoolingConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_originalConnName(ioFlag);
   return status;
}

// We override many ioParam-methods because TransposePoolingConn will determine
// the associated parameters from the originalConn's values.
// communicateInitInfo will check if those parameters exist in params for
// the CloneKernelConn group, and whether they are consistent with the
// originalConn parameters.
// If consistent, issue a warning that the param is unnecessary and continue.
// If inconsistent, issue an error and quit.
// We can't do that in the read-method because we can't be sure originalConn
// has set its own parameter yet (or even if it's been instantiated),
// and in theory originalConn could be a subclass that determines
// the parameter some way other than reading its own parameter
// group's param directly.

void TransposePoolingConn::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   // During the communication phase, numAxonalArbors will be copied from originalConn
   if (ioFlag==PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryStringParameter(name, "sharedWeights");
   }
}

void TransposePoolingConn::ioParam_weightInitType(enum ParamsIOFlag ioFlag) {
   // TransposePoolingConn doesn't use a weight initializer
   if (ioFlag==PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryStringParameter(name, "weightInitType", NULL);
   }
}

void TransposePoolingConn::ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      initializeFromCheckpointFlag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "initializeFromCheckpointFlag");
   }
   // During the setInitialValues phase, the conn will be computed from the original conn, so initializeFromCheckpointFlag is not needed.
}

void TransposePoolingConn::ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag) {
}

void TransposePoolingConn::ioParam_plasticityFlag(enum ParamsIOFlag ioFlag) {
   // During the communication phase, plasticityFlag will be copied from originalConn
}

void TransposePoolingConn::ioParam_triggerFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) {
      triggerFlag = false; // make sure that TransposePoolingConn always checks if its originalConn has updated
      parent->parameters()->handleUnnecessaryParameter(name, "triggerFlag", triggerFlag);
   }
}

void TransposePoolingConn::ioParam_combine_dW_with_W_flag(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) {
      combine_dW_with_W_flag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "combine_dW_with_W_flag", combine_dW_with_W_flag);
   }
}

void TransposePoolingConn::ioParam_nxp(enum ParamsIOFlag ioFlag) {
   // TransposePoolingConn determines nxp from originalConn, during communicateInitInfo
}

void TransposePoolingConn::ioParam_nyp(enum ParamsIOFlag ioFlag) {
   // TransposePoolingConn determines nyp from originalConn, during communicateInitInfo
}

void TransposePoolingConn::ioParam_nfp(enum ParamsIOFlag ioFlag) {
   // TransposePoolingConn determines nfp from originalConn, during communicateInitInfo
}

void TransposePoolingConn::ioParam_dWMax(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      dWMax = 1.0;
      parent->parameters()->handleUnnecessaryParameter(name, "dWMax", dWMax);
   }
}

void TransposePoolingConn::ioParam_keepKernelsSynchronized(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      keepKernelsSynchronized_flag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "keepKernelsSynchronized", keepKernelsSynchronized_flag);
   }
}

void TransposePoolingConn::ioParam_weightUpdatePeriod(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      weightUpdatePeriod = parent->getDeltaTime();
      // Every timestep needUpdate checks originalConn's lastUpdateTime against transpose's lastUpdateTime, so weightUpdatePeriod and initialWeightUpdateTime aren't needed
      parent->parameters()->handleUnnecessaryParameter(name, "weightUpdatePeriod");
   }
}

void TransposePoolingConn::ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      initialWeightUpdateTime = parent->getStartTime();
      // Every timestep needUpdate checks originalConn's lastUpdateTime against transpose's lastUpdateTime, so weightUpdatePeriod and initialWeightUpdateTime aren't needed
      parent->parameters()->handleUnnecessaryParameter(name, "initialWeightUpdateTime", initialWeightUpdateTime);
      weightUpdateTime = initialWeightUpdateTime;
   }
}

#ifdef OBSOLETE // Marked obsolete Dec 2, 2014.  Use sharedWeights=false instead of windowing.
void TransposePoolingConn::ioParam_useWindowPost(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      useWindowPost = false;
      parent->parameters()->handleUnnecessaryParameter(name, "useWindowPost");
   }
}
#endif // OBSOLETE

void TransposePoolingConn::ioParam_shrinkPatches(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      shrinkPatches_flag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "shrinkPatches", shrinkPatches_flag);
   }
}

void TransposePoolingConn::ioParam_normalizeMethod(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      normalizer = NULL;
      normalizeMethod = strdup("none");
      parent->parameters()->handleUnnecessaryStringParameter(name, "normalizeMethod", "none");
   }
}

void TransposePoolingConn::ioParam_originalConnName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "originalConnName", &originalConnName);
}

#ifdef OBSOLETE // Marked obsolete Mar 20, 2015.  Not used since creating the InitWeights object was taken out of HyPerConn.
InitWeights * TransposePoolingConn::handleMissingInitWeights(PVParams * params) {
   // TransposePoolingConn doesn't use InitWeights; it initializes the weight by transposing the initial weights of originalConn
   return NULL;
}
#endif // OBSOLETE // Marked obsolete Mar 20, 2015.  Not used since creating the InitWeights object was taken out of HyPerConn.

int TransposePoolingConn::communicateInitInfo() {
   int status = PV_SUCCESS;
   BaseConnection * originalConnBase = parent->getConnFromName(this->originalConnName);
   if (originalConnBase==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: originalConnName \"%s\" does not refer to any connection in the column.\n", parent->parameters()->groupKeywordFromName(name), name, this->originalConnName);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   originalConn = dynamic_cast<PoolingConn *>(originalConnBase);
   if (originalConn == NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "TransposePoolingConn \"%s\" error: originalConnName \"%s\" is not a PoolingConn.\n", name, originalConnName);
         status = PV_FAILURE;
      }
   }
   if (status != PV_SUCCESS) return status;

   if (!originalConn->getInitInfoCommunicatedFlag()) {
      if (parent->columnId()==0) {
         const char * connectiontype = parent->parameters()->groupKeywordFromName(name);
         printf("%s \"%s\" must wait until original connection \"%s\" has finished its communicateInitInfo stage.\n", connectiontype, name, originalConn->getName());
      }
      return PV_POSTPONE;
   }


   sharedWeights = originalConn->usingSharedWeights();
   parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights", sharedWeights);

   numAxonalArborLists = originalConn->numberOfAxonalArborLists();
   parent->parameters()->handleUnnecessaryParameter(name, "numAxonalArbors", numAxonalArborLists);

   plasticityFlag = originalConn->getPlasticityFlag();
   parent->parameters()->handleUnnecessaryParameter(name, "plasticityFlag", plasticityFlag);

   if(originalConn->getShrinkPatches_flag()) {
      if (parent->columnId()==0) {
         fprintf(stderr, "TransposePoolingConn \"%s\" error: original conn \"%s\" has shrinkPatches set to true.  TransposePoolingConn has not been implemented for that case.\n", name, originalConn->getName());
      }
#ifdef PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }

   status = HyPerConn::communicateInitInfo(); // calls setPatchSize()
   if (status != PV_SUCCESS) return status;

   if(!originalConn->needPostIndex()){
      if (parent->columnId()==0) {
         fprintf(stderr, "TransposePoolingConn \"%s\" error: original pooling conn \"%s\" needs to have a postIndexLayer.\n", name, originalConnName);
         status = PV_FAILURE;
      }
   }
   if (status != PV_SUCCESS) return status;

   //Check post layer phases to make sure it matches
   if(originalConn->postSynapticLayer()->getPhase() >= post->getPhase()){
      fprintf(stderr, "TransposePoolingConn \"%s\" warning: originalConn's post layer phase is greater or equal than this layer's post. Behavior undefined.\n", name);
   }

   if(originalConn->getPvpatchAccumulateType() != ACCUMULATE_MAXPOOLING){
      fprintf(stderr, "TransposePoolingConn \"%s\" error: originalConn is not doing max pooling. TransposePoolingConn requires max pooling for the original conn.\n", name);
      exit(-1);
   }


   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const PVLayerLoc * origPostLoc = originalConn->postSynapticLayer()->getLayerLoc();
   if (preLoc->nx != origPostLoc->nx || preLoc->ny != origPostLoc->ny || preLoc->nf != origPostLoc->nf) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: transpose's pre layer and original connection's post layer must have the same dimensions.\n", parent->parameters()->groupKeywordFromName(name), name);
         fprintf(stderr, "    (x=%d, y=%d, f=%d) versus (x=%d, y=%d, f=%d).\n", preLoc->nx, preLoc->ny, preLoc->nf, origPostLoc->nx, origPostLoc->ny, origPostLoc->nf);
      }
#ifdef PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }
   const PVLayerLoc * postLoc = pre->getLayerLoc();
   const PVLayerLoc * origPreLoc = originalConn->postSynapticLayer()->getLayerLoc();
   if (postLoc->nx != origPreLoc->nx || postLoc->ny != origPreLoc->ny || postLoc->nf != origPreLoc->nf) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: transpose's post layer and original connection's pre layer must have the same dimensions.\n", parent->parameters()->groupKeywordFromName(name), name);
         fprintf(stderr, "    (x=%d, y=%d, f=%d) versus (x=%d, y=%d, f=%d).\n", postLoc->nx, postLoc->ny, postLoc->nf, origPreLoc->nx, origPreLoc->ny, origPreLoc->nf);
      }
#ifdef PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }

   originalConn->setNeedPost(true);
   originalConn->setNeedAllocPostWeights(false);

   //Synchronize margines of this post and orig pre, and vice versa
   originalConn->preSynapticLayer()->synchronizeMarginWidth(post);
   post->synchronizeMarginWidth(originalConn->preSynapticLayer());

   originalConn->postSynapticLayer()->synchronizeMarginWidth(pre);
   pre->synchronizeMarginWidth(originalConn->postSynapticLayer());

   //Sync pre margins
   //Synchronize margines of this post and the postIndexLayer, and vice versa
   pre->synchronizeMarginWidth(originalConn->getPostIndexLayer());
   originalConn->getPostIndexLayer()->synchronizeMarginWidth(pre);

   //Need to tell postIndexLayer the number of delays needed by this connection
   int allowedDelay = originalConn->getPostIndexLayer()->increaseDelayLevels(getDelayArraySize());
   if( allowedDelay < getDelayArraySize()) {
      if( this->getParent()->columnId() == 0 ) {
         fflush(stdout);
         fprintf(stderr, "Connection \"%s\": attempt to set delay to %d, but the maximum allowed delay is %d.  Exiting\n", this->getName(), getDelayArraySize(), allowedDelay);
      }
      exit(EXIT_FAILURE);
   }


   return status;
}

int TransposePoolingConn::setPatchSize() {
   // If originalConn is many-to-one, the transpose connection is one-to-many; then xscaleDiff > 0.
   // Similarly, if originalConn is one-to-many, xscaleDiff < 0.

   // Some of the code duplication might be eliminated by adding some functions to convert.h

   assert(pre && post);
   assert(originalConn);

   int xscaleDiff = pre->getXScale() - post->getXScale();
   int nxp_orig = originalConn->xPatchSize();
   int nyp_orig = originalConn->yPatchSize();
   nxp = nxp_orig;
   if(xscaleDiff > 0 ) {
      nxp *= (int) pow( 2, xscaleDiff );
   }
   else if(xscaleDiff < 0) {
      nxp /= (int) pow(2,-xscaleDiff);
      assert(nxp_orig==nxp*pow( 2, (float) (-xscaleDiff) ));
   }

   int yscaleDiff = pre->getYScale() - post->getYScale();
   nyp = nyp_orig;
   if(yscaleDiff > 0 ) {
      nyp *= (int) pow( 2, yscaleDiff );
   }
   else if(yscaleDiff < 0) {
      nyp /= (int) pow(2,-yscaleDiff);
      assert(nyp_orig==nyp*pow( 2, (float) (-yscaleDiff) ));
   }

   nfp = post->getLayerLoc()->nf;
   // post->getLayerLoc()->nf must be the same as originalConn->preSynapticLayer()->getLayerLoc()->nf.
   // This requirement is checked in communicateInitInfo

   parent->parameters()->handleUnnecessaryParameter(name, "nxp", nxp);
   parent->parameters()->handleUnnecessaryParameter(name, "nyp", nyp);
   parent->parameters()->handleUnnecessaryParameter(name, "nfp", nfp);
   return PV_SUCCESS;

}

int TransposePoolingConn::allocateDataStructures() {
   if (!originalConn->getDataStructuresAllocatedFlag()) {
      if (parent->columnId()==0) {
         const char * connectiontype = parent->parameters()->groupKeywordFromName(name);
         printf("%s \"%s\" must wait until original connection \"%s\" has finished its allocateDataStructures stage.\n", connectiontype, name, originalConn->getName());
      }
      return PV_POSTPONE;
   }

   bool tempNeedPost = false;
   //Turn off need post so postConn doesn't get allocated
   if(needPost){
      needPost = false;
      postConn = originalConn;
      //TODO this buffer is only needed if this transpose conn is receiving from post
      originalConn->postConn->allocatePreToPostBuffer();
      postToPreActivity = originalConn->postConn->getPostToPreActivity();
      tempNeedPost = true;
   }
   HyPerConn::allocateDataStructures();

   //Set nessessary buffers
   if(tempNeedPost){
      needPost = true;
   }

   normalizer = NULL;
   
   // normalize_flag = false; // replaced by testing whether normalizer!=NULL
   return PV_SUCCESS;
}

int TransposePoolingConn::constructWeights(){
   setPatchStrides();
   wPatches = this->originalConn->postConn->get_wPatches();
   wDataStart = this->originalConn->postConn->get_wDataStart();
   gSynPatchStart = this->originalConn->postConn->getGSynPatchStart();
   aPostOffset = this->originalConn->postConn->getAPostOffset();
   dwDataStart = this->originalConn->postConn->get_dwDataStart();
   return PV_SUCCESS;
}

int TransposePoolingConn::deleteWeights() {
   // Have to make sure not to free memory belonging to originalConn.
   // Set pointers that point into originalConn to NULL so that free() has no effect
   // when HyPerConn::deleteWeights or HyPerConn::deleteWeights is called
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

int TransposePoolingConn::setInitialValues() {
   int status = PV_SUCCESS;
   if (originalConn->getInitialValuesSetFlag()) {
      status = HyPerConn::setInitialValues(); // calls initializeWeights
   }
   else {
      status = PV_POSTPONE;
   }
   return status;
}

PVPatch*** TransposePoolingConn::initializeWeights(PVPatch*** patches, pvwdata_t** dataStart) {
   // TransposePoolingConn must wait until after originalConn has been normalized, so weight initialization doesn't take place until HyPerCol::run calls finalizeUpdate
   return patches;
}

bool TransposePoolingConn::needUpdate(double timed, double dt) {
   return plasticityFlag && originalConn->getLastUpdateTime() > lastUpdateTime;
}

int TransposePoolingConn::updateState(double time, double dt) {
   return PV_SUCCESS;
}

double TransposePoolingConn::computeNewWeightUpdateTime(double time, double currentUpdateTime) {
   return weightUpdateTime; // TransposePoolingConn does not use weightUpdateTime to determine when to update
}

int TransposePoolingConn::deliverPostsynapticPerspective(PVLayerCube const * activity, int arborID) {
   std::cout << "Delivering from PostSynapticPerspective for TransposePoolingConn not implented yet\n";
   exit(-1);
}

int TransposePoolingConn::deliverPresynapticPerspective(PVLayerCube const * activity, int arborID) {
   //Check if we need to update based on connection's channel
   if(getChannel() == CHANNEL_NOUPDATE){
      return PV_SUCCESS;
   }
   assert(post->getChannel(getChannel()));

   const PVLayerLoc * preLoc = preSynapticLayer()->getLayerLoc();
   const PVLayerLoc * postLoc = postSynapticLayer()->getLayerLoc();

   assert(arborID >= 0);
   const int numExtended = activity->numItems;

#ifdef PV_USE_OPENMP_THREADS
   //Clear all thread gsyn buffer
   if(thread_gSyn){
      int numNeurons = post->getNumNeurons();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for(int i = 0; i < parent->getNumThreads() * numNeurons; i++){
         int ti = i/numNeurons;
         int ni = i % numNeurons;
         thread_gSyn[ti][ni] = 0;
      }
   }
#endif // PV_USE_OPENMP_THREADS

   int numLoop;
   if(activity->isSparse){
      numLoop = activity->numActive;
   }
   else{
      numLoop = numExtended;
   }

   //Grab postIdxLayer's data
   PoolingIndexLayer* postIndexLayer = originalConn->getPostIndexLayer();
   assert(postIndexLayer);
   //Make sure this layer is an integer layer
   assert(postIndexLayer->getDataType() == PV_INT);
   //DataStore * store = parent->icCommunicator()->publisherStore(postIndexLayer->getLayerId());
   //int delay = getDelay(arborID);

   ////TODO this is currently a hack, need to properly implement data types.
   ////int* postIdxData = (int*) store->buffer(LOCAL, delay);
   //float* postIdxData = (float*)store->buffer(LOCAL, delay);
   int* postIdxData = postIndexLayer->getActivity();

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(guided)
#endif
   for (int loopIndex = 0; loopIndex < numLoop; loopIndex++) {
      int kPreExt;
      if(activity->isSparse){
         kPreExt = activity->activeIndices[loopIndex];
      }
      else{
         kPreExt = loopIndex;
      }

      //Only doing global restricted
      const int kxPreExt = kxPos(kPreExt, preLoc->nx + preLoc->halo.lt + preLoc->halo.rt, preLoc->ny + preLoc->halo.dn + preLoc->halo.up, preLoc->nf);
      const int kyPreExt = kyPos(kPreExt, preLoc->nx + preLoc->halo.lt + preLoc->halo.rt, preLoc->ny + preLoc->halo.dn + preLoc->halo.up, preLoc->nf);
      const int kfPre = featureIndex(kPreExt, preLoc->nx + preLoc->halo.lt + preLoc->halo.rt, preLoc->ny + preLoc->halo.dn + preLoc->halo.up, preLoc->nf);

      const int kxPreGlobalExt = kxPreExt + preLoc->kx0;
      const int kyPreGlobalExt = kyPreExt + preLoc->ky0;
      if(kxPreGlobalExt < preLoc->halo.lt || kxPreGlobalExt >= preLoc->nxGlobal + preLoc->halo.lt ||
         kyPreGlobalExt < preLoc->halo.up || kyPreGlobalExt >= preLoc->nyGlobal + preLoc->halo.up){
         continue;
      }

      float a = activity->data[kPreExt];
      //if (a == 0.0f) continue;

      //Convert stored global extended index into local extended index
      int postGlobalExtIdx = postIdxData[kPreExt];

      //Make sure the index is in bounds
      assert(postGlobalExtIdx >= 0 && postGlobalExtIdx <
            (postLoc->nxGlobal + postLoc->halo.lt + postLoc->halo.rt) * 
            (postLoc->nyGlobal + postLoc->halo.up + postLoc->halo.dn) * 
            postLoc->nf);

      const int kxPostGlobalExt = kxPos(postGlobalExtIdx, postLoc->nxGlobal + postLoc->halo.lt + postLoc->halo.rt, postLoc->nyGlobal + postLoc->halo.dn + postLoc->halo.up, postLoc->nf);
      const int kyPostGlobalExt = kyPos(postGlobalExtIdx, postLoc->nxGlobal + postLoc->halo.lt + postLoc->halo.rt, postLoc->nyGlobal + postLoc->halo.dn + postLoc->halo.up, postLoc->nf);
      const int kfPost = featureIndex(postGlobalExtIdx, postLoc->nxGlobal + postLoc->halo.lt + postLoc->halo.rt, postLoc->nyGlobal + postLoc->halo.dn + postLoc->halo.up, postLoc->nf);

      const int kxPostLocalRes = kxPostGlobalExt - postLoc->kx0 - postLoc->halo.lt;
      const int kyPostLocalRes = kyPostGlobalExt - postLoc->ky0 - postLoc->halo.up;
      if(kxPostLocalRes < 0 || kxPostLocalRes >= postLoc->nx|| 
         kyPostLocalRes < 0 || kyPostLocalRes >= postLoc->ny){
         continue;
      }

      const int kPostLocalRes = kIndex(kxPostLocalRes, kyPostLocalRes, kfPost, postLoc->nx, postLoc->ny, postLoc->nf);

      //printf("address: %p  kPreExt: %d idxout: %d, activity: %f\n", &(postIdxData[kPreExt]), kPreExt, kPostLocalRes, a);

      //If we're using thread_gSyn, set this here
      pvdata_t * gSynPatchHead;
#ifdef PV_USE_OPENMP_THREADS
      if(thread_gSyn){
         int ti = omp_get_thread_num();
         gSynPatchHead = thread_gSyn[ti];
      }
      else{
         gSynPatchHead = post->getChannel(getChannel());
      }
#else // PV_USE_OPENMP_THREADS
      gSynPatchHead = post->getChannel(getChannel());
#endif // PV_USE_OPENMP_THREADS
      gSynPatchHead[kPostLocalRes] = a;
   }

#ifdef PV_USE_OPENMP_THREADS
   //Set back into gSyn
   if(thread_gSyn){
      pvdata_t * gSynPatchHead = post->getChannel(getChannel());
      int numNeurons = post->getNumNeurons();
      //Looping over neurons first to be thread safe
#pragma omp parallel for
      for(int ni = 0; ni < numNeurons; ni++){
         for(int ti = 0; ti < parent->getNumThreads(); ti++){
            if(gSynPatchHead[ni] < fabs(thread_gSyn[ti][ni])){
               gSynPatchHead[ni] = thread_gSyn[ti][ni];
            }
         }
      }
   }
#endif
   return PV_SUCCESS;
}

int TransposePoolingConn::checkpointRead(const char * cpDir, double * timeptr) {
   return PV_SUCCESS;
}

int TransposePoolingConn::checkpointWrite(const char * cpDir) {
   return PV_SUCCESS;
}





} // end namespace PV
