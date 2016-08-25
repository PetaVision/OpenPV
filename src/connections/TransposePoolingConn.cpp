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
   // It is okay for either of weightInitializer or weightNormalizer to be null at this point, either because we're in a subclass that doesn't need it, or because we are allowing for // backward compatibility.
   // The two lines needs to be before the call to BaseConnection::initialize, because that function calls ioParamsFillGroup,
   // which will call ioParam_weightInitType and ioParam_normalizeMethod, which for reasons of backward compatibility
   // will create the initializer and normalizer if those member variables are null.
   this->weightInitializer = NULL;
   this->normalizer = NULL;

   int status = BaseConnection::initialize(name, hc); // BaseConnection should *NOT* take weightInitializer or weightNormalizer as arguments, as it does not know about InitWeights or NormalizeBase

   assert(parent);
   PVParams * inputParams = parent->parameters();

   //set accumulateFunctionPointer
   assert(!inputParams->presentAndNotBeenRead(name, "pvpatchAccumulateType"));
   switch (poolingType) {
#ifdef OBSOLETE // Marked obsolete May 3, 2016.  HyPerConn defines HyPerConnAccumulateType and PoolingConn defines PoolingType
   case ACCUMULATE_CONVOLVE:
      pvError() << "ACCUMULATE_CONVOLVE not allowed in TransposePoolingConn\n";
      break;
   case ACCUMULATE_STOCHASTIC:
      pvError() << "ACCUMULATE_STOCASTIC not allowed in TransposePoolingConn\n";
      break;
#endif // OBSOLETE // Marked obsolete May 3, 2016.  HyPerConn defines HyPerConnAccumulateType and PoolingConn defines PoolingType
   case PoolingConn::MAX:
      accumulateFunctionPointer = &pvpatch_max_pooling;
      accumulateFunctionFromPostPointer = &pvpatch_max_pooling_from_post;
      break;
   case PoolingConn::SUM:
      accumulateFunctionPointer = &pvpatch_sum_pooling;
      accumulateFunctionFromPostPointer = &pvpatch_sumpooling_from_post;
      break;
   case PoolingConn::AVG:
      accumulateFunctionPointer = &pvpatch_sum_pooling;
      accumulateFunctionFromPostPointer = &pvpatch_sumpooling_from_post;
      break;
   default:
      pvAssert(0);
      break;
   }

   //ioAppend = parent->getCheckpointReadFlag();

   this->io_timer     = new Timer(getName(), "conn", "io     ");
   this->update_timer = new Timer(getName(), "conn", "update ");

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

void TransposePoolingConn::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) {
      // make sure that TransposePoolingConn always checks if its originalConn has updated
      triggerFlag = false;
      triggerLayerName = NULL;
      parent->parameters()->handleUnnecessaryParameter(name, "triggerFlag", triggerFlag);
      parent->parameters()->handleUnnecessaryStringParameter(name, "triggerLayerName", NULL);
   }
}

// TODO: ioParam_pvpatchAccumulateType and unsetAccumulateType are copied from PV::PoolingConn.
// Can we make TransposePoolingConn a derived class of PoolingConn?  (TransposeConn is a derived class of HyPerConn.)
void TransposePoolingConn::ioParam_pvpatchAccumulateType(enum ParamsIOFlag ioFlag) {
   PVParams * params = parent->parameters();

   parent->ioParamStringRequired(ioFlag, name, "pvpatchAccumulateType", &pvpatchAccumulateTypeString);
   if (ioFlag==PARAMS_IO_READ) {
      if (pvpatchAccumulateTypeString==NULL) {
         unsetAccumulateType();
         return;
      }
      // Convert string to lowercase so that capitalization doesn't matter.
      for (char * c = pvpatchAccumulateTypeString; *c!='\0'; c++) {
         *c = (char) tolower((int) *c);
      }

      if ((strcmp(pvpatchAccumulateTypeString,"maxpooling")==0) ||
           (strcmp(pvpatchAccumulateTypeString,"max_pooling")==0) ||
           (strcmp(pvpatchAccumulateTypeString,"max pooling")==0)) {
         poolingType = PoolingConn::MAX;
      }
      else if ((strcmp(pvpatchAccumulateTypeString,"sumpooling")==0) ||
           (strcmp(pvpatchAccumulateTypeString,"sum_pooling")==0)  ||
           (strcmp(pvpatchAccumulateTypeString,"sum pooling")==0)) {
         poolingType = PoolingConn::SUM;
      }
      else if ((strcmp(pvpatchAccumulateTypeString,"avgpooling")==0) ||
           (strcmp(pvpatchAccumulateTypeString,"avg_pooling")==0)  ||
           (strcmp(pvpatchAccumulateTypeString,"avg pooling")==0)) {
         poolingType = PoolingConn::AVG;
      }
      else {
         unsetAccumulateType();
      }
   }
}

void TransposePoolingConn::unsetAccumulateType() {
   if (parent->columnId()==0) {
      pvErrorNoExit(errorMessage);
      if (pvpatchAccumulateTypeString) {
         errorMessage.printf("%s: pvpatchAccumulateType \"%s\" is unrecognized.",
               getDescription_c(), pvpatchAccumulateTypeString);
      }
      else {
         errorMessage.printf("%s: pvpatchAccumulateType NULL is unrecognized.",
               getDescription_c());
      }
      errorMessage.printf("  Allowed values are \"maxpooling\", \"sumpooling\", or \"avgpooling\".");
   }
   MPI_Barrier(parent->getCommunicator()->communicator());
   exit(EXIT_FAILURE);
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

int TransposePoolingConn::communicateInitInfo() {
   int status = PV_SUCCESS;
   BaseConnection * originalConnBase = parent->getConnFromName(this->originalConnName);
   if (originalConnBase==NULL) {
      if (parent->columnId()==0) {
         pvErrorNoExit().printf("%s: originalConnName \"%s\" does not refer to any connection in the column.\n", getDescription_c(), this->originalConnName);
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   originalConn = dynamic_cast<PoolingConn *>(originalConnBase);
   if (originalConn == NULL) {
      if (parent->columnId()==0) {
         pvErrorNoExit().printf("%s: originalConnName \"%s\" is not a PoolingConn.\n", getDescription_c(), originalConnName);
         status = PV_FAILURE;
      }
   }
   if (status != PV_SUCCESS) return status;

   if (!originalConn->getInitInfoCommunicatedFlag()) {
      if (parent->columnId()==0) {
         pvInfo().printf("%s must wait until original connection \"%s\" has finished its communicateInitInfo stage.\n", getDescription_c(), originalConn->getName());
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
         pvErrorNoExit().printf("TransposePoolingConn \"%s\": original conn \"%s\" has shrinkPatches set to true.  TransposePoolingConn has not been implemented for that case.\n", name, originalConn->getName());
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   status = HyPerConn::communicateInitInfo(); // calls setPatchSize()
   if (status != PV_SUCCESS) return status;

   if(!originalConn->needPostIndex() && poolingType == PoolingConn::MAX){
      if (parent->columnId()==0) {
         pvErrorNoExit().printf("TransposePoolingConn \"%s\": original pooling conn \"%s\" needs to have a postIndexLayer if unmax pooling.\n", name, originalConnName);
         status = PV_FAILURE;
      }
   }
   if (status != PV_SUCCESS) return status;

   //Check post layer phases to make sure it matches
   if(originalConn->postSynapticLayer()->getPhase() >= post->getPhase()){
      pvWarn().printf("TransposePoolingConn \"%s\": originalConn's post layer phase is greater or equal than this layer's post. Behavior undefined.\n", name);
   }

   if(originalConn->getPoolingType() != poolingType){
      pvError().printf("TransposePoolingConn \"%s\" error: originalConn accumulateType does not match this layer's accumulate type.\n", name);
   }

   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const PVLayerLoc * origPostLoc = originalConn->postSynapticLayer()->getLayerLoc();
   if (preLoc->nx != origPostLoc->nx || preLoc->ny != origPostLoc->ny || preLoc->nf != origPostLoc->nf) {
      if (parent->columnId()==0) {
         pvErrorNoExit(errorMessage);
         errorMessage.printf("%s: transpose's pre layer and original connection's post layer must have the same dimensions.\n", getDescription_c());
         errorMessage.printf("    (x=%d, y=%d, f=%d) versus (x=%d, y=%d, f=%d).\n", preLoc->nx, preLoc->ny, preLoc->nf, origPostLoc->nx, origPostLoc->ny, origPostLoc->nf);
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   const PVLayerLoc * postLoc = pre->getLayerLoc();
   const PVLayerLoc * origPreLoc = originalConn->postSynapticLayer()->getLayerLoc();
   if (postLoc->nx != origPreLoc->nx || postLoc->ny != origPreLoc->ny || postLoc->nf != origPreLoc->nf) {
      if (parent->columnId()==0) {
         pvErrorNoExit(errorMessage);
         errorMessage.printf("%s: transpose's post layer and original connection's pre layer must have the same dimensions.\n", getDescription_c());
         errorMessage.printf("    (x=%d, y=%d, f=%d) versus (x=%d, y=%d, f=%d).\n", postLoc->nx, postLoc->ny, postLoc->nf, origPreLoc->nx, origPreLoc->ny, origPreLoc->nf);
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   originalConn->setNeedPost(true);
   originalConn->setNeedAllocPostWeights(false);

   //Synchronize margines of this post and orig pre, and vice versa
   originalConn->preSynapticLayer()->synchronizeMarginWidth(post);
   post->synchronizeMarginWidth(originalConn->preSynapticLayer());

   originalConn->postSynapticLayer()->synchronizeMarginWidth(pre);
   pre->synchronizeMarginWidth(originalConn->postSynapticLayer());

   if(poolingType == PoolingConn::MAX){
      //Sync pre margins
      //Synchronize margines of this post and the postIndexLayer, and vice versa
      pre->synchronizeMarginWidth(originalConn->getPostIndexLayer());
      originalConn->getPostIndexLayer()->synchronizeMarginWidth(pre);

      //Need to tell postIndexLayer the number of delays needed by this connection
      int allowedDelay = originalConn->getPostIndexLayer()->increaseDelayLevels(getDelayArraySize());
      if( allowedDelay < getDelayArraySize()) {
         if( this->getParent()->columnId() == 0 ) {
            pvErrorNoExit().printf("%s: attempt to set delay to %d, but the maximum allowed delay is %d.  Exiting\n", this->getDescription_c(), getDelayArraySize(), allowedDelay);
         }
         exit(EXIT_FAILURE);
      }
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
         pvInfo().printf("%s must wait until original connection \"%s\" has finished its allocateDataStructures stage.\n", getDescription_c(), originalConn->getName());
      }
      return PV_POSTPONE;
   }

   bool tempNeedPost = false;
   //Turn off need post so postConn doesn't get allocated
   if(needPost){
      needPost = false;
      postConn = originalConn;
      //TODO this buffer is only needed if this transpose conn is receiving from post
      originalConn->postConn->allocatePostToPreBuffer();
      postToPreActivity = originalConn->postConn->getPostToPreActivity();
      tempNeedPost = true;
   }
   int status = HyPerConn::allocateDataStructures();
   if (status != PV_SUCCESS) { return status; }

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
   lastTimeUpdateCalled = time;
   return PV_SUCCESS;
}

double TransposePoolingConn::computeNewWeightUpdateTime(double time, double currentUpdateTime) {
   return weightUpdateTime; // TransposePoolingConn does not use weightUpdateTime to determine when to update
}

int TransposePoolingConn::deliverPostsynapticPerspective(PVLayerCube const * activity, int arborID) {
   pvError() << "Delivering from PostSynapticPerspective for TransposePoolingConn not implented yet\n";
   return PV_FAILURE; // suppresses warning in compilers that don't recognize pvError always exits.
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

   //Grab postIdxLayer's data
   pvdata_t * postIdxData = nullptr;
   if(poolingType == PoolingConn::MAX){
      PoolingIndexLayer* postIndexLayer = originalConn->getPostIndexLayer();
      assert(postIndexLayer);
      //Make sure this layer is an integer layer
      assert(postIndexLayer->getDataType() == PV_INT);
      DataStore * store = postIndexLayer->getPublisher()->dataStore();
      int delay = getDelay(arborID);

      postIdxData = store->buffer(Communicator::LOCAL, delay);
   }

   for(int b = 0; b < parent->getNBatch(); b++){
      pvdata_t * activityBatch = activity->data + b * (preLoc->nx + preLoc->halo.rt + preLoc->halo.lt) * (preLoc->ny + preLoc->halo.up + preLoc->halo.dn) * preLoc->nf;
      pvdata_t * gSynPatchHeadBatch = post->getChannel(getChannel()) + b * postLoc->nx * postLoc->ny * postLoc->nf;
      pvdata_t * postIdxDataBatch = nullptr;
      if(poolingType == PoolingConn::MAX){
         postIdxDataBatch = postIdxData + b * originalConn->getPostIndexLayer()->getNumExtended();
      }

      unsigned int * activeIndicesBatch = NULL;
      if(activity->isSparse){
         activeIndicesBatch = activity->activeIndices + b * (preLoc->nx + preLoc->halo.rt + preLoc->halo.lt) * (preLoc->ny + preLoc->halo.up + preLoc->halo.dn) * preLoc->nf;
      }

      int numLoop;
      if(activity->isSparse){
         numLoop = activity->numActive[b];
      }
      else{
         numLoop = numExtended;
      }

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


#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
      for (int loopIndex = 0; loopIndex < numLoop; loopIndex++) {
         int kPreExt;
         if(activity->isSparse){
            kPreExt = activeIndicesBatch[loopIndex];
         }
         else{
            kPreExt = loopIndex;
         }

         float a = activityBatch[kPreExt];
         if (a == 0.0f) continue;

         //If we're using thread_gSyn, set this here
         pvdata_t * gSynPatchHead;
#ifdef PV_USE_OPENMP_THREADS
         if(thread_gSyn){
            int ti = omp_get_thread_num();
            gSynPatchHead = thread_gSyn[ti];
         }
         else{
            gSynPatchHead = gSynPatchHeadBatch;
         }
#else // PV_USE_OPENMP_THREADS
         gSynPatchHead = gSynPatchHeadBatch;
#endif // PV_USE_OPENMP_THREADS

         const int kxPreExt = kxPos(kPreExt, preLoc->nx + preLoc->halo.lt + preLoc->halo.rt, preLoc->ny + preLoc->halo.dn + preLoc->halo.up, preLoc->nf);
         const int kyPreExt = kyPos(kPreExt, preLoc->nx + preLoc->halo.lt + preLoc->halo.rt, preLoc->ny + preLoc->halo.dn + preLoc->halo.up, preLoc->nf);
         const int kfPre = featureIndex(kPreExt, preLoc->nx + preLoc->halo.lt + preLoc->halo.rt, preLoc->ny + preLoc->halo.dn + preLoc->halo.up, preLoc->nf);

         if(poolingType == PoolingConn::MAX){
            const int kxPreGlobalExt = kxPreExt + preLoc->kx0;
            const int kyPreGlobalExt = kyPreExt + preLoc->ky0;
            if(kxPreGlobalExt < preLoc->halo.lt || kxPreGlobalExt >= preLoc->nxGlobal + preLoc->halo.lt ||
               kyPreGlobalExt < preLoc->halo.up || kyPreGlobalExt >= preLoc->nyGlobal + preLoc->halo.up){
               continue;
            }

            //Convert stored global extended index into local extended index
            int postGlobalExtIdx = (int) postIdxDataBatch[kPreExt];

            // If all inputs are zero and input layer is sparse, postGlobalExtIdx will still be -1.
            if(postGlobalExtIdx == -1) { continue; }

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
            gSynPatchHead[kPostLocalRes] = a;
         }
         else{
            PVPatch * weights = getWeights(kPreExt, arborID);
            const int nk = weights->nx * fPatchSize();
            const int ny = weights->ny;
            pvgsyndata_t * postPatchStart = gSynPatchHead + getGSynPatchStart(kPreExt, arborID);
            const int sy  = getPostNonextStrides()->sy;       // stride in layer

            int offset = kfPre;
            int sf = fPatchSize();

            pvwdata_t w = 1.0;
            if(poolingType == PoolingConn::MAX){
              //float relative_XScale = pow(2, (post->getXScale() - pre->getXScale()));
              //float relative_YScale = pow(2, (post->getYScale() - pre->getYScale()));
              //w = 1.0/(nxp*nyp*relative_XScale*relative_YScale);
              w = 1.0;
            }
            else if(poolingType == PoolingConn::MAX){
              float relative_XScale = pow(2, (post->getXScale() - pre->getXScale()));
              float relative_YScale = pow(2, (post->getYScale() - pre->getYScale()));
              float normVal = nxp*nyp;
              w = 1.0/normVal;
            }
            void* auxPtr = NULL;
            for (int y = 0; y < ny; y++) {
               (accumulateFunctionPointer)(0, nk, postPatchStart + y*sy + offset, a, &w, auxPtr, sf);
            }
         }
      }

#ifdef PV_USE_OPENMP_THREADS
      //Set back into gSyn
      if(thread_gSyn){
         pvdata_t * gSynPatchHead = gSynPatchHeadBatch;
         int numNeurons = post->getNumNeurons();
         //Looping over neurons first to be thread safe
#pragma omp parallel for
         for(int ni = 0; ni < numNeurons; ni++){
            if(poolingType == PoolingConn::MAX){
               //Grab maxumum magnitude of thread_gSyn and set that value
               float maxMag = -INFINITY;
               int maxMagIdx = -1;
               for(int ti = 0; ti < parent->getNumThreads(); ti++){
                  if(maxMag < fabs(thread_gSyn[ti][ni])){
                     maxMag = fabs(thread_gSyn[ti][ni]);
                     maxMagIdx = ti;
                  }
               }
               assert(maxMagIdx >= 0);
               gSynPatchHead[ni] = thread_gSyn[maxMagIdx][ni];
            }
            else{
               for(int ti = 0; ti < parent->getNumThreads(); ti++){
                  gSynPatchHead[ni] += thread_gSyn[ti][ni];
               }
            }
         }
      }
#endif
   }
   return PV_SUCCESS;
}

int TransposePoolingConn::checkpointRead(const char * cpDir, double * timeptr) {
   return PV_SUCCESS;
}

int TransposePoolingConn::checkpointWrite(const char * cpDir) {
   return PV_SUCCESS;
}

} // end namespace PV
