/* CloneConn.cpp
 *
 * Created on: May 23, 2011
 *     Author: peteschultz
 */

#include "CloneConn.hpp"

namespace PV {

CloneConn::CloneConn(){
   initialize_base();
}

CloneConn::CloneConn(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

int CloneConn::initialize_base() {
   originalConn = NULL;
   return PV_SUCCESS;
}

int CloneConn::initialize(const char * name, HyPerCol * hc) {
   int status = HyPerConn::initialize(name, hc, NULL, NULL);
   return status;
}

int CloneConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_originalConnName(ioFlag);
   return status;
}

void CloneConn::ioParam_writeStep(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "writeStep");
      writeStep = -1;
   }   
}

void CloneConn::ioParam_weightInitType(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryStringParameter(name, "weightInitType", NULL);
   }
}

void CloneConn::ioParam_normalizeMethod(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryStringParameter(name, "normalizeMethod", NULL);
   }
}

void CloneConn::ioParam_originalConnName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "originalConnName", &originalConnName);
}

int CloneConn::setWeightInitializer() {
   weightInitializer = new InitCloneKernelWeights();
   return PV_SUCCESS;
}

int CloneConn::constructWeights() {
   int status = PV_SUCCESS;

   // CloneConn::ioParam_shrinkPatches does nothing; shrinkPatches_flag is set in communicateInitInfo()

   // if( status == PV_SUCCESS ) status = setPatchSize(NULL);
   if( status == PV_SUCCESS ) status = setPatchStrides();

   wPatches = this->originalConn->get_wPatches();
   wDataStart = this->originalConn->get_wDataStart();
   gSynPatchStart = this->originalConn->getGSynPatchStart();
   aPostOffset = this->originalConn->getAPostOffset();
   dwDataStart = this->originalConn->get_dwDataStart();

   // Don't call initPlasticityPatches since plasticityFlag is always false.
   // Don't call shrinkPatches() since the original connection will have already shrunk patches
   return status;
}

void CloneConn::constructWeightsOutOfMemory() {
   connOutOfMemory("CloneConn::constructWeightsOutOfMemory()");
}

int CloneConn::createAxonalArbors(int arborId) {
   return PV_SUCCESS;
}

//void CloneConn::initPatchToDataLUT() {
//   assert(patch2datalookuptable==NULL);
//   patch2datalookuptable = originalConn->getPatchToDataLUT();
//}

PVPatch *** CloneConn::initializeWeights(PVPatch *** patches, pvdata_t ** dataStart) {
   return patches;
   // nothing to be done as the weight patches point to originalConn's space.
}

// We override many read-methods because CloneConn will use
// originalConn's values.  communicateInitInfo will check if the associated
// parameters exist in params for theCloneKernelConn group, and whether they
// are consistent with the originalConn parameters.
// If consistent, issue a warning that the param is unnecessary and continue.
// If inconsistent, issue an error and quit.
// We can't do that in the read-method because we can't be sure originalConn
// has set its own parameter yet (or even if it's been instantiated),
// and in theory originalConn could be a subclass that determines
// the parameter some way other than reading its own parameter
// group's param directly.

void CloneConn::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights");
   }
   // During the communication phase, sharedWeights will be copied from originalConn
}

void CloneConn::ioParam_shrinkPatches(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "shrinkPatches");
   }
   // During the communication phase, shrinkPatches_flag will be copied from originalConn
}

void CloneConn::ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "numAxonalArbors");
   }
   // During the communication phase, numAxonalArbors will be copied from originalConn
}

void CloneConn::ioParam_plasticityFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      plasticityFlag = false; // CloneConn updates automatically, since it's done using pointer magic.
      parent->parameters()->handleUnnecessaryParameter(name, "plasticityFlag", plasticityFlag);
   }
}

void CloneConn::ioParam_keepKernelsSynchronized(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      keepKernelsSynchronized_flag = false;
      // CloneConns do not have to synchronize because the pointers keep them synchronized whenever the original is.
      // We override this method because sharedWeights is not determined when this function is called.
   }
}

void CloneConn::ioParam_nxp(enum ParamsIOFlag ioFlag) {
   // During the communication phase, nxp will be copied from originalConn
}
void CloneConn::ioParam_nyp(enum ParamsIOFlag ioFlag) {
   // During the communication phase, nyp will be copied from originalConn
}

void CloneConn::ioParam_nfp(enum ParamsIOFlag ioFlag) {
   // During the communication phase, nfp will be copied from originalConn
}

void CloneConn::ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      initializeFromCheckpointFlag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "initializeFromCheckpointFlag");
   }
   // CloneConn does not checkpoint, so we don't need initializeFromCheckpointFlag
}

void CloneConn::ioParam_writeCompressedWeights(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      initializeFromCheckpointFlag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "writeCompressedWeights");
   }
   // CloneConn does not write during outputState, so we don't need writeCompressedWeights
}

void CloneConn::ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      initializeFromCheckpointFlag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "writeCompressedWeights");
   }
   // CloneConn does not checkpoint, so we don't need writeCompressedCheckpoints
}

int CloneConn::communicateInitInfo() {
   // Need to set originalConn before calling HyPerConn::communicate, since HyPerConn::communicate calls setPatchSize, which needs originalConn.
   BaseConnection * originalConnBase = parent->getConnFromName(originalConnName);
   if (originalConnBase == NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "CloneConn \"%s\" error: originalConnName \"%s\" is not a connection in the column.\n",
               name, originalConnName);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   originalConn = dynamic_cast<HyPerConn *>(originalConnBase);
   if (originalConn == NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "CloneConn \"%s\" error: originalConnName \"%s\" is not a HyPerConn or HyPerConn-derived class.\n",
               name, originalConnName);
      }
   }
   if (!originalConn->getInitInfoCommunicatedFlag()) {
      if (parent->columnId()==0) {
         const char * connectiontype = this->getKeyword();
         printf("%s \"%s\" must wait until original connection \"%s\" has finished its communicateInitInfo stage.\n", connectiontype, name, originalConn->getName());
      }
      return PV_POSTPONE;
   }

   // Copy some parameters from originalConn.  Check if parameters exist is
   // the clone's param group, and issue a warning (if the param has the right
   // value) or an error (if it has the wrong value).
   int status = cloneParameters();

   status = HyPerConn::communicateInitInfo();
   if (status != PV_SUCCESS) return status;

   //Don't allocate post, just grab in allocate from orig
   if(needPost){
      originalConn->setNeedPost(true);
   }

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   if((updateGSynFromPostPerspective && receiveGpu) || allocPostDeviceWeights){
      originalConn->setAllocPostDeviceWeights();
   }
   if((!updateGSynFromPostPerspective && receiveGpu) || allocDeviceWeights){
      originalConn->setAllocDeviceWeights();
   }
#endif


   // Presynaptic layers of the CloneConn and its original conn must have the same size, or the patches won't line up with each other.
   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const PVLayerLoc * origPreLoc = originalConn->preSynapticLayer()->getLayerLoc();

   if (preLoc->nx != origPreLoc->nx || preLoc->ny != origPreLoc->ny || preLoc->nf != origPreLoc->nf ) {
      if (parent->icCommunicator()->commRank()==0) {
         const char * classname = this->getKeyword();
         fprintf(stderr, "%s \"%s\" error in rank %d process: CloneConn and originalConn \"%s\" must have presynaptic layers with the same nx,ny,nf.\n",
               classname, name, parent->columnId(), originalConn->getName());
         fprintf(stderr, "{nx=%d, ny=%d, nf=%d} versus {nx=%d, ny=%d, nf=%d}\n",
                 preLoc->nx, preLoc->ny, preLoc->nf, origPreLoc->nx, origPreLoc->ny, origPreLoc->nf);
      }
      abort();
   }

   // Make sure the original's and the clone's margin widths stay equal
   originalConn->preSynapticLayer()->synchronizeMarginWidth(pre);
   pre->synchronizeMarginWidth(originalConn->preSynapticLayer());

   //// Make sure the original's and the clone's margin widths stay equal
   //originalConn->postSynapticLayer()->synchronizeMarginWidth(post);
   //post->synchronizeMarginWidth(originalConn->postSynapticLayer());

   //Redudant read in case it's a clone of a clone

   return status;
}

//#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
//void CloneConn::setAllocDeviceWeights(){
//   originalConn->setAllocDeviceWeights();
//   cloneNeedDeviceWeights = true;
//   allocDeviceWeights = false;
//}
//#endif


//Overwriting HyPerConn's allocate, since it needs to just grab postConn and preToPostActivity from orig conn
int CloneConn::allocatePostConn(){
   postConn = originalConn->postConn;
   //postToPreActivity = originalConn->postToPreActivity;
   return PV_SUCCESS;
}

int CloneConn::allocateDataStructures() {
   if (!originalConn->getDataStructuresAllocatedFlag()) {
      if (parent->columnId()==0) {
         const char * connectiontype = this->getKeyword();
         printf("%s \"%s\" must wait until original connection \"%s\" has finished its communicateInitInfo stage.\n", connectiontype, name, originalConn->getName());
      }
      return PV_POSTPONE;
   }
   int status = HyPerConn::allocateDataStructures();
   return status;
}


#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
//Device buffers live in origConn
int CloneConn::allocateDeviceWeights(){
   return PV_SUCCESS;
}
int CloneConn::allocatePostDeviceWeights(){
   return PV_SUCCESS;
}
#endif


int CloneConn::setPatchSize() {
   assert(originalConn);
   nxp = originalConn->xPatchSize();
   nyp = originalConn->yPatchSize();
   nfp = originalConn->fPatchSize();
   parent->parameters()->handleUnnecessaryParameter(name, "nxp", nxp);
   parent->parameters()->handleUnnecessaryParameter(name, "nyp", nyp);
   parent->parameters()->handleUnnecessaryParameter(name, "nfp", nfp);
   return PV_SUCCESS;
}

int CloneConn::cloneParameters() {
   // Copy sharedWeights, numAxonalArborLists, shrinkPatches_flag from originalConn

   PVParams * params = parent->parameters();

   sharedWeights = originalConn->usingSharedWeights();
   params->handleUnnecessaryParameter(name, "sharedWeights", sharedWeights);

   numAxonalArborLists = originalConn->numberOfAxonalArborLists();
   params->handleUnnecessaryParameter(name, "numAxonalArbors", numAxonalArborLists);

   shrinkPatches_flag = originalConn->getShrinkPatches_flag();
   parent->parameters()->handleUnnecessaryParameter(name, "shrinkPatches", shrinkPatches_flag);
   return PV_SUCCESS;
}

int CloneConn::updateState(double time, double dt) {
   update_timer->start();

   lastUpdateTime = originalConn->getLastUpdateTime();

   update_timer->stop();
   return PV_SUCCESS;
}

int CloneConn::finalizeUpdate(double timed, double dt){
   //Orig conn is in charge of calling finalizeUpdate for postConn.
   return PV_SUCCESS;
}

int CloneConn::deleteWeights() {
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

CloneConn::~CloneConn() {
   free(originalConnName);
   deleteWeights();
   postConn = NULL;
   postToPreActivity = NULL;
}

BaseObject * createCloneConn(char const * name, HyPerCol * hc) {
   return hc ? new CloneConn(name, hc) : NULL;
}

} // end namespace PV
