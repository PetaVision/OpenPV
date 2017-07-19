/* CloneConn.cpp
 *
 * Created on: May 23, 2011
 *     Author: peteschultz
 */

#include "CloneConn.hpp"

namespace PV {

CloneConn::CloneConn() { initialize_base(); }

CloneConn::CloneConn(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

int CloneConn::initialize_base() {
   originalConn = NULL;
   return PV_SUCCESS;
}

int CloneConn::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerConn::initialize(name, hc);
   return status;
}

int CloneConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_originalConnName(ioFlag);
   return status;
}

void CloneConn::ioParam_writeStep(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "writeStep");
      writeStep = -1;
   }
}

void CloneConn::ioParam_weightInitType(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryStringParameter(name, "weightInitType", NULL);
      weightInitializer = nullptr;
   }
}

void CloneConn::ioParam_normalizeMethod(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryStringParameter(name, "normalizeMethod", NULL);
   }
}

void CloneConn::ioParam_originalConnName(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamStringRequired(ioFlag, name, "originalConnName", &originalConnName);
}

int CloneConn::registerData(Checkpointer *checkpointer) {
   registerTimers(checkpointer);
   return PV_SUCCESS;
}

int CloneConn::constructWeights() {
   int status = setPatchStrides();

   wPatches       = this->originalConn->get_wPatches();
   wDataStart     = this->originalConn->get_wDataStart();
   gSynPatchStart = this->originalConn->getGSynPatchStart();
   aPostOffset    = this->originalConn->getAPostOffset();
   dwDataStart    = this->originalConn->get_dwDataStart();

   // Don't call initPlasticityPatches since plasticityFlag is always false.
   // Don't call shrinkPatches() since the original connection will have already shrunk patches
   return status;
}

void CloneConn::constructWeightsOutOfMemory() {
   Fatal().printf(
         "Out of memory error in CloneConn::constructWeightsOutOfMemory() for \"%s\"\n", name);
}

int CloneConn::createAxonalArbors(int arborId) { return PV_SUCCESS; }

PVPatch ***CloneConn::initializeWeights(PVPatch ***patches, float **dataStart) {
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
      plasticityFlag = false;
      // CloneConn updates automatically, since it's done using pointer magic.
      parent->parameters()->handleUnnecessaryParameter(name, "plasticityFlag", plasticityFlag);
   }
}

void CloneConn::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   triggerFlag      = false;
   triggerLayerName = NULL;
   parent->parameters()->handleUnnecessaryStringParameter(name, "triggerLayerName");
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

int CloneConn::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   // Need to set originalConn before calling HyPerConn::communicate, since HyPerConn::communicate
   // calls setPatchSize, which needs originalConn.
   originalConn = message->lookup<HyPerConn>(std::string(originalConnName));
   if (originalConn == NULL) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "%s: originalConnName \"%s\" is not a HyPerConn or HyPerConn-derived object.\n",
               getDescription_c(),
               originalConnName);
      }
   }
   if (!originalConn->getInitInfoCommunicatedFlag()) {
      if (parent->columnId() == 0) {
         InfoLog().printf(
               "%s must wait until original connection \"%s\" has finished its communicateInitInfo "
               "stage.\n",
               getDescription_c(),
               originalConn->getName());
      }
      return PV_POSTPONE;
   }

   // Copy some parameters from originalConn.  Check if parameters exist is
   // the clone's param group, and issue a warning (if the param has the right
   // value) or an error (if it has the wrong value).
   int status = cloneParameters();

   status = HyPerConn::communicateInitInfo(message);
   if (status != PV_SUCCESS)
      return status;

   // Don't allocate post, just grab in allocate from orig
   if (needPost) {
      originalConn->setNeedPost();
   }

#ifdef PV_USE_CUDA
   if ((updateGSynFromPostPerspective && receiveGpu) || allocPostDeviceWeights) {
      originalConn->setAllocPostDeviceWeights();
   }
   if ((!updateGSynFromPostPerspective && receiveGpu) || allocDeviceWeights) {
      originalConn->setAllocDeviceWeights();
   }
#endif

   // Presynaptic layers of the CloneConn and its original conn must have the same size, or the
   // patches won't line up with each other.
   const PVLayerLoc *preLoc     = pre->getLayerLoc();
   const PVLayerLoc *origPreLoc = originalConn->preSynapticLayer()->getLayerLoc();

   if (preLoc->nx != origPreLoc->nx || preLoc->ny != origPreLoc->ny
       || preLoc->nf != origPreLoc->nf) {
      if (parent->getCommunicator()->commRank() == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "%s: CloneConn and originalConn \"%s\" must have presynaptic layers with the same "
               "nx,ny,nf.\n",
               getDescription_c(),
               parent->columnId(),
               originalConn->getName());
         errorMessage.printf(
               "{nx=%d, ny=%d, nf=%d} versus {nx=%d, ny=%d, nf=%d}\n",
               preLoc->nx,
               preLoc->ny,
               preLoc->nf,
               origPreLoc->nx,
               origPreLoc->ny,
               origPreLoc->nf);
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      abort();
   }

   // Make sure the original's and the clone's margin widths stay equal
   originalConn->preSynapticLayer()->synchronizeMarginWidth(pre);
   pre->synchronizeMarginWidth(originalConn->preSynapticLayer());

   // Make sure the original's and the clone's margin widths stay equal
   // Only if this layer receives from post for patch to data LUT
   if (getUpdateGSynFromPostPerspective()) {
      originalConn->postSynapticLayer()->synchronizeMarginWidth(post);
      post->synchronizeMarginWidth(originalConn->postSynapticLayer());
   }
   // Redudant read in case it's a clone of a clone

   return status;
}

// Overwriting HyPerConn's allocate, since it needs to just grab postConn and preToPostActivity from
// orig conn
int CloneConn::allocatePostConn() {
   postConn = originalConn->postConn;
   return PV_SUCCESS;
}

int CloneConn::allocateDataStructures() {
   if (!originalConn->getDataStructuresAllocatedFlag()) {
      if (parent->columnId() == 0) {
         InfoLog().printf(
               "%s must wait until original connection \"%s\" has finished its communicateInitInfo "
               "stage.\n",
               getDescription_c(),
               originalConn->getName());
      }
      return PV_POSTPONE;
   }
   int status = HyPerConn::allocateDataStructures();
   return status;
}

#ifdef PV_USE_CUDA
// Device buffers live in origConn
int CloneConn::allocateDeviceWeights() { return PV_SUCCESS; }
int CloneConn::allocatePostDeviceWeights() { return PV_SUCCESS; }
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

   PVParams *params = parent->parameters();

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
   lastTimeUpdateCalled = time;
   return PV_SUCCESS;
}

int CloneConn::finalizeUpdate(double timed, double dt) {
   // Orig conn is in charge of calling finalizeUpdate for postConn.
   return PV_SUCCESS;
}

int CloneConn::deleteWeights() {
   // Have to make sure not to free memory belonging to originalConn.
   // Set pointers that point into originalConn to NULL so that free() has no effect
   // when HyPerConn::deleteWeights or HyPerConn::deleteWeights is called
   wPatches       = NULL;
   wDataStart     = NULL;
   gSynPatchStart = NULL;
   aPostOffset    = NULL;
   dwDataStart    = NULL;
   return 0;
}

CloneConn::~CloneConn() {
   free(originalConnName);
   deleteWeights();
   postConn          = NULL;
   postToPreActivity = NULL;
}

} // end namespace PV
