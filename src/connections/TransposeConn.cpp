/*
 * TransposeConn.cpp
 *
 *  Created on: May 16, 2011
 *      Author: peteschultz
 */

#include "TransposeConn.hpp"
#include "privateTransposeConn.hpp"

namespace PV {

TransposeConn::TransposeConn() { initialize_base(); } // TransposeConn::~TransposeConn()

TransposeConn::TransposeConn(const char *name, HyPerCol *hc) {
   initialize_base();
   int status = initialize(name, hc);
}

TransposeConn::~TransposeConn() {
   free(originalConnName);
   originalConnName = NULL;
   deleteWeights();
   postConn = NULL;
   // Transpose conn doesn't allocate postToPreActivity
   postToPreActivity = NULL;
} // TransposeConn::~TransposeConn()

int TransposeConn::initialize_base() {
   plasticityFlag     = false; // Default value; override in params
   weightUpdatePeriod = 1; // Default value; override in params
   weightUpdateTime   = 0;
   // TransposeConn::initialize_base() gets called after
   // HyPerConn::initialize_base() so these default values override
   // those in HyPerConn::initialize_base().
   // TransposeConn::initialize_base() gets called before
   // HyPerConn::initialize(), so these values still get overridden
   // by the params file values.

   originalConnName = NULL;
   originalConn     = NULL;
   needFinalize     = true;
   return PV_SUCCESS;
} // TransposeConn::initialize_base()

int TransposeConn::initialize(const char *name, HyPerCol *hc) {
   int status = PV_SUCCESS;
   if (status == PV_SUCCESS)
      status = HyPerConn::initialize(name, hc);
   return status;
}

int TransposeConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_originalConnName(ioFlag);
   return status;
}

// We override many ioParam-methods because TransposeConn will determine
// the associated parameters from the originalConn's values.
// communicateInitInfo will check if those parameters exist in params for
// the TransposeConn group, and whether they are consistent with the
// originalConn parameters.
// If consistent, issue a warning that the param is unnecessary and continue.
// If inconsistent, issue an error and quit.
// We can't do that in the read-method because we can't be sure originalConn
// has set its own parameter yet (or even if it's been instantiated),
// and in theory originalConn could be a subclass that determines
// the parameter some way other than reading its own parameter
// group's param directly.

void TransposeConn::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   // During the communication phase, numAxonalArbors will be copied from originalConn
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryStringParameter(name, "sharedWeights");
   }
}

void TransposeConn::ioParam_weightInitType(enum ParamsIOFlag ioFlag) {
   // TransposeConn doesn't use a weight initializer
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryStringParameter(name, "weightInitType", NULL);
   }
}

void TransposeConn::ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      initializeFromCheckpointFlag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "initializeFromCheckpointFlag");
   }
   // During the setInitialValues phase, the conn will be computed from the original conn, so
   // initializeFromCheckpointFlag is not needed.
}

void TransposeConn::ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag) {}

void TransposeConn::ioParam_plasticityFlag(enum ParamsIOFlag ioFlag) {
   // During the communication phase, plasticityFlag will be copied from originalConn
}

void TransposeConn::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      // make sure that TransposePoolingConn always checks if its originalConn has updated
      triggerFlag      = false;
      triggerLayerName = NULL;
      parent->parameters()->handleUnnecessaryStringParameter(name, "triggerLayerName", NULL);
   }
}

void TransposeConn::ioParam_combine_dW_with_W_flag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      combine_dW_with_W_flag = false;
      parent->parameters()->handleUnnecessaryParameter(
            name, "combine_dW_with_W_flag", combine_dW_with_W_flag);
   }
}

void TransposeConn::ioParam_nxp(enum ParamsIOFlag ioFlag) {
   // TransposeConn determines nxp from originalConn, during communicateInitInfo
}

void TransposeConn::ioParam_nyp(enum ParamsIOFlag ioFlag) {
   // TransposeConn determines nyp from originalConn, during communicateInitInfo
}

void TransposeConn::ioParam_nfp(enum ParamsIOFlag ioFlag) {
   // TransposeConn determines nfp from originalConn, during communicateInitInfo
}

void TransposeConn::ioParam_dWMax(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      dWMax = 1.0;
      parent->parameters()->handleUnnecessaryParameter(name, "dWMax", dWMax);
   }
}

void TransposeConn::ioParam_weightUpdatePeriod(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      weightUpdatePeriod = parent->getDeltaTime();
      // Every timestep needUpdate checks originalConn's lastUpdateTime against transpose's
      // lastUpdateTime, so weightUpdatePeriod and initialWeightUpdateTime aren't needed
      parent->parameters()->handleUnnecessaryParameter(name, "weightUpdatePeriod");
   }
}

void TransposeConn::ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      initialWeightUpdateTime = parent->getStartTime();
      // Every timestep needUpdate checks originalConn's lastUpdateTime against transpose's
      // lastUpdateTime, so weightUpdatePeriod and initialWeightUpdateTime aren't needed
      parent->parameters()->handleUnnecessaryParameter(
            name, "initialWeightUpdateTime", initialWeightUpdateTime);
      weightUpdateTime = initialWeightUpdateTime;
   }
}

void TransposeConn::ioParam_shrinkPatches(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      shrinkPatches_flag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "shrinkPatches", shrinkPatches_flag);
   }
}

void TransposeConn::ioParam_normalizeMethod(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      normalizer      = NULL;
      normalizeMethod = strdup("none");
      parent->parameters()->handleUnnecessaryStringParameter(name, "normalizeMethod", "none");
   }
}

void TransposeConn::ioParam_originalConnName(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamStringRequired(ioFlag, name, "originalConnName", &originalConnName);
}

int TransposeConn::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status         = PV_SUCCESS;
   this->originalConn = message->lookup<HyPerConn>(std::string(originalConnName));
   if (originalConn == NULL) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "%s: originalConnName \"%s\" is not an connection in the column.\n",
               getDescription_c(),
               originalConnName);
         status = PV_FAILURE;
      }
   }
   if (status != PV_SUCCESS)
      return status;

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

   sharedWeights = originalConn->usingSharedWeights();
   parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights", sharedWeights);

   numAxonalArborLists = originalConn->numberOfAxonalArborLists();
   parent->parameters()->handleUnnecessaryParameter(name, "numAxonalArbors", numAxonalArborLists);

   if (originalConn->getShrinkPatches_flag()) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "TransposeConn \"%s\": original conn \"%s\" has shrinkPatches set to true.  "
               "TransposeConn has not been implemented for that case.\n",
               name,
               originalConn->getName());
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   status = HyPerConn::communicateInitInfo(message); // calls setPatchSize()
   if (status != PV_SUCCESS)
      return status;

   const PVLayerLoc *preLoc      = pre->getLayerLoc();
   const PVLayerLoc *origPostLoc = originalConn->postSynapticLayer()->getLayerLoc();
   if (preLoc->nx != origPostLoc->nx || preLoc->ny != origPostLoc->ny
       || preLoc->nf != origPostLoc->nf) {
      if (parent->columnId() == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "%s: transpose's pre layer and original connection's post layer must have the same "
               "dimensions.\n",
               getDescription_c());
         errorMessage.printf(
               "    (x=%d, y=%d, f=%d) versus (x=%d, y=%d, f=%d).\n",
               preLoc->nx,
               preLoc->ny,
               preLoc->nf,
               origPostLoc->nx,
               origPostLoc->ny,
               origPostLoc->nf);
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   const PVLayerLoc *postLoc    = pre->getLayerLoc();
   const PVLayerLoc *origPreLoc = originalConn->postSynapticLayer()->getLayerLoc();
   if (postLoc->nx != origPreLoc->nx || postLoc->ny != origPreLoc->ny
       || postLoc->nf != origPreLoc->nf) {
      if (parent->columnId() == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "%s: transpose's post layer and original connection's pre layer must have the same "
               "dimensions.\n",
               getDescription_c());
         errorMessage.printf(
               "    (x=%d, y=%d, f=%d) versus (x=%d, y=%d, f=%d).\n",
               postLoc->nx,
               postLoc->ny,
               postLoc->nf,
               origPreLoc->nx,
               origPreLoc->ny,
               origPreLoc->nf);
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   if (!updateGSynFromPostPerspective) {
      originalConn->setNeedPost();
   }
   if (writeStep >= 0) {
      originalConn->setNeedPost();
   }

#ifdef PV_USE_CUDA
   if ((updateGSynFromPostPerspective && receiveGpu) || allocPostDeviceWeights) {
      originalConn->setAllocDeviceWeights();
   }
   if ((!updateGSynFromPostPerspective && receiveGpu) || allocDeviceWeights) {
      originalConn->setAllocPostDeviceWeights();
   }
#endif

   // Synchronize margines of this post and orig pre, and vice versa
   originalConn->preSynapticLayer()->synchronizeMarginWidth(post);
   post->synchronizeMarginWidth(originalConn->preSynapticLayer());

   originalConn->postSynapticLayer()->synchronizeMarginWidth(pre);
   pre->synchronizeMarginWidth(originalConn->postSynapticLayer());

   return status;
}

int TransposeConn::setPatchSize() {
   // If originalConn is many-to-one, the transpose connection is one-to-many; then xscaleDiff > 0.
   // Similarly, if originalConn is one-to-many, xscaleDiff < 0.

   // Some of the code duplication might be eliminated by adding some functions to convert.h

   assert(pre && post);
   assert(originalConn);

   int xscaleDiff = pre->getXScale() - post->getXScale();
   int yscaleDiff = pre->getYScale() - post->getYScale();
   int nxp_orig   = originalConn->xPatchSize();
   int nyp_orig   = originalConn->yPatchSize();

   nxp = nxp_orig;
   if (xscaleDiff > 0) {
      nxp *= (int)pow(2, xscaleDiff);
   }
   else if (xscaleDiff < 0) {
      nxp /= (int)pow(2, -xscaleDiff);
      assert(nxp_orig == nxp * pow(2, (float)(-xscaleDiff)));
   }

   nyp = nyp_orig;
   if (yscaleDiff > 0) {
      nyp *= (int)pow(2, yscaleDiff);
   }
   else if (yscaleDiff < 0) {
      nyp /= (int)pow(2, -yscaleDiff);
      assert(nyp_orig == nyp * pow(2, (float)(-yscaleDiff)));
   }

   // post->getLayerLoc()->nf must be the same as
   // originalConn->preSynapticLayer()->getLayerLoc()->nf.
   // This requirement is checked in communicateInitInfo
   nfp = post->getLayerLoc()->nf;

   parent->parameters()->handleUnnecessaryParameter(name, "nxp", nxp);
   parent->parameters()->handleUnnecessaryParameter(name, "nyp", nyp);
   parent->parameters()->handleUnnecessaryParameter(name, "nfp", nfp);
   return PV_SUCCESS;
}

#ifdef PV_USE_CUDA
// Device buffers live in origConn
int TransposeConn::allocateDeviceWeights() { return PV_SUCCESS; }
int TransposeConn::allocatePostDeviceWeights() { return PV_SUCCESS; }
#endif

// Set this post to orig
int TransposeConn::allocatePostConn() {
   InfoLog() << "Connection " << name << " setting " << originalConn->getName() << " as postConn\n";
   postConn = originalConn;

   // Can't do this with shrink patches flag
   if (needPost && !shrinkPatches_flag) {
      allocatePostToPreBuffer();
   }
   return PV_SUCCESS;
}

int TransposeConn::allocateDataStructures() {
   if (!originalConn->getDataStructuresAllocatedFlag()) {
      if (parent->columnId() == 0) {
         InfoLog().printf(
               "%s must wait until original connection \"%s\" has finished its "
               "allocateDataStructures stage.\n",
               getDescription_c(),
               originalConn->getName());
      }
      return PV_POSTPONE;
   }

   int status = HyPerConn::allocateDataStructures();
   if (status != PV_SUCCESS) {
      return status;
   }

   normalizer = NULL;

   return status;
}

int TransposeConn::registerData(Checkpointer *checkpointer) {
   // Skip over HyPerConn, because TransposeConn doesn't need to write any checkpoint files.
   int status = BaseConnection::registerData(checkpointer);
   if (status != PV_SUCCESS) {
      return status;
   }

   // Still need to do the things HyPerConn does other than register checkpoint data.
   openOutputStateFile(checkpointer);
   registerTimers(checkpointer);
   return status;
}

int TransposeConn::constructWeights() {
   if (originalConn->postConn) {
      setPatchStrides();
      wPatches       = originalConn->postConn->get_wPatches();
      wDataStart     = originalConn->postConn->get_wDataStart();
      gSynPatchStart = originalConn->postConn->getGSynPatchStart();
      aPostOffset    = originalConn->postConn->getAPostOffset();
      dwDataStart    = originalConn->postConn->get_dwDataStart();
   }
   return PV_SUCCESS;
}

int TransposeConn::deleteWeights() {
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

int TransposeConn::setInitialValues() {
   int status = PV_SUCCESS;
   if (originalConn->getInitialValuesSetFlag()) {
      status = HyPerConn::setInitialValues(); // calls initializeWeights
   }
   else {
      status = PV_POSTPONE;
   }
   return status;
}

PVPatch ***TransposeConn::initializeWeights(PVPatch ***patches, float **dataStart) {
   // TransposeConn must wait until after originalConn has been normalized, so weight initialization
   // doesn't take place until HyPerCol::run calls finalizeUpdate
   return patches;
}

bool TransposeConn::needUpdate(double timed, double dt) { return false; }

int TransposeConn::updateState(double time, double dt) {
   lastTimeUpdateCalled = time;
   return PV_SUCCESS;
}

double TransposeConn::computeNewWeightUpdateTime(double time, double currentUpdateTime) {
   return weightUpdateTime; // TransposeConn does not use weightUpdateTime to determine when to
   // update
}

int TransposeConn::finalizeUpdate(double timed, double dt) {
   // Orig conn is in charge of calling finalizeUpdate for postConn.
   return PV_SUCCESS;
}

} // end namespace PV
