/*
 * TransposePoolingConn.cpp *
 *  Created on: March 25, 2015
 *      Author: slundquist
 */

#include "TransposePoolingConn.hpp"

namespace PV {

TransposePoolingConn::TransposePoolingConn() {
   initialize_base();
} // TransposePoolingConn::~TransposePoolingConn()

TransposePoolingConn::TransposePoolingConn(const char *name, HyPerCol *hc) {
   initialize_base();
   int status = initialize(name, hc);
}

TransposePoolingConn::~TransposePoolingConn() {
   free(mOriginalConnName);
   mOriginalConnName = NULL;
   deleteWeights();
   if (needPost) {
      postConn = NULL;
   }
   // Transpose conn doesn't allocate postToPreActivity
   if (this->postToPreActivity) {
      postToPreActivity = NULL;
   }
} // TransposePoolingConn::~TransposePoolingConn()

int TransposePoolingConn::initialize_base() {
   plasticityFlag     = false; // Default value; override in params
   weightUpdatePeriod = 1; // Default value; override in params
   weightUpdateTime   = 0;
   // TransposePoolingConn::initialize_base() gets called after
   // HyPerConn::initialize_base() so these default values override
   // those in HyPerConn::initialize_base().

   needFinalize = true;
   return PV_SUCCESS;
} // TransposePoolingConn::initialize_base()

int TransposePoolingConn::initialize(const char *name, HyPerCol *hc) {
   int status = BaseConnection::initialize(name, hc);

   assert(parent);
   PVParams *inputParams = parent->parameters();

   // set accumulateFunctionPointer
   assert(!inputParams->presentAndNotBeenRead(name, "pvpatchAccumulateType"));
   switch (mPoolingType) {
      case PoolingConn::MAX:
         accumulateFunctionPointer         = &pvpatch_max_pooling;
         accumulateFunctionFromPostPointer = &pvpatch_max_pooling_from_post;
         break;
      case PoolingConn::SUM:
         accumulateFunctionPointer         = &pvpatch_sum_pooling;
         accumulateFunctionFromPostPointer = &pvpatch_sumpooling_from_post;
         break;
      case PoolingConn::AVG:
         accumulateFunctionPointer         = &pvpatch_sum_pooling;
         accumulateFunctionFromPostPointer = &pvpatch_sumpooling_from_post;
         break;
      default: pvAssert(0); break;
   }

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
// the associated parameters from the original connection's values.
// communicateInitInfo will check if those parameters exist in params for
// the TransposePoolingConn group, and whether they are consistent with the
// originalConn parameters.
// If consistent, issue a warning that the param is unnecessary and continue.
// If inconsistent, issue an error and quit.
// We can't do that in the read-method because we can't be sure originalConn
// has set its own parameter yet (or even if it's been instantiated),
// and in theory originalConn could be a subclass that determines
// the parameter some way other than reading its own parameter
// group's param directly.

void TransposePoolingConn::ioParam_receiveGpu(enum ParamsIOFlag ioFlag) {
   // During the communication phase, receiveGpu will be copied from mOriginalConn
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "receiveGpu");
   }
}

void TransposePoolingConn::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   // During the communication phase, sharedWeights will be copied from mOriginalConn
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryStringParameter(name, "sharedWeights");
   }
}

void TransposePoolingConn::ioParam_weightInitType(enum ParamsIOFlag ioFlag) {
   // TransposePoolingConn doesn't use a weight initializer
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryStringParameter(name, "weightInitType", NULL);
   }
}

void TransposePoolingConn::ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      initializeFromCheckpointFlag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "initializeFromCheckpointFlag");
   }
   // During the setInitialValues phase, the conn will be computed from the original conn, so
   // initializeFromCheckpointFlag is not needed.
}

void TransposePoolingConn::ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag) {}

void TransposePoolingConn::ioParam_plasticityFlag(enum ParamsIOFlag ioFlag) {
   // During the communication phase, plasticityFlag will be copied from mOriginalConn
}

void TransposePoolingConn::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      // make sure that TransposePoolingConn always checks if its mOriginalConn has updated
      triggerFlag      = false;
      triggerLayerName = NULL;
      parent->parameters()->handleUnnecessaryParameter(name, "triggerFlag", triggerFlag);
      parent->parameters()->handleUnnecessaryStringParameter(name, "triggerLayerName", NULL);
   }
}

// TODO: ioParam_pvpatchAccumulateType and unsetAccumulateType are copied from PV::PoolingConn.
// Can we make TransposePoolingConn a derived class of PoolingConn?  (TransposeConn is currently a
// derived class of HyPerConn.)
void TransposePoolingConn::ioParam_pvpatchAccumulateType(enum ParamsIOFlag ioFlag) {
   // During the communication phase, pvpatchAccumulateType will be copied from mOriginalConn
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryStringParameter(name, "pvpatchAccumulateType");
   }
}

void TransposePoolingConn::ioParam_writeStep(enum ParamsIOFlag ioFlag) {
   writeStep = -1;
   parent->parameters()->handleUnnecessaryParameter(name, "writeStep");
}

void TransposePoolingConn::ioParam_combine_dW_with_W_flag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      combine_dW_with_W_flag = false;
      parent->parameters()->handleUnnecessaryParameter(
            name, "combine_dW_with_W_flag", combine_dW_with_W_flag);
   }
}

void TransposePoolingConn::ioParam_nxp(enum ParamsIOFlag ioFlag) {
   parent->parameters()->handleUnnecessaryParameter(name, "nxp");
   // TransposePoolingConn determines nxp from mOriginalConn, during communicateInitInfo
}

void TransposePoolingConn::ioParam_nyp(enum ParamsIOFlag ioFlag) {
   parent->parameters()->handleUnnecessaryParameter(name, "nyp");
   // TransposePoolingConn determines nyp from mOriginalConn, during communicateInitInfo
}

void TransposePoolingConn::ioParam_nfp(enum ParamsIOFlag ioFlag) {
   parent->parameters()->handleUnnecessaryParameter(name, "nfp");
   // TransposePoolingConn determines nfp from mOriginalConn, during communicateInitInfo
}

void TransposePoolingConn::ioParam_dWMax(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      dWMax = 1.0;
      parent->parameters()->handleUnnecessaryParameter(name, "dWMax", dWMax);
   }
}

void TransposePoolingConn::ioParam_weightUpdatePeriod(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      weightUpdatePeriod = parent->getDeltaTime();
      // Every timestep needUpdate checks mOriginalConn's lastUpdateTime against transpose's
      // lastUpdateTime, so weightUpdatePeriod and initialWeightUpdateTime aren't needed
      parent->parameters()->handleUnnecessaryParameter(name, "weightUpdatePeriod");
   }
}

void TransposePoolingConn::ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      initialWeightUpdateTime = parent->getStartTime();
      // Every timestep needUpdate checks mOriginalConn's lastUpdateTime against transpose's
      // lastUpdateTime, so weightUpdatePeriod and initialWeightUpdateTime aren't needed
      parent->parameters()->handleUnnecessaryParameter(
            name, "initialWeightUpdateTime", initialWeightUpdateTime);
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
      normalizer      = NULL;
      normalizeMethod = strdup("none");
      parent->parameters()->handleUnnecessaryStringParameter(name, "normalizeMethod", "none");
   }
}

void TransposePoolingConn::ioParam_originalConnName(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamStringRequired(
         ioFlag, name, "originalConnName", &mOriginalConnName);
}

int TransposePoolingConn::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status    = PV_SUCCESS;
   mOriginalConn = message->lookup<PoolingConn>(std::string(mOriginalConnName));
   if (mOriginalConn == NULL) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "%s: originalConnName \"%s\" is not a PoolingConn.\n",
               getDescription_c(),
               mOriginalConnName);
         status = PV_FAILURE;
      }
   }
   if (status != PV_SUCCESS)
      return status;

   if (!mOriginalConn->getInitInfoCommunicatedFlag()) {
      if (parent->columnId() == 0) {
         InfoLog().printf(
               "%s must wait until original connection \"%s\" has finished its communicateInitInfo "
               "stage.\n",
               getDescription_c(),
               mOriginalConn->getName());
      }
      return PV_POSTPONE;
   }

#ifdef PV_USE_CUDA
   receiveGpu = mOriginalConn->getReceiveGpu();
   parent->parameters()->handleUnnecessaryParameter(name, "receiveGpu", receiveGpu);
#endif // PV_USE_CUDA

   sharedWeights = mOriginalConn->usingSharedWeights();
   parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights", sharedWeights);

   numAxonalArborLists = mOriginalConn->numberOfAxonalArborLists();
   parent->parameters()->handleUnnecessaryParameter(name, "numAxonalArbors", numAxonalArborLists);

   plasticityFlag = mOriginalConn->getPlasticityFlag();
   parent->parameters()->handleUnnecessaryParameter(name, "plasticityFlag", plasticityFlag);

   if (mOriginalConn->getShrinkPatches_flag()) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "TransposePoolingConn \"%s\": original conn \"%s\" has shrinkPatches set to true.  "
               "TransposePoolingConn has not been implemented for that case.\n",
               name,
               mOriginalConn->getName());
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   status = HyPerConn::communicateInitInfo(message); // calls setPatchSize()
   if (status != PV_SUCCESS)
      return status;

   // Check post layer phases to make sure it matches
   if (mOriginalConn->postSynapticLayer()->getPhase() >= post->getPhase()) {
      WarnLog().printf(
            "TransposePoolingConn \"%s\": originalConn's post layer phase is greater or equal than "
            "this layer's post. Behavior undefined.\n",
            name);
   }

   mPoolingType = mOriginalConn->getPoolingType();
   if (parent->parameters()->stringPresent(name, "pvpatchAccumulateType")) {
      char const *checkStringPresent =
            parent->parameters()->stringValue(name, "pvpatchAccumulateType");
      if (PoolingConn::parseAccumulateTypeString(checkStringPresent) != mPoolingType) {
         Fatal() << getDescription()
                 << ": originalConn accumulateType does not match this layer's accumulate type.\n";
      }
   }
   if (mPoolingType == PoolingConn::MAX && !mOriginalConn->needPostIndex()) {
#ifdef PV_USE_CUDA // Hopefully the awkwardness of this macro management will go away once I clean
      // up this class.
      if (!receiveGpu)
#endif // PV_USE_CUDA
      {
         if (parent->columnId() == 0) {
            ErrorLog().printf(
                  "TransposePoolingConn \"%s\": original pooling conn \"%s\" needs to have a "
                  "postIndexLayer if unmax pooling.\n",
                  name,
                  mOriginalConnName);
            status = PV_FAILURE;
         }
      }
   }
   if (status != PV_SUCCESS)
      return status;

   const PVLayerLoc *preLoc      = pre->getLayerLoc();
   const PVLayerLoc *origPostLoc = mOriginalConn->postSynapticLayer()->getLayerLoc();
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
   const PVLayerLoc *origPreLoc = mOriginalConn->postSynapticLayer()->getLayerLoc();
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
      mOriginalConn->setNeedPost();
   }
   mOriginalConn->setNeedAllocPostWeights(false);

   // Synchronize margines of this post and orig pre, and vice versa
   mOriginalConn->preSynapticLayer()->synchronizeMarginWidth(post);
   post->synchronizeMarginWidth(mOriginalConn->preSynapticLayer());

   mOriginalConn->postSynapticLayer()->synchronizeMarginWidth(pre);
   pre->synchronizeMarginWidth(mOriginalConn->postSynapticLayer());

   if (mOriginalConn->getPostIndexLayer()) {
      // Sync pre margins
      // Synchronize margines of this post and the postIndexLayer, and vice versa
      pre->synchronizeMarginWidth(mOriginalConn->getPostIndexLayer());
      mOriginalConn->getPostIndexLayer()->synchronizeMarginWidth(pre);

      // Need to tell postIndexLayer the number of delays needed by this connection
      int allowedDelay = mOriginalConn->getPostIndexLayer()->increaseDelayLevels(maxDelaySteps());
      if (allowedDelay < getDelayArraySize()) {
         if (this->parent->columnId() == 0) {
            ErrorLog().printf(
                  "%s: attempt to set delay to %d, but the maximum allowed delay is %d.  Exiting\n",
                  this->getDescription_c(),
                  getDelayArraySize(),
                  allowedDelay);
         }
         exit(EXIT_FAILURE);
      }
   }

   return status;
}

int TransposePoolingConn::setPatchSize() {
   // If mOriginalConn is many-to-one, the transpose connection is one-to-many; then xscaleDiff > 0.
   // Similarly, if mOriginalConn is one-to-many, xscaleDiff < 0.

   // Some of the code duplication might be eliminated by adding some functions to convert.h

   assert(pre && post);
   assert(mOriginalConn);

   int xscaleDiff = pre->getXScale() - post->getXScale();
   int nxp_orig   = mOriginalConn->xPatchSize();
   int nyp_orig   = mOriginalConn->yPatchSize();
   nxp            = nxp_orig;
   if (xscaleDiff > 0) {
      nxp *= (int)pow(2, xscaleDiff);
   }
   else if (xscaleDiff < 0) {
      nxp /= (int)pow(2, -xscaleDiff);
      assert(nxp_orig == nxp * pow(2, (float)(-xscaleDiff)));
   }

   int yscaleDiff = pre->getYScale() - post->getYScale();
   nyp            = nyp_orig;
   if (yscaleDiff > 0) {
      nyp *= (int)pow(2, yscaleDiff);
   }
   else if (yscaleDiff < 0) {
      nyp /= (int)pow(2, -yscaleDiff);
      assert(nyp_orig == nyp * pow(2, (float)(-yscaleDiff)));
   }

   nfp = post->getLayerLoc()->nf;
   // post->getLayerLoc()->nf must be the same as
   // mOriginalConn->preSynapticLayer()->getLayerLoc()->nf.
   // This requirement is checked in communicateInitInfo

   parent->parameters()->handleUnnecessaryParameter(name, "nxp", nxp);
   parent->parameters()->handleUnnecessaryParameter(name, "nyp", nyp);
   parent->parameters()->handleUnnecessaryParameter(name, "nfp", nfp);
   return PV_SUCCESS;
}

int TransposePoolingConn::allocateDataStructures() {
   if (!mOriginalConn->getDataStructuresAllocatedFlag()) {
      if (parent->columnId() == 0) {
         InfoLog().printf(
               "%s must wait until original connection \"%s\" has finished its "
               "allocateDataStructures stage.\n",
               getDescription_c(),
               mOriginalConn->getName());
      }
      return PV_POSTPONE;
   }

   bool tempNeedPost = false;
   // Turn off need post so postConn doesn't get allocated
   if (needPost) {
      needPost = false;
      postConn = mOriginalConn;
      // TODO this buffer is only needed if this transpose conn is receiving from post
      mOriginalConn->postConn->allocatePostToPreBuffer();
      postToPreActivity = mOriginalConn->postConn->getPostToPreActivity();
      tempNeedPost      = true;
   }
   int status = HyPerConn::allocateDataStructures();
   if (status != PV_SUCCESS) {
      return status;
   }

   // Set nessessary buffers
   if (tempNeedPost) {
      needPost = true;
   }

   normalizer = NULL;

   return PV_SUCCESS;
}

int TransposePoolingConn::constructWeights() {
   setPatchStrides();
   wPatches       = this->mOriginalConn->postConn->get_wPatches();
   wDataStart     = this->mOriginalConn->postConn->get_wDataStart();
   gSynPatchStart = this->mOriginalConn->postConn->getGSynPatchStart();
   aPostOffset    = this->mOriginalConn->postConn->getAPostOffset();
   dwDataStart    = this->mOriginalConn->postConn->get_dwDataStart();
   return PV_SUCCESS;
}

int TransposePoolingConn::deleteWeights() {
   // Have to make sure not to free memory belonging to mOriginalConn.
   // Set pointers that point into mOriginalConn to NULL so that free() has no effect
   // when HyPerConn::deleteWeights or HyPerConn::deleteWeights is called
   wPatches       = NULL;
   wDataStart     = NULL;
   gSynPatchStart = NULL;
   aPostOffset    = NULL;
   dwDataStart    = NULL;
   return 0;
}

int TransposePoolingConn::registerData(Checkpointer *checkpointer) {
   registerTimers(checkpointer);
   return PV_SUCCESS;
}

int TransposePoolingConn::setInitialValues() {
#ifdef PV_USE_CUDA
   if (receiveGpu) {
      return initializeTransposePoolingDeliverKernelArgs();
   }
#endif // PV_USE_CUDA
   return PV_SUCCESS;
}

#ifdef PV_USE_CUDA
int TransposePoolingConn::initializeTransposePoolingDeliverKernelArgs() {
   PVCuda::CudaDevice *device         = parent->getDevice();
   PVCuda::CudaBuffer *d_preDatastore = pre->getDeviceDatastore();
   PVCuda::CudaBuffer *d_postGSyn     = post->getDeviceGSyn();
   PVCuda::CudaBuffer *d_origConnPreDatastore =
         mOriginalConn->preSynapticLayer()->getDeviceDatastore();
   PVCuda::CudaBuffer *d_origConnPostGSyn = mOriginalConn->postSynapticLayer()->getDeviceGSyn();
   cudnnPoolingMode_t poolingMode;
   int multiplier = 1;
   switch (mPoolingType) {
      case PoolingConn::MAX: poolingMode = CUDNN_POOLING_MAX; break;
      case PoolingConn::SUM:
         poolingMode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
         multiplier  = nxpPost * nypPost;
         break;
      case PoolingConn::AVG: poolingMode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING; break;
      default: pvAssert(0); break;
   }
   mTransposePoolingDeliverKernel = new PVCuda::CudaTransposePoolingDeliverKernel(device);
   mTransposePoolingDeliverKernel->setArgs(
         pre->getLayerLoc(),
         post->getLayerLoc(),
         mOriginalConn->preSynapticLayer()->getLayerLoc(),
         mOriginalConn->postSynapticLayer()->getLayerLoc(),
         nxpPost,
         nypPost,
         poolingMode,
         multiplier,
         d_preDatastore,
         d_postGSyn,
         d_origConnPreDatastore,
         d_origConnPostGSyn,
         (int)channel);
   return PV_SUCCESS;
}
#endif // PV_USE_CUDA

bool TransposePoolingConn::needUpdate(double timed, double dt) {
   return plasticityFlag && mOriginalConn->getLastUpdateTime() > lastUpdateTime;
}

int TransposePoolingConn::updateState(double time, double dt) {
   lastTimeUpdateCalled = time;
   return PV_SUCCESS;
}

double TransposePoolingConn::computeNewWeightUpdateTime(double time, double currentUpdateTime) {
   return weightUpdateTime; // TransposePoolingConn does not use weightUpdateTime to determine when
   // to update
}

int TransposePoolingConn::deliverPostsynapticPerspective(PVLayerCube const *activity, int arborID) {
   Fatal()
         << "Delivering from PostSynapticPerspective for TransposePoolingConn not implented yet\n";
   return PV_FAILURE; // suppresses warning in compilers that don't recognize Fatal always exits.
}

int TransposePoolingConn::deliverPresynapticPerspective(PVLayerCube const *activity, int arborID) {
   // Check if we need to update based on connection's channel
   if (getChannel() == CHANNEL_NOUPDATE) {
      return PV_SUCCESS;
   }
   assert(post->getChannel(getChannel()));

   const PVLayerLoc *preLoc  = preSynapticLayer()->getLayerLoc();
   const PVLayerLoc *postLoc = postSynapticLayer()->getLayerLoc();

   assert(arborID >= 0);
   const int numExtended = activity->numItems;

   // Grab postIdxLayer's data
   float *postIdxData = nullptr;
   if (mPoolingType == PoolingConn::MAX) {
      PoolingIndexLayer *postIndexLayer = mOriginalConn->getPostIndexLayer();
      assert(postIndexLayer);
      // Make sure this layer is an integer layer
      assert(postIndexLayer->getDataType() == PV_INT);
      int delay        = getDelay(arborID);
      PVLayerCube cube = postIndexLayer->getPublisher()->createCube(delay);
      postIdxData      = cube.data;
   }

   for (int b = 0; b < parent->getNBatch(); b++) {
      float *activityBatch = activity->data
                             + b * (preLoc->nx + preLoc->halo.rt + preLoc->halo.lt)
                                     * (preLoc->ny + preLoc->halo.up + preLoc->halo.dn)
                                     * preLoc->nf;
      float *gSynPatchHeadBatch =
            post->getChannel(getChannel()) + b * postLoc->nx * postLoc->ny * postLoc->nf;
      float *postIdxDataBatch = nullptr;
      if (mPoolingType == PoolingConn::MAX) {
         postIdxDataBatch = postIdxData + b * mOriginalConn->getPostIndexLayer()->getNumExtended();
      }

      SparseList<float>::Entry const *activeIndicesBatch = NULL;
      if (activity->isSparse) {
         activeIndicesBatch = (SparseList<float>::Entry *)activity->activeIndices
                              + b * (preLoc->nx + preLoc->halo.rt + preLoc->halo.lt)
                                      * (preLoc->ny + preLoc->halo.up + preLoc->halo.dn)
                                      * preLoc->nf;
      }

      int numLoop;
      if (activity->isSparse) {
         numLoop = activity->numActive[b];
      }
      else {
         numLoop = numExtended;
      }

#ifdef PV_USE_OPENMP_THREADS
      // Clear all thread gsyn buffer
      if (thread_gSyn) {
         int numNeurons = post->getNumNeurons();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int i = 0; i < parent->getNumThreads() * numNeurons; i++) {
            int ti              = i / numNeurons;
            int ni              = i % numNeurons;
            thread_gSyn[ti][ni] = 0;
         }
      }
#endif // PV_USE_OPENMP_THREADS

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
      for (int loopIndex = 0; loopIndex < numLoop; loopIndex++) {
         float a     = 0.0f;
         int kPreExt = loopIndex;
         if (activity->isSparse) {
            a       = activeIndicesBatch[loopIndex].value;
            kPreExt = activeIndicesBatch[loopIndex].index;
         }
         else {
            a = activityBatch[loopIndex];
         }
         if (a == 0.0f) {
            continue;
         }

         // If we're using thread_gSyn, set this here
         float *gSynPatchHead;
#ifdef PV_USE_OPENMP_THREADS
         if (thread_gSyn) {
            int ti        = omp_get_thread_num();
            gSynPatchHead = thread_gSyn[ti];
         }
         else {
            gSynPatchHead = gSynPatchHeadBatch;
         }
#else // PV_USE_OPENMP_THREADS
         gSynPatchHead = gSynPatchHeadBatch;
#endif // PV_USE_OPENMP_THREADS

         const int kxPreExt =
               kxPos(kPreExt,
                     preLoc->nx + preLoc->halo.lt + preLoc->halo.rt,
                     preLoc->ny + preLoc->halo.dn + preLoc->halo.up,
                     preLoc->nf);
         const int kyPreExt =
               kyPos(kPreExt,
                     preLoc->nx + preLoc->halo.lt + preLoc->halo.rt,
                     preLoc->ny + preLoc->halo.dn + preLoc->halo.up,
                     preLoc->nf);
         const int kfPre = featureIndex(
               kPreExt,
               preLoc->nx + preLoc->halo.lt + preLoc->halo.rt,
               preLoc->ny + preLoc->halo.dn + preLoc->halo.up,
               preLoc->nf);

         if (mPoolingType == PoolingConn::MAX) {
            const int kxPreGlobalExt = kxPreExt + preLoc->kx0;
            const int kyPreGlobalExt = kyPreExt + preLoc->ky0;
            if (kxPreGlobalExt < preLoc->halo.lt
                || kxPreGlobalExt >= preLoc->nxGlobal + preLoc->halo.lt
                || kyPreGlobalExt < preLoc->halo.up
                || kyPreGlobalExt >= preLoc->nyGlobal + preLoc->halo.up) {
               continue;
            }

            // Convert stored global extended index into local extended index
            int postGlobalExtIdx = (int)postIdxDataBatch[kPreExt];

            // If all inputs are zero and input layer is sparse, postGlobalExtIdx will still be -1.
            if (postGlobalExtIdx == -1) {
               continue;
            }

            // Make sure the index is in bounds
            assert(
                  postGlobalExtIdx >= 0
                  && postGlobalExtIdx
                           < (postLoc->nxGlobal + postLoc->halo.lt + postLoc->halo.rt)
                                   * (postLoc->nyGlobal + postLoc->halo.up + postLoc->halo.dn)
                                   * postLoc->nf);

            const int kxPostGlobalExt =
                  kxPos(postGlobalExtIdx,
                        postLoc->nxGlobal + postLoc->halo.lt + postLoc->halo.rt,
                        postLoc->nyGlobal + postLoc->halo.dn + postLoc->halo.up,
                        postLoc->nf);
            const int kyPostGlobalExt =
                  kyPos(postGlobalExtIdx,
                        postLoc->nxGlobal + postLoc->halo.lt + postLoc->halo.rt,
                        postLoc->nyGlobal + postLoc->halo.dn + postLoc->halo.up,
                        postLoc->nf);
            const int kfPost = featureIndex(
                  postGlobalExtIdx,
                  postLoc->nxGlobal + postLoc->halo.lt + postLoc->halo.rt,
                  postLoc->nyGlobal + postLoc->halo.dn + postLoc->halo.up,
                  postLoc->nf);

            const int kxPostLocalRes = kxPostGlobalExt - postLoc->kx0 - postLoc->halo.lt;
            const int kyPostLocalRes = kyPostGlobalExt - postLoc->ky0 - postLoc->halo.up;
            if (kxPostLocalRes < 0 || kxPostLocalRes >= postLoc->nx || kyPostLocalRes < 0
                || kyPostLocalRes >= postLoc->ny) {
               continue;
            }

            const int kPostLocalRes = kIndex(
                  kxPostLocalRes, kyPostLocalRes, kfPost, postLoc->nx, postLoc->ny, postLoc->nf);
            if (fabs(a) > fabs(gSynPatchHead[kPostLocalRes])) {
               gSynPatchHead[kPostLocalRes] = a;
            }
         }
         else {
            PVPatch *weights      = getWeights(kPreExt, arborID);
            const int nk          = weights->nx * fPatchSize();
            const int ny          = weights->ny;
            float *postPatchStart = gSynPatchHead + getGSynPatchStart(kPreExt, arborID);
            const int sy          = getPostNonextStrides()->sy; // stride in layer

            int offset = kfPre;
            int sf     = fPatchSize();

            float w = 1.0f;
            if (mPoolingType == PoolingConn::MAX) {
               w = 1.0f;
            }
            else if (mPoolingType == PoolingConn::MAX) {
               float relative_XScale = pow(2, (post->getXScale() - pre->getXScale()));
               float relative_YScale = pow(2, (post->getYScale() - pre->getYScale()));
               float normVal         = nxp * nyp;
               w                     = 1.0f / normVal;
            }
            void *auxPtr = NULL;
            for (int y = 0; y < ny; y++) {
               (accumulateFunctionPointer)(
                     0, nk, postPatchStart + y * sy + offset, a, &w, auxPtr, sf);
            }
         }
      }

#ifdef PV_USE_OPENMP_THREADS
      // Set back into gSyn
      if (thread_gSyn) {
         float *gSynPatchHead = gSynPatchHeadBatch;
         int numNeurons       = post->getNumNeurons();
// Looping over neurons first to be thread safe
#pragma omp parallel for
         for (int ni = 0; ni < numNeurons; ni++) {
            if (mPoolingType == PoolingConn::MAX) {
               // Grab maxumum magnitude of thread_gSyn and set that value
               float maxMag  = -INFINITY;
               int maxMagIdx = -1;
               for (int ti = 0; ti < parent->getNumThreads(); ti++) {
                  if (maxMag < fabsf(thread_gSyn[ti][ni])) {
                     maxMag    = fabsf(thread_gSyn[ti][ni]);
                     maxMagIdx = ti;
                  }
               }
               assert(maxMagIdx >= 0);
               gSynPatchHead[ni] = thread_gSyn[maxMagIdx][ni];
            }
            else {
               for (int ti = 0; ti < parent->getNumThreads(); ti++) {
                  gSynPatchHead[ni] += thread_gSyn[ti][ni];
               }
            }
         }
      }
#endif
   }
   return PV_SUCCESS;
}

#ifdef PV_USE_CUDA
int TransposePoolingConn::deliverPresynapticPerspectiveGPU(
      PVLayerCube const *activity,
      int arborID) {
   return deliverGPU(activity, arborID);
}

int TransposePoolingConn::deliverPostsynapticPerspectiveGPU(
      PVLayerCube const *activity,
      int arborID) {
   return deliverGPU(activity, arborID);
}

int TransposePoolingConn::deliverGPU(PVLayerCube const *activity, int arborID) {
   // Check channel number for noupdate
   if (getChannel() == CHANNEL_NOUPDATE) {
      return PV_SUCCESS;
   }
   pvAssert(post->getChannel(getChannel()));

   if (pre->getUpdatedDeviceDatastoreFlag()) {
      float *h_preDatastore              = activity->data;
      PVCuda::CudaBuffer *d_preDatastore = pre->getDeviceDatastore();
      pvAssert(d_preDatastore);
      d_preDatastore->copyToDevice(h_preDatastore);
      // Device now has updated
      pre->setUpdatedDeviceDatastoreFlag(false);
   }

   mTransposePoolingDeliverKernel->run();
   return PV_SUCCESS;
}
#endif // PV_USE_CUDA

} // end namespace PV
