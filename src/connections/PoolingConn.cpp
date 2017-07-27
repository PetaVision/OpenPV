/*
 * PoolingConn.cpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#include "PoolingConn.hpp"
#include <cmath>
#include <cstring>
#include <locale>
#include <string>

namespace PV {

PoolingConn::PoolingConn() { initialize_base(); }

PoolingConn::PoolingConn(const char *name, HyPerCol *hc) : HyPerConn() {
   initialize_base();
   initialize(name, hc);
}

PoolingConn::~PoolingConn() {
   if (thread_gateIdxBuffer) {
      for (int ti = 0; ti < parent->getNumThreads(); ti++) {
         free(thread_gateIdxBuffer[ti]);
         thread_gateIdxBuffer[ti] = NULL;
      }
      free(thread_gateIdxBuffer);
      thread_gateIdxBuffer = NULL;
   }
   if (postIndexLayerName) {
      free(postIndexLayerName);
   }
}

int PoolingConn::initialize_base() {
   pvpatchAccumulateType = HyPerConn::UNDEFINED;
   thread_gateIdxBuffer  = NULL;
   needPostIndexLayer    = false;
   postIndexLayerName    = NULL;
   postIndexLayer        = NULL;
   poolingType           = UNDEFINED;

   return PV_SUCCESS;
}

int PoolingConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_needPostIndexLayer(ioFlag);
   ioParam_postIndexLayerName(ioFlag);

   return status;
}

void PoolingConn::ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      initializeFromCheckpointFlag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "initializeFromCheckpointFlag");
   }
}

void PoolingConn::ioParam_plasticityFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      plasticityFlag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "plasticityFlag");
   }
}

void PoolingConn::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      sharedWeights = false;
      parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights");
   }
}

void PoolingConn::ioParam_weightInitType(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryStringParameter(name, "weightInitType", NULL);
   }
}

void PoolingConn::ioParam_pvpatchAccumulateType(enum ParamsIOFlag ioFlag) {
   PVParams *params = parent->parameters();

   parent->parameters()->ioParamStringRequired(
         ioFlag, name, "pvpatchAccumulateType", &pvpatchAccumulateTypeString);
   if (ioFlag == PARAMS_IO_READ) {
      poolingType = parseAccumulateTypeString(pvpatchAccumulateTypeString);
      if (poolingType == UNDEFINED) {
         unsetAccumulateType();
      }
   }
}

PoolingConn::AccumulateType PoolingConn::parseAccumulateTypeString(char const *poolingTypeString) {
   if (poolingTypeString == nullptr) {
      return UNDEFINED;
   }
   PoolingConn::AccumulateType accType;
   std::string str(poolingTypeString);
   // Convert string to lowercase so that capitalization doesn't matter.
   for (auto &c : str) {
      c = std::tolower(c, std::locale());
   }
   // "max_pooling", "max pooling", "maxpooling" are equally acceptable (same for
   // sum and avg)
   if (str.size() >= 4 && (str[3] == ' ' || str[3] == '_')) {
      str.erase(3, 1);
   }

   if (strcmp(str.c_str(), "maxpooling") == 0) {
      accType = MAX;
   }
   else if (strcmp(str.c_str(), "sumpooling") == 0) {
      accType = SUM;
   }
   else if (strcmp(str.c_str(), "avgpooling") == 0) {
      accType = AVG;
   }
   else {
      accType = UNDEFINED;
   }
   return accType;
}

void PoolingConn::unsetAccumulateType() {
   if (parent->columnId() == 0) {
      ErrorLog(errorMessage);
      if (pvpatchAccumulateTypeString) {
         errorMessage.printf(
               "%s: pvpatchAccumulateType \"%s\" is unrecognized.",
               getDescription_c(),
               pvpatchAccumulateTypeString);
      }
      else {
         errorMessage.printf("%s: pvpatchAccumulateType NULL is unrecognized.", getDescription_c());
      }
      errorMessage.printf(
            "  Allowed values are \"maxpooling\", \"sumpooling\", "
            "or \"avgpooling\".");
   }
   MPI_Barrier(parent->getCommunicator()->communicator());
   exit(EXIT_FAILURE);
}

void PoolingConn::ioParam_needPostIndexLayer(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "needPostIndexLayer", &needPostIndexLayer, needPostIndexLayer);
}

void PoolingConn::ioParam_postIndexLayerName(enum ParamsIOFlag ioFlag) {
   if (needPostIndexLayer) {
      parent->parameters()->ioParamStringRequired(
            ioFlag, name, "postIndexLayerName", &postIndexLayerName);
   }
}

void PoolingConn::ioParam_writeStep(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      writeStep = -1;
      parent->parameters()->handleUnnecessaryParameter(name, "writeStep");
   }
}

void PoolingConn::ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      writeCompressedCheckpoints = false;
      parent->parameters()->handleUnnecessaryParameter(
            name, "writeCompressedCheckpoints", writeCompressedCheckpoints /*correct value*/);
   }
}

void PoolingConn::ioParam_normalizeMethod(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryStringParameter(
            name, "normalizeMethod", "none", false /*case_insensitive*/);
   }
}

int PoolingConn::initialize(const char *name, HyPerCol *hc) {
   int status = BaseConnection::initialize(name, hc);

#ifdef PV_USE_CUDA
   if (needPostIndexLayer && receiveGpu) {
      if (parent->getCommunicator()->commRank() == 0) {
         Fatal() << getDescription() << ": receiveGpu and needPostIndexLayer both set.  The GPU "
                                        "version does not currently compute the post index "
                                        "layer.";
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
#endif // PV_USE_CUDA

   assert(parent);
   PVParams *inputParams = parent->parameters();

   // set accumulateFunctionPointer
   assert(!inputParams->presentAndNotBeenRead(name, "pvpatchAccumulateType"));
   switch (poolingType) {
      case MAX:
         accumulateFunctionPointer         = &pvpatch_max_pooling;
         accumulateFunctionFromPostPointer = &pvpatch_max_pooling_from_post;
         break;
      case SUM:
         accumulateFunctionPointer         = &pvpatch_sum_pooling;
         accumulateFunctionFromPostPointer = &pvpatch_sumpooling_from_post;
         break;
      case AVG:
         accumulateFunctionPointer         = &pvpatch_sum_pooling;
         accumulateFunctionFromPostPointer = &pvpatch_sumpooling_from_post;
         break;
      default:
         pvAssert(0); // Only MAX, SUM, and AVG are defined in PoolingConn; other
         // methods should be
         // handled in other classes.
         break;
   }

   this->io_timer     = new Timer(getName(), "conn", "io     ");
   this->update_timer = new Timer(getName(), "conn", "update ");

   return status;
}

int PoolingConn::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = HyPerConn::communicateInitInfo(message);

   // Check pre/post connections here
   const PVLayerLoc *preLoc  = pre->getLayerLoc();
   const PVLayerLoc *postLoc = post->getLayerLoc();

   if (preLoc->nf != postLoc->nf) {
      Fatal() << "Pooling Layer " << name << ":  preLayer " << pre->getName() << " nf of "
              << preLoc->nf << " does not match postLayer " << post->getName() << " nf of "
              << postLoc->nf << ". Features must match\n";
   }

   float preToPostScaleX = (float)preLoc->nx / postLoc->nx;
   float preToPostScaleY = (float)preLoc->ny / postLoc->ny;
   if (preToPostScaleX < 1 || preToPostScaleY < 1) {
      Fatal() << "Pooling Layer " << name << ":  preLayer to postLayer must be "
                                             "a many to one or one to one "
                                             "conection\n";
   }

   if (needPostIndexLayer) {
      postIndexLayer = message->lookup<PoolingIndexLayer>(std::string(this->postIndexLayerName));
      if (postIndexLayer == NULL) {
         if (parent->columnId() == 0) {
            ErrorLog().printf(
                  "%s: postIndexLayerName \"%s\" is not a PoolingIndexLayer.\n",
                  getDescription_c(),
                  this->postIndexLayerName);
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }

      if (postIndexLayer->getDataType() != PV_INT) {
         if (parent->columnId() == 0) {
            ErrorLog().printf(
                  "%s: postIndexLayer \"%s\" must have data type "
                  "of int. Specify parameter "
                  "dataType in this layer to be \"int\".\n",
                  getDescription_c(),
                  this->postIndexLayerName);
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }

      const PVLayerLoc *idxLoc = postIndexLayer->getLayerLoc();
      // postIndexLayer must be the same size as the post layer
      //(margins doesnt matter)
      if (idxLoc->nxGlobal != postLoc->nxGlobal || idxLoc->nyGlobal != postLoc->nyGlobal
          || idxLoc->nf != postLoc->nf) {
         if (parent->columnId() == 0) {
            ErrorLog().printf(
                  "%s: postIndexLayer \"%s\" must have the same "
                  "dimensions as the post pooling "
                  "layer \"%s\".\n",
                  getDescription_c(),
                  this->postIndexLayerName,
                  this->postLayerName);
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }

      // TODO this is currently a hack, need to properly implement data types.
      assert(sizeof(int) == sizeof(float));
   }

   if (getUpdateGSynFromPostPerspective()) {
      setNeedPost();
      needAllocPostWeights = false;
   }

   return status;
}

void PoolingConn::clearGateIdxBuffer() {
   if (needPostIndexLayer) {
      // Reset postIndexLayer's gsyn
      resetGSynBuffers_PoolingIndexLayer(
            parent->getNBatch(),
            postIndexLayer->getNumNeurons(),
            postIndexLayer->getNumChannels(),
            postIndexLayer->getChannel(CHANNEL_EXC));
   }
}

int PoolingConn::allocateDataStructures() {
   if (postIndexLayer && postIndexLayer->getDataStructuresAllocatedFlag() == false) {
      if (parent->columnId() == 0) {
         InfoLog().printf(
               "%s must wait until postIndexLayer layer \"%s\" has finished its "
               "allocateDataStructures stage.\n",
               getDescription_c(),
               postIndexLayer->getName());
      }
      return PV_POSTPONE;
   }
   int status = HyPerConn::allocateDataStructures();
   if (status == PV_POSTPONE) {
      return status;
   }
   assert(status == PV_SUCCESS);

   if (needPostIndexLayer) {
      // Allocate temp buffers if needed, 1 for each thread
      if (parent->getNumThreads() > 1) {
         thread_gateIdxBuffer = (float **)malloc(sizeof(int *) * parent->getNumThreads());
         assert(thread_gateIdxBuffer);
         // Assign thread_gSyn to different points of tempMem
         for (int i = 0; i < parent->getNumThreads(); i++) {
            float *thread_buffer = (float *)malloc(sizeof(float) * post->getNumNeurons());
            if (!thread_buffer) {
               Fatal().printf(
                     "HyPerLayer \"%s\" error: rank %d unable to "
                     "allocate %zu memory for "
                     "thread_gateIdxBuffer: %s\n",
                     name,
                     parent->columnId(),
                     sizeof(int) * post->getNumNeurons(),
                     strerror(errno));
            }
            thread_gateIdxBuffer[i] = thread_buffer;
         }
      }

      if (thread_gateIdxBuffer) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int i = 0; i < parent->getNumThreads() * post->getNumNeurons(); i++) {
            int ti                       = i / post->getNumNeurons();
            int ni                       = i % post->getNumNeurons();
            thread_gateIdxBuffer[ti][ni] = -1;
         }
      }

      clearGateIdxBuffer();
   }
   return PV_SUCCESS;
}

int PoolingConn::registerData(Checkpointer *checkpointer) {
   registerTimers(checkpointer);
   return PV_SUCCESS;
}

int PoolingConn::setInitialValues() {
#ifdef PV_USE_CUDA
   if (receiveGpu) {
      return initializeDeliverKernelArgs();
   }
#endif // PV_USE_CUDA
   return PV_SUCCESS;
}

// On the GPU, pooling uses cudnnPoolingForward, so pre and post do the same
// thing.

#ifdef PV_USE_CUDA
int PoolingConn::initializeDeliverKernelArgs() {
   PVCuda::CudaDevice *device         = parent->getDevice();
   PVCuda::CudaBuffer *d_preDatastore = pre->getDeviceDatastore();
   PVCuda::CudaBuffer *d_postGSyn     = post->getDeviceGSyn();
   cudnnPoolingMode_t poolingMode;
   int multiplier = 1;
   switch (poolingType) {
      case MAX: poolingMode = CUDNN_POOLING_MAX; break;
      case SUM:
         poolingMode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
         multiplier  = nxpPost * nypPost;
         break;
      case AVG: poolingMode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING; break;
      default: pvAssert(0); break;
   }

   krPoolingDeliver = new PVCuda::CudaPoolingDeliverKernel(device);
   krPoolingDeliver->setArgs(
         pre->getLayerLoc(),
         post->getLayerLoc(),
         nxpPost,
         nypPost,
         poolingMode,
         multiplier,
         d_preDatastore,
         d_postGSyn,
         (int)channel);
   return PV_SUCCESS;
}
#endif // PV_USE_CUDA

int PoolingConn::constructWeights() {
   int sx       = nfp;
   int sy       = sx * nxp;
   int sp       = sy * nyp;
   int nPatches = getNumDataPatches();
   int status   = PV_SUCCESS;

   assert(!parent->parameters()->presentAndNotBeenRead(name, "shrinkPatches"));
   // createArbors() uses the value of shrinkPatches.  It should have already
   // been read in
   // ioParamsFillGroup.
   // allocate the arbor arrays:
   createArbors();

   setPatchStrides();

   for (int arborId = 0; arborId < numAxonalArborLists; arborId++) {
      PVPatch ***wPatches = get_wPatches();
      status              = createWeights(wPatches, arborId);
      assert(wPatches[arborId] != NULL);
      if (shrinkPatches_flag || arborId == 0) {
         status |= adjustAxonalArbors(arborId);
      }
   } // arborId

   // call to initializeWeights moved to BaseConnection::initializeState()
   status |= initPlasticityPatches();
   assert(status == 0);
   if (shrinkPatches_flag) {
      for (int arborId = 0; arborId < numAxonalArborLists; arborId++) {
         shrinkPatches(arborId);
      }
   }

   return status;
}

float PoolingConn::minWeight(int arborId) {
   if (getPoolingType() == MAX) {
      return 1.0;
   }
   else if (getPoolingType() == SUM) {
      return 1;
   }
   else if (getPoolingType() == AVG) {
      int relative_XScale = (int)pow(2, pre->getXScale() - post->getXScale());
      int relative_YScale = (int)pow(2, pre->getYScale() - post->getYScale());
      return (1.0 / (nxp * nyp * relative_XScale * relative_YScale));
   }
   else {
      assert(0); // only possibilities are PoolingConn::MAX, PoolingConn::SUM,
      // PoolingConn::AVG
      return 0.0f; // gets rid of a compile warning
   }
}

float PoolingConn::maxWeight(int arborId) {
   if (getPoolingType() == MAX) {
      return 1.0;
   }
   else if (getPoolingType() == SUM) {
      return 1;
   }
   else if (getPoolingType() == AVG) {
      int relative_XScale = (int)pow(2, pre->getXScale() - post->getXScale());
      int relative_YScale = (int)pow(2, pre->getYScale() - post->getYScale());
      return (1.0 / (nxp * nyp * relative_XScale * relative_YScale));
   }
   else {
      assert(0); // only possibilities are PoolingConn::MAX, PoolingConn::SUM,
      // PoolingConn::AVG
      return 0.0f; // gets rid of a compile warning
   }
}

int PoolingConn::deliverPresynapticPerspective(PVLayerCube const *activity, int arborID) {

   // Check if we need to update based on connection's channel
   if (getChannel() == CHANNEL_NOUPDATE) {
      return PV_SUCCESS;
   }
   assert(post->getChannel(getChannel()));

   float dt_factor = getConvertToRateDeltaTimeFactor();

   const PVLayerLoc *preLoc  = preSynapticLayer()->getLayerLoc();
   const PVLayerLoc *postLoc = postSynapticLayer()->getLayerLoc();

   assert(arborID >= 0);
   const int numExtended = activity->numItems;

   float resetVal = 0;
   if (getPoolingType() == MAX) {
      resetVal    = -INFINITY;
      float *gSyn = post->getChannel(getChannel());
// gSyn is res
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (int i = 0; i < post->getNumNeuronsAllBatches(); i++) {
         gSyn[i] = resetVal;
      }
   }

   clearGateIdxBuffer();

   for (int b = 0; b < parent->getNBatch(); b++) {
      float *activityBatch = activity->data
                             + b * (preLoc->nx + preLoc->halo.rt + preLoc->halo.lt)
                                     * (preLoc->ny + preLoc->halo.up + preLoc->halo.dn)
                                     * preLoc->nf;
      float *gSynPatchHeadBatch =
            post->getChannel(getChannel()) + b * postLoc->nx * postLoc->ny * postLoc->nf;
      float *gatePatchHeadBatch = NULL;
      if (needPostIndexLayer) {
         gatePatchHeadBatch =
               postIndexLayer->getChannel(CHANNEL_EXC) + b * postIndexLayer->getNumNeurons();
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

      if (thread_gateIdxBuffer) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int i = 0; i < parent->getNumThreads() * post->getNumNeurons(); i++) {
            int ti                       = i / post->getNumNeurons();
            int ni                       = i % post->getNumNeurons();
            thread_gateIdxBuffer[ti][ni] = -1;
         }
      }

#ifdef PV_USE_OPENMP_THREADS
      // Clear all gsyn buffers
      if (thread_gSyn) {
         int numNeurons = post->getNumNeurons();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int i = 0; i < parent->getNumThreads() * numNeurons; i++) {
            int ti              = i / numNeurons;
            int ni              = i % numNeurons;
            thread_gSyn[ti][ni] = resetVal;
         }
      }
#endif // PV_USE_OPENMP_THREADS

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
      for (int loopIndex = 0; loopIndex < numLoop; loopIndex++) {
         int kPreExt;
         float a = dt_factor;
         if (activity->isSparse) {
            kPreExt = activeIndicesBatch[loopIndex].index;
            a *= activeIndicesBatch[loopIndex].value;
         }
         else {
            kPreExt = loopIndex;
            a *= activityBatch[kPreExt];
         }

         // If we're using thread_gSyn, set this here
         float *gSynPatchHead;
         float *gatePatchHead = NULL;
#ifdef PV_USE_OPENMP_THREADS
         if (thread_gSyn) {
            int ti        = omp_get_thread_num();
            gSynPatchHead = thread_gSyn[ti];
         }
         else {
            gSynPatchHead = gSynPatchHeadBatch;
         }

         if (needPostIndexLayer) {
            if (thread_gateIdxBuffer) {
               int ti        = omp_get_thread_num();
               gatePatchHead = thread_gateIdxBuffer[ti];
            }
            else {
               gatePatchHead = gatePatchHeadBatch;
            }
         }
#else // PV_USE_OPENMP_THREADS
         gSynPatchHead = gSynPatchHeadBatch;
         if (needPostIndexLayer) {
            gatePatchHead = gatePatchHeadBatch;
         }
#endif // PV_USE_OPENMP_THREADS

         PVPatch *weights          = getWeights(kPreExt, arborID);
         const int nk              = weights->nx * fPatchSize();
         const int ny              = weights->ny;
         const int sy              = getPostNonextStrides()->sy; // stride in layer
         float *weightDataStart    = NULL;
         float *postPatchStart     = gSynPatchHead + getGSynPatchStart(kPreExt, arborID);
         float *postGatePatchStart = gatePatchHead + getGSynPatchStart(kPreExt, arborID);

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

         const int kxPreGlobalExt = kxPreExt + preLoc->kx0;
         const int kyPreGlobalExt = kyPreExt + preLoc->ky0;

         const int kPreGlobalExt = kIndex(
               kxPreGlobalExt,
               kyPreGlobalExt,
               kfPre,
               preLoc->nxGlobal + preLoc->halo.lt + preLoc->halo.rt,
               preLoc->nyGlobal + preLoc->halo.up + preLoc->halo.dn,
               preLoc->nf);

         int offset = kfPre;
         int sf     = fPatchSize();
         float w    = 1.0f;
         if (getPoolingType() == SUM) {
            w = 1.0f;
         }
         else if (getPoolingType() == AVG) {
            float relative_XScale = pow(2, (post->getXScale() - pre->getXScale()));
            float relative_YScale = pow(2, (post->getYScale() - pre->getYScale()));
            w                     = 1.0f / (nxp * nyp * relative_XScale * relative_YScale);
         }
         void *auxPtr = nullptr;
         for (int y = 0; y < ny; y++) {
            if (needPostIndexLayer) {
               auxPtr = &postGatePatchStart[y * sy + offset];
            }
            (accumulateFunctionPointer)(
                  kPreGlobalExt, nk, postPatchStart + y * sy + offset, a, &w, auxPtr, sf);
         }
      }
#ifdef PV_USE_OPENMP_THREADS
      // Accumulate back into gSyn // Should this be done in HyPerLayer where it
      // can be done once,
      // as opposed to once per connection?
      if (thread_gSyn) {
         float *gSynPatchHead = gSynPatchHeadBatch;
         float *gateIdxBuffer = nullptr;
         if (needPostIndexLayer && thread_gateIdxBuffer) {
            gateIdxBuffer = gatePatchHeadBatch;
         }
         int numNeurons = post->getNumNeurons();
// Looping over neurons first to be thread safe
#pragma omp parallel for
         for (int ni = 0; ni < numNeurons; ni++) {
            // Different for maxpooling
            if (getPoolingType() == MAX) {
               for (int ti = 0; ti < parent->getNumThreads(); ti++) {
                  if (gSynPatchHead[ni] < thread_gSyn[ti][ni]) {
                     gSynPatchHead[ni] = thread_gSyn[ti][ni];
                     if (needPostIndexLayer && thread_gateIdxBuffer) {
                        gateIdxBuffer[ni] = thread_gateIdxBuffer[ti][ni];
                     }
                  }
               }
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
   if (activity->isSparse) {
      float *gSyn = post->getChannel(getChannel());
      for (int k = 0; k < post->getNumNeuronsAllBatches(); k++) {
         if (gSyn[k] == -INFINITY) {
            gSyn[k] = 0.0f;
         }
      }
   }
   return PV_SUCCESS;
}

int PoolingConn::deliverPostsynapticPerspective(PVLayerCube const *activity, int arborID) {
   // Check channel number for noupdate
   if (getChannel() == CHANNEL_NOUPDATE) {
      return PV_SUCCESS;
   }
   assert(post->getChannel(getChannel()));

   assert(arborID >= 0);
   // Get number of neurons restricted target
   const int numPostRestricted = post->getNumNeurons();

   float dt_factor = getConvertToRateDeltaTimeFactor();

   const PVLayerLoc *sourceLoc = preSynapticLayer()->getLayerLoc();
   const PVLayerLoc *targetLoc = post->getLayerLoc();

   const int sourceNx = sourceLoc->nx;
   const int sourceNy = sourceLoc->ny;
   const int sourceNf = sourceLoc->nf;
   const int targetNx = targetLoc->nx;
   const int targetNy = targetLoc->ny;
   const int targetNf = targetLoc->nf;

   const PVHalo *sourceHalo = &sourceLoc->halo;
   const PVHalo *targetHalo = &targetLoc->halo;

   // get source layer's extended y stride
   int sy = (sourceNx + sourceHalo->lt + sourceHalo->rt) * sourceNf;

   // The start of the gsyn buffer
   float *gSynPatchHead = post->getChannel(this->getChannel());

   clearGateIdxBuffer();
   float *gatePatchHead = nullptr;
   if (needPostIndexLayer) {
      gatePatchHead = postIndexLayer->getChannel(CHANNEL_EXC);
   }

   long *startSourceExtBuf = getPostToPreActivity();
   if (!startSourceExtBuf) {
      Fatal() << "HyPerLayer::recvFromPost unable to get preToPostActivity "
                 "from connection. Is "
                 "shrink_patches on?\n";
   }

   float resetVal = 0;
   if (getPoolingType() == MAX) {
      resetVal = -INFINITY;
   }

   for (int b = 0; b < parent->getNBatch(); b++) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (int kTargetRes = 0; kTargetRes < numPostRestricted; kTargetRes++) {
         float *activityBatch = activity->data
                                + b * (sourceNx + sourceHalo->rt + sourceHalo->lt)
                                        * (sourceNy + sourceHalo->up + sourceHalo->dn) * sourceNf;
         float *gSynPatchHeadBatch = gSynPatchHead + b * targetNx * targetNy * targetNf;

         // Change restricted to extended post neuron
         int kTargetExt = kIndexExtended(
               kTargetRes,
               targetNx,
               targetNy,
               targetNf,
               targetHalo->lt,
               targetHalo->rt,
               targetHalo->dn,
               targetHalo->up);

         // Read from buffer
         long startSourceExt = startSourceExtBuf[kTargetRes];

         // Calculate target's start of gsyn
         float *gSynPatchPos = gSynPatchHeadBatch + kTargetRes;
         // Initialize patch as a huge negative number
         *gSynPatchPos = resetVal;

         float *gatePatchPos = nullptr;
         if (needPostIndexLayer) {
            gatePatchPos = gatePatchHead + b * postIndexLayer->getNumNeurons() + kTargetRes;
            // Initialize gatePatchPos as a negative number
            *gatePatchPos = (float)-1;
         }

         float *activityStartBuf = &(activityBatch[startSourceExt]);

         float *weightY   = NULL; // No weights in pooling
         int sf           = postConn->fPatchSize();
         int yPatchSize   = postConn->yPatchSize();
         int numPerStride = postConn->xPatchSize() * postConn->fPatchSize();

         const PVLayerLoc *postLoc = post->getLayerLoc();
         const int kfPost          = featureIndex(
               kTargetExt,
               postLoc->nx + postLoc->halo.lt + postLoc->halo.rt,
               postLoc->ny + postLoc->halo.dn + postLoc->halo.up,
               postLoc->nf);
         int offset = kfPost;

         float w = 1.0f;
         if (getPoolingType() == SUM) {
            w = 1.0f;
         }
         else if (getPoolingType() == AVG) {
            float relative_XScale = pow(2, (post->getXScale() - pre->getXScale()));
            float relative_YScale = pow(2, (post->getYScale() - pre->getYScale()));
            w                     = 1.0f / (nxp * nyp * relative_XScale * relative_YScale);
         }

         for (int ky = 0; ky < yPatchSize; ky++) {
            int kPreExt = startSourceExt + ky * sy + offset;
            const int kxPreExt =
                  kxPos(kPreExt,
                        sourceLoc->nx + sourceLoc->halo.lt + sourceLoc->halo.rt,
                        sourceLoc->ny + sourceLoc->halo.dn + sourceLoc->halo.up,
                        sourceLoc->nf);
            const int kyPreExt =
                  kyPos(kPreExt,
                        sourceLoc->nx + sourceLoc->halo.lt + sourceLoc->halo.rt,
                        sourceLoc->ny + sourceLoc->halo.dn + sourceLoc->halo.up,
                        sourceLoc->nf);
            const int kfPre = featureIndex(
                  kPreExt,
                  sourceLoc->nx + sourceLoc->halo.lt + sourceLoc->halo.rt,
                  sourceLoc->ny + sourceLoc->halo.dn + sourceLoc->halo.up,
                  sourceLoc->nf);
            const int kxPreGlobalExt = kxPreExt + sourceLoc->kx0;
            const int kyPreGlobalExt = kyPreExt + sourceLoc->ky0;
            const int kPreGlobalExt  = kIndex(
                  kxPreGlobalExt,
                  kyPreGlobalExt,
                  kfPre,
                  sourceLoc->nxGlobal + sourceLoc->halo.lt + sourceLoc->halo.rt,
                  sourceLoc->nyGlobal + sourceLoc->halo.up + sourceLoc->halo.dn,
                  sourceLoc->nf);

            float *activityY = &(activityStartBuf[ky * sy + offset]);

            (accumulateFunctionFromPostPointer)(
                  kPreGlobalExt,
                  numPerStride,
                  gSynPatchPos,
                  activityY,
                  &w,
                  dt_factor,
                  gatePatchPos,
                  sf);
         }
      }
   }
   return PV_SUCCESS;
}

#ifdef PV_USE_CUDA
int PoolingConn::deliverPresynapticPerspectiveGPU(PVLayerCube const *activity, int arborID) {
   return deliverGPU(activity, arborID);
}

int PoolingConn::deliverPostsynapticPerspectiveGPU(PVLayerCube const *activity, int arborID) {
   return deliverPresynapticPerspectiveGPU(activity, arborID);
}

int PoolingConn::deliverGPU(PVLayerCube const *activity, int arborID) {
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

   krPoolingDeliver->run();
   return PV_SUCCESS;
}
#endif // PV_USE_CUDA

} // end namespace PV
