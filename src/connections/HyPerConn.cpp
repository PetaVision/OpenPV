/*
 * HyPerConn.cpp
 *
 *  Created on: Oct 21, 2008
 *      Author: Craig Rasmussen
 */

#include "HyPerConn.hpp"
#include "checkpointing/CheckpointEntryWeightPvp.hpp"
#include "include/default_params.h"
#include "io/fileio.hpp"
#include "io/io.hpp"
#include "utils/conversions.h"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <float.h>
#include <iostream>
#include <limits.h>
#include <limits>
#include <stdlib.h>
#include <string.h>

#include "PlasticCloneConn.hpp"
#include "columns/Factory.hpp"
#include "io/FileStream.hpp"
#include "io/PrintStream.hpp"
#include "normalizers/NormalizeBase.hpp"
#include "privateTransposeConn.hpp"
#include "weightinit/InitWeights.hpp"

namespace PV {

HyPerConn::HyPerConn() { initialize_base(); }

HyPerConn::HyPerConn(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

HyPerConn::~HyPerConn() {
   cleanup();
   delete io_timer;
   io_timer = NULL;
   delete update_timer;
   update_timer = NULL;

   free(pvpatchAccumulateTypeString);

#ifdef PV_USE_CUDA
   pvDelete(d_WData);
   pvDelete(d_Patches);
   pvDelete(d_GSynPatchStart);
   pvDelete(d_PostToPreActivity);
   pvDelete(d_Patch2DataLookupTable);
   pvDelete(krRecvPost);
   pvDelete(krRecvPre);

#ifdef PV_USE_CUDNN
   pvDelete(cudnn_WData);
#endif
#endif // PV_USE_CUDA

   deleteWeights();

   // free the task information
   free(normalizeMethod);

   free(weightInitTypeString);

   pvDelete(weightInitializer);
   pvDelete(randState);

   if (postToPreActivity) {
      free(postToPreActivity);
      postToPreActivity = NULL;
   }

   if (maskLayerName) {
      free(maskLayerName);
      maskLayerName = NULL;
   }

   if (triggerLayerName) {
      free(triggerLayerName);
      triggerLayerName = NULL;
   }

   if (thread_gSyn) {
      for (int i = 0; i < parent->getNumThreads(); i++) {
         free(thread_gSyn[i]);
         thread_gSyn[i] = NULL;
      }
      free(thread_gSyn);
      thread_gSyn = NULL;
   }

   if (needPost && postConn) {
      delete postConn;
   }

   if (batchSkip) {
      free(batchSkip);
   }

   delete mOutputStateStream;
}

int HyPerConn::initialize_base() {
   nxp            = 1;
   nyp            = 1;
   nfp            = -1; // A negative value for nfp will be converted to postsynaptic layer's nf.
   warnDefaultNfp = true; // Issue a warning if default value of nfp (post's nf) is used.  Derived
   // layers can set to false if only one nfp is allowed (e.g. IdentConn)
   sxp    = 1;
   syp    = 1;
   sfp    = 1;
   parent = NULL;

   weightInitTypeString = NULL;
   weightInitializer    = NULL;

   io_timer     = NULL;
   update_timer = NULL;

   postConn = NULL;
   needPost = false;

   wPostTime                  = -1.0;
   wPostPatches               = NULL;
   wPostDataStart             = NULL;
   wPostPatchesp              = NULL;
   wPostDataStartp            = NULL;
   nxpPost                    = 0;
   nypPost                    = 0;
   nfpPost                    = 0;
   writeCompressedWeights     = false;
   writeCompressedCheckpoints = false;
   fileType = PVP_WGT_FILE_TYPE; // Subclass's initialize_base() gets called after HyPerConn's
   // initialize_base(), so this can be changed in subclasses.

   wDataStart     = NULL;
   dwDataStart    = NULL;
   wPatches       = NULL;
   aPostOffset    = NULL;
   gSynPatchStart = NULL;

   selfFlag =
         false; // specifies whether connection is from a layer to itself (i.e. a self-connection)
   combine_dW_with_W_flag = false;
   normalizeMethod        = NULL;
   normalizer             = NULL;
   plasticityFlag         = false;
   shrinkPatches_flag = false; // default value, overridden by params file parameter "shrinkPatches"
   // in readShrinkPatches()
   shrinkPatchesThresh         = 0;
   dWMax                       = std::numeric_limits<float>::quiet_NaN();
   strengthParamHasBeenWritten = false;

   updateGSynFromPostPerspective = false;
   thread_gSyn                   = NULL;

   pvpatchAccumulateTypeString = NULL;
   pvpatchAccumulateType       = CONVOLVE;

   initInfoCommunicatedFlag    = false;
   dataStructuresAllocatedFlag = false;
   initialValuesSetFlag        = false;

   randState = NULL;

   triggerFlag             = false; // Default to update every timestamp
   triggerLayer            = NULL;
   triggerLayerName        = NULL;
   triggerOffset           = 0;
   weightUpdatePeriod      = 0;
   initialWeightUpdateTime = 0;
   weightUpdateTime        = 0;

   clones.clear();

   postToPreActivity    = NULL;
   needFinalize         = true;
   needAllocPostWeights = true;

   lastUpdateTime        = 0.0;
   lastTimeUpdateCalled  = 0.0;
   symmetrizeWeightsFlag = false;
   patch2datalookuptable = NULL;
   numKernelActivations  = NULL;

   normalizeDwFlag = true;
   useMask         = false;
   maskLayerName   = NULL;
   maskFeatureIdx  = -1;
   mask            = NULL;

   batchSkip = NULL;

#ifdef PV_USE_CUDA
   receiveGpu              = false;
   allocDeviceWeights      = false;
   allocPostDeviceWeights  = false;
   d_WData                 = NULL;
   d_Patches               = NULL;
   d_GSynPatchStart        = NULL;
   d_PostToPreActivity     = NULL;
   d_Patch2DataLookupTable = NULL;
   krRecvPost              = NULL;
   krRecvPre               = NULL;
   mGpuGroupIdx            = -1;
#ifdef PV_USE_CUDNN
   cudnn_WData = NULL;
#endif
#endif

   return PV_SUCCESS;
}

int HyPerConn::createArbors() {
   const int shrinkNum =
         (shrinkPatches_flag ? numAxonalArborLists : 1) * preSynapticLayer()->getNumExtended();

   wPatches = (PVPatch ***)pvCalloc(numAxonalArborLists, sizeof(PVPatch **));
   // GTK:  gSynPatchStart is offset from beginning of gSyn buffer for the corresponding channel
   gSynPatchStart = (size_t **)pvCalloc(numAxonalArborLists, sizeof(size_t *));

   size_t *gSynPatchStartBuffer = (size_t *)pvCalloc(shrinkNum, sizeof(size_t));

   for (int k = 0; k < numAxonalArborLists; k++) {
      gSynPatchStart[k] =
            gSynPatchStartBuffer + shrinkPatches_flag * k * preSynapticLayer()->getNumExtended();
   }

   aPostOffset = (size_t **)pvCalloc(numAxonalArborLists, sizeof(size_t *));

   size_t *aPostOffsetBuffer = (size_t *)pvCalloc(shrinkNum, sizeof(size_t));

   for (int k = 0; k < numAxonalArborLists; k++) {
      aPostOffset[k] =
            aPostOffsetBuffer + shrinkPatches_flag * k * preSynapticLayer()->getNumExtended();
   }

   wDataStart = (float **)pvCalloc(numAxonalArborLists, sizeof(float *));

   dwDataStart = (float **)pvCalloc(numAxonalArborLists, sizeof(float *));

   if (sharedWeights && normalizeDwFlag) {
      numKernelActivations = (long **)pvCalloc(numAxonalArborLists, sizeof(long *));
   }

   return PV_SUCCESS;
}

void HyPerConn::createArborsOutOfMemory() {
   Fatal().printf("Out of memory error in HyPerConn::createArbors() for %s\n", getDescription_c());
}

//!
/*!
 * REMARKS:
 *      - Each neuron in the pre-synaptic layer can project "up"
 *      a number of arbors. Each arbor connects to a patch in the post-synaptic
 *      layer.
 *      - writeTime and writeStep are used to write post-synaptic patches.These
 *      patches are written every writeStep.
 *      .
 */
int HyPerConn::constructWeights() {
   int sx       = nfp;
   int sy       = sx * nxp;
   int sp       = sy * nyp;
   int nPatches = getNumDataPatches();
   int status   = PV_SUCCESS;

   // createArbors() uses the value of shrinkPatches.  It should have already been read in
   // ioParamsFillGroup.
   // allocate the arbor arrays:
   createArbors();

   setPatchStrides();

   ////allocate weight patches and axonal arbors for each arbor
   ////Allocate all the weights
   wDataStart[0] = allocWeights(nPatches, nxp, nyp, nfp);
   pvAssert(get_wDataStart(0) != NULL);

   for (int arborId = 0; arborId < numAxonalArborLists; arborId++) {
      status = createWeights(wPatches, arborId);
      pvAssert(wPatches[arborId] != NULL);

      if (arborId > 0) { // wDataStart already allocated
         wDataStart[arborId] = (get_wDataStart(0) + sp * nPatches * arborId);
         pvAssert(wDataStart[arborId] != NULL);
      }
      if (shrinkPatches_flag || arborId == 0) {
         status |= adjustAxonalArbors(arborId);
      }
   } // arborId

   // call to initializeWeights moved to BaseConnection::initializeState()
   status |= initPlasticityPatches();
   pvAssert(status == 0);
   if (shrinkPatches_flag) {
      for (int arborId = 0; arborId < numAxonalArborLists; arborId++) {
         shrinkPatches(arborId);
      }
   }

   return status;
}

int HyPerConn::shrinkPatches(int arborId) {
   int numPatches = getNumWeightPatches();
   for (int kex = 0; kex < numPatches; kex++) {
      shrinkPatch(kex, arborId /* arbor */);
   } // loop over pre-synaptic neurons

   return 0;
}

int HyPerConn::shrinkPatch(int kExt, int arborId) {

   int kIndex = patchToDataLUT(kExt);

   PVPatch *weights = getWeights(kExt, arborId);

   float *w = &get_wDataStart(arborId)[patchStartIndex(kIndex) + weights->offset];

   int nx = weights->nx;
   int ny = weights->ny;

   int maxnx = INT_MIN;
   int minnx = INT_MAX;
   int maxny = INT_MIN;
   int minny = INT_MAX;

   bool nonZeroWeightFound = false;
   // loop over all post-synaptic cells in patch
   for (int y = 0; y < ny; y++) {
      for (int x = 0; x < nx; x++) {
         for (int f = 0; f < nfp; f++) {
            if (fabsf(w[x * sxp + y * syp + f * sfp]) <= shrinkPatchesThresh) {
               nonZeroWeightFound = true;
               maxnx              = maxnx < x ? x : maxnx;
               minnx              = minnx > x ? x : minnx;
               maxny              = maxny < y ? y : maxny;
               minny              = minny > y ? y : minny;
            }
         }
      }
   }

   if (nonZeroWeightFound) {
      // Plus one to capture all of the patch
      int nxNew = maxnx + 1 - minnx;
      int nyNew = maxny + 1 - minny;
      int dxNew = minnx;
      int dyNew = minny;

      // adjust patch size (shrink) to fit within interior of post-synaptic layer
      //
      pvpatch_adjust(weights, sxp, syp, nxNew, nyNew, dxNew, dyNew);

      gSynPatchStart[arborId][kExt] +=
            dxNew * getPostNonextStrides()->sx + dyNew * getPostNonextStrides()->sy;
      aPostOffset[arborId][kExt] += dxNew * getPostExtStrides()->sx
                                    + dyNew * getPostExtStrides()->sy; // Someone who uses these
      // routines, please check
      // that this is correct.
   }
   return 0;
}

int HyPerConn::initialize(char const *name, HyPerCol *hc) {
   int status = BaseConnection::initialize(name, hc);

   pvAssert(parent);
   PVParams *inputParams = parent->parameters();

   // set accumulateFunctionPointer
   pvAssert(!inputParams->presentAndNotBeenRead(name, "pvpatchAccumulateType"));
   switch (pvpatchAccumulateType) {
      case CONVOLVE:
         accumulateFunctionPointer         = &pvpatch_accumulate;
         accumulateFunctionFromPostPointer = &pvpatch_accumulate_from_post;
         break;
      case STOCHASTIC:
         accumulateFunctionPointer         = &pvpatch_accumulate_stochastic;
         accumulateFunctionFromPostPointer = &pvpatch_accumulate_stochastic_from_post;
         break;
      default: pvAssertMessage(0, "Unrecognized pvpatchAccumulate type"); break;
   }

   mSparseWeightsAllocated.resize(numAxonalArborLists);
   std::fill(mSparseWeightsAllocated.begin(), mSparseWeightsAllocated.end(), false);

   return status;
}

void HyPerConn::setWeightInitializer() {
   FatalIf(
         weightInitTypeString == nullptr or weightInitTypeString[0] == '\0',
         "%s must set weightInitType.\n",
         getDescription_c());
   pvAssert(weightInitializer == nullptr);
   {
      BaseObject *baseObject = nullptr;
      try {
         baseObject = Factory::instance()->createByKeyword(weightInitTypeString, name, parent);
      } catch (const std::exception &e) {
         Fatal() << getDescription() << " unable to create weightInitializer: " << e.what() << "\n";
      }
      weightInitializer = dynamic_cast<InitWeights *>(baseObject);
      FatalIf(
            weightInitializer == nullptr,
            "%s unable to create weightInitializer: %s is not an InitWeights keyword.\n",
            getDescription_c(),
            weightInitTypeString);
   }
}

int HyPerConn::setPreLayerName(const char *pre_name) {
   pvAssert(parent != NULL);
   pvAssert(preLayerName == NULL);

   if (pre_name != NULL) {
      preLayerName = strdup(pre_name);
      pvAssertMessage(
            preLayerName != NULL,
            "%s: rank %d process unable to allocate memory for name of presynaptic layer \"%s\": "
            "%s",
            getDescription_c(),
            parent->columnId(),
            pre_name);
   }
   return PV_SUCCESS;
}

int HyPerConn::setPostLayerName(const char *post_name) {
   pvAssert(postLayerName == NULL);
   if (post_name != NULL) {
      postLayerName = strdup(post_name);
      pvAssertMessage(
            postLayerName != NULL,
            "%s: unable to allocate memory for name of postsynaptic layer \"%s\": %s",
            getDescription_c(),
            parent->columnId(),
            post_name,
            strerror(errno));
   }
   return PV_SUCCESS;
}

int HyPerConn::initNumWeightPatches() {
   numWeightPatches = pre->getNumExtended();
   return PV_SUCCESS;
}

int HyPerConn::initNumDataPatches() {
   if (sharedWeights) {
      mNumDataPatchesX = (pre->getXScale() < post->getXScale())
                               ? (int)pow(2, post->getXScale() - pre->getXScale())
                               : 1;
      mNumDataPatchesY = (pre->getYScale() < post->getYScale())
                               ? (int)pow(2, post->getYScale() - pre->getYScale())
                               : 1;
   }
   else {
      PVHalo const &preHalo = pre->getLayerLoc()->halo;
      mNumDataPatchesX      = pre->getLayerLoc()->nx + preHalo.lt + preHalo.rt;
      mNumDataPatchesY      = pre->getLayerLoc()->ny + preHalo.dn + preHalo.up;
   }
   mNumDataPatchesF = pre->getLayerLoc()->nf;
   numDataPatches   = mNumDataPatchesX * mNumDataPatchesY * mNumDataPatchesF;
   pvAssert(sharedWeights or (numDataPatches == pre->getNumExtended()));
   return PV_SUCCESS;
}

int HyPerConn::initPlasticityPatches() {
   if (!plasticityFlag)
      return PV_SUCCESS;
   int sx       = nfp;
   int sy       = sx * nxp;
   int sp       = sy * nyp;
   int nPatches = getNumDataPatches();

   const int numAxons = numberOfAxonalArborLists();

   if (combine_dW_with_W_flag) {
      dwDataStart = wDataStart;
      return PV_SUCCESS;
   }
   dwDataStart[0] = allocWeights(nPatches, nxp, nyp, nfp);
   pvAssert(get_dwDataStart(0) != NULL);
   for (int arborId = 0; arborId < numAxons; arborId++) {
      dwDataStart[arborId] = (dwDataStart[0] + sp * nPatches * arborId);
      pvAssert(get_dwDataStart(arborId) != NULL);
   } // loop over arbors

   if (sharedWeights && normalizeDwFlag) {
      std::size_t numWeights  = (std::size_t)(nxp * nyp * nfp) * (std::size_t)nPatches;
      numKernelActivations[0] = (long *)pvCalloc(numWeights, sizeof(long));
      for (int arborId = 0; arborId < numAxons; arborId++) {
         numKernelActivations[arborId] = (numKernelActivations[0] + sp * nPatches * arborId);
         pvAssert(get_dwDataStart(arborId) != NULL);
      } // loop over arbors
   }

   return PV_SUCCESS;
}

// set member variables specified by user
int HyPerConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   BaseConnection::ioParamsFillGroup(ioFlag);
   ioParam_sharedWeights(ioFlag);
   ioParam_weightInitType(ioFlag);
   if (weightInitializer != nullptr) {
      weightInitializer->ioParams(ioFlag, false, false);
   }
   ioParam_initializeFromCheckpointFlag(ioFlag);
   ioParam_triggerLayerName(ioFlag);
   ioParam_triggerOffset(ioFlag);
   ioParam_weightUpdatePeriod(ioFlag);
   ioParam_initialWeightUpdateTime(ioFlag);
   ioParam_immediateWeightUpdate(ioFlag);
   ioParam_updateGSynFromPostPerspective(ioFlag);
   ioParam_pvpatchAccumulateType(ioFlag);
   ioParam_writeStep(ioFlag);
   ioParam_initialWriteTime(ioFlag);
   ioParam_writeCompressedWeights(ioFlag);
   ioParam_writeCompressedCheckpoints(ioFlag);
   ioParam_selfFlag(ioFlag);
   ioParam_combine_dW_with_W_flag(ioFlag);
   ioParam_nxp(ioFlag);
   ioParam_nyp(ioFlag);
   ioParam_nfp(ioFlag);
   ioParam_shrinkPatches(ioFlag);
   ioParam_normalizeMethod(ioFlag);
   if (normalizer != nullptr && !strcmp(normalizer->getName(), getName())) {
      normalizer->ioParams(ioFlag, false, false);
   }
   ioParam_dWMax(ioFlag);

   ioParam_normalizeDw(ioFlag);
   ioParam_useMask(ioFlag);
   ioParam_maskLayerName(ioFlag);
   ioParam_maskFeatureIdx(ioFlag);
   ioParam_dWMaxDecayInterval(ioFlag);
   ioParam_dWMaxDecayFactor(ioFlag);

#ifdef PV_USE_CUDA
   ioParam_gpuGroupIdx(ioFlag);
#endif // PV_USE_CUDA
   // Weight sparsity
   ioParam_weightSparsity(ioFlag);
   return PV_SUCCESS;
}

#ifdef PV_USE_CUDA
void HyPerConn::ioParam_gpuGroupIdx(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "receiveGpu"));
   if (receiveGpu) {
      parent->parameters()->ioParamValue(
            ioFlag,
            name,
            "gpuGroupIdx",
            &mGpuGroupIdx,
            mGpuGroupIdx /*default*/,
            false /*warn if absent*/);
   }
}
#endif // PV_USE_CUDA

void HyPerConn::ioParam_channelCode(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      int ch = 0;
      parent->parameters()->ioParamValueRequired(ioFlag, name, "channelCode", &ch);
      int status = decodeChannel(ch, &channel);
      if (status != PV_SUCCESS) {
         if (parent->columnId() == 0) {
            ErrorLog().printf(
                  "%s: channelCode %d is not a valid channel.\n", getDescription_c(), ch);
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         pvAssert(0);
      }
   }
   else if (ioFlag == PARAMS_IO_WRITE) {
      int ch = (int)channel;
      parent->parameters()->ioParamValueRequired(ioFlag, name, "channelCode", &ch);
   }
   else {
      Fatal().printf("All possibilities of ioFlag are covered above.");
   }
}

void HyPerConn::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "receiveGpu"));
   parent->parameters()->ioParamValue(
         ioFlag, name, "sharedWeights", &sharedWeights, true /*default*/, true /*warn if absent*/);
   if (sharedWeights == false and receiveGpu == true) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog().printf("%s: sharedWeights must be true in order to receive on the GPU.\n", getDescription_c());
      }
      MPI_Barrier(parent->getCommunicator()->globalCommunicator());
      MPI_Finalize();
      exit(EXIT_FAILURE);
   }
}

void HyPerConn::ioParam_weightInitType(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamString(
         ioFlag, name, "weightInitType", &weightInitTypeString, NULL, true /*warnIfAbsent*/);
   if (ioFlag == PARAMS_IO_READ) {
      setWeightInitializer();
      pvAssertMessage(
            weightInitializer != nullptr,
            "%s: Rank %d process unable to construct weightInitializer",
            getDescription_c(),
            parent->columnId());
   }
}

void HyPerConn::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag) {
      parent->parameters()->ioParamString(
            ioFlag, name, "triggerLayerName", &triggerLayerName, NULL, false /*warnIfAbsent*/);
      if (ioFlag == PARAMS_IO_READ) {
         triggerFlag = (triggerLayerName != NULL && triggerLayerName[0] != '\0');
      }
   }
}

void HyPerConn::ioParam_triggerOffset(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag) {
      pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
      if (triggerFlag) {
         parent->parameters()->ioParamValue(
               ioFlag, name, "triggerOffset", &triggerOffset, triggerOffset);
         if (triggerOffset < 0) {
            Fatal().printf(
                  "%s error in rank %d process: TriggerOffset (%f) must be positive",
                  getDescription_c(),
                  parent->columnId(),
                  triggerOffset);
         }
      }
   }
}

void HyPerConn::ioParam_weightUpdatePeriod(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag) {
      pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
      if (!triggerLayerName) {
         parent->parameters()->ioParamValueRequired(
               ioFlag, name, "weightUpdatePeriod", &weightUpdatePeriod);
      }
      else
         FatalIf(
               parent->parameters()->present(name, "weightUpdatePeriod"),
               "%s sets both triggerLayerName and weightUpdatePeriod; "
               "only one of these can be set.\n",
               getDescription_c());
   }
}

void HyPerConn::ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag) {
      pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
      if (!triggerLayerName) {
         parent->parameters()->ioParamValue(
               ioFlag,
               name,
               "initialWeightUpdateTime",
               &initialWeightUpdateTime,
               initialWeightUpdateTime,
               true /*warnIfAbsent*/);
      }
   }
   if (ioFlag == PARAMS_IO_READ) {
      weightUpdateTime = initialWeightUpdateTime;
   }
}

void HyPerConn::ioParam_immediateWeightUpdate(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag) {
      parent->parameters()->ioParamValue(
            ioFlag,
            name,
            "immediateWeightUpdate",
            &mImmediateWeightUpdate,
            mImmediateWeightUpdate,
            true /*warnIfAbsent*/);
   }
}

void HyPerConn::ioParam_pvpatchAccumulateType(enum ParamsIOFlag ioFlag) {
   PVParams *params = parent->parameters();
   parent->parameters()->ioParamString(
         ioFlag, name, "pvpatchAccumulateType", &pvpatchAccumulateTypeString, "convolve");
   if (ioFlag == PARAMS_IO_READ) {
      if (pvpatchAccumulateTypeString == NULL) {
         unsetAccumulateType();
         return;
      }
      // Convert string to lowercase so that capitalization doesn't matter.
      for (char *c = pvpatchAccumulateTypeString; *c != '\0'; c++) {
         *c = (char)tolower((int)*c);
      }

      if (strcmp(pvpatchAccumulateTypeString, "convolve") == 0) {
         pvpatchAccumulateType = CONVOLVE;
      }
      else if (strcmp(pvpatchAccumulateTypeString, "stochastic") == 0) {
         pvpatchAccumulateType = STOCHASTIC;
      }
      else {
         unsetAccumulateType();
      }
   }
}

void HyPerConn::ioParam_dWMaxDecayInterval(enum ParamsIOFlag ioFlag) {
   if (plasticityFlag) {
      parent->parameters()->ioParamValue(
            ioFlag, name, "dWMaxDecayInterval", &mDWMaxDecayInterval, mDWMaxDecayInterval, false);
   }
}

void HyPerConn::ioParam_dWMaxDecayFactor(enum ParamsIOFlag ioFlag) {
   if (plasticityFlag) {
      parent->parameters()->ioParamValue(
            ioFlag, name, "dWMaxDecayFactor", &mDWMaxDecayFactor, mDWMaxDecayFactor, false);
      FatalIf(
            mDWMaxDecayFactor < 0.0f || mDWMaxDecayFactor >= 1.0f,
            "%s: dWMaxDecayFactor must be in the interval [0.0, 1.0)\n",
            getName());
   }
}

void HyPerConn::unsetAccumulateType() {
   if (parent->columnId() == 0) {
      if (pvpatchAccumulateTypeString) {
         ErrorLog().printf(
               "%s error: pvpatchAccumulateType \"%s\" is unrecognized.",
               getDescription_c(),
               pvpatchAccumulateTypeString);
      }
      else {
         ErrorLog().printf(
               "%s error: pvpatchAccumulateType NULL is unrecognized.", getDescription_c());
      }
      ErrorLog().printf("  Allowed values are \"convolve\" or \"stochastic\".");
   }
   MPI_Barrier(parent->getCommunicator()->communicator());
   exit(EXIT_FAILURE);
}

void HyPerConn::ioParam_writeStep(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "writeStep", &writeStep, parent->getDeltaTime());
}

void HyPerConn::ioParam_initialWriteTime(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "writeStep"));
   if (writeStep >= 0) {
      double start_time = parent->getStartTime();
      parent->parameters()->ioParamValue(
            ioFlag, name, "initialWriteTime", &initialWriteTime, start_time);
      if (ioFlag == PARAMS_IO_READ) {
         if (writeStep > 0 && initialWriteTime < start_time) {
            if (parent->columnId() == 0) {
               WarnLog(adjustInitialWriteTime);
               adjustInitialWriteTime.printf(
                     "%s: initialWriteTime %f earlier than starting time %f.  Adjusting "
                     "initialWriteTime:\n",
                     getDescription_c(),
                     initialWriteTime,
                     start_time);
               adjustInitialWriteTime.flush();
            }
            while (initialWriteTime < start_time) {
               initialWriteTime += writeStep;
            }
            if (parent->columnId() == 0) {
               InfoLog().printf(
                     "%s: initialWriteTime adjusted to %f\n", getDescription_c(), initialWriteTime);
            }
         }
         writeTime = initialWriteTime;
      }
   }
}

void HyPerConn::ioParam_writeCompressedWeights(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "writeStep"));
   if (writeStep >= 0) {
      parent->parameters()->ioParamValue(
            ioFlag,
            name,
            "writeCompressedWeights",
            &writeCompressedWeights,
            writeCompressedWeights,
            /*warnifabsent*/ true);
   }
}

void HyPerConn::ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag,
         name,
         "writeCompressedCheckpoints",
         &writeCompressedCheckpoints,
         writeCompressedCheckpoints,
         /*warnifabsent*/ true);
}

void HyPerConn::ioParam_selfFlag(enum ParamsIOFlag ioFlag) {
   // selfFlag indicates whether pre and post layers refer to the same neurons.
   // The default value for selfFlag should be pre==post, but at the time ioParams(PARAMS_IO_READ)
   // is called,
   // pre and post have not been set.  So we read the value with no warning if it's present;
   // if it's absent, set the value to pre==post in the communicateInitInfo stage and issue
   // the using-default-value warning then.
   parent->parameters()->ioParamValue(
         ioFlag, name, "selfFlag", &selfFlag, selfFlag, false /*warnIfAbsent*/);
}

void HyPerConn::ioParam_combine_dW_with_W_flag(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag) {
      parent->parameters()->ioParamValue(
            ioFlag,
            name,
            "combine_dW_with_W_flag",
            &combine_dW_with_W_flag,
            combine_dW_with_W_flag,
            true /*warnIfAbsent*/);
   }
}

void HyPerConn::ioParam_nxp(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "nxp", &nxp, 1);
}

void HyPerConn::ioParam_nyp(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "nyp", &nyp, 1);
}

void HyPerConn::ioParam_nfp(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "nfp", &nfp, -1, false);
   if (ioFlag == PARAMS_IO_READ && nfp == -1 && !parent->parameters()->present(name, "nfp")
       && parent->columnId() == 0) {
      InfoLog().printf(
            "%s: nfp will be set in the communicateInitInfo() stage.\n", getDescription_c());
   }
}

void HyPerConn::ioParam_shrinkPatches(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "shrinkPatches", &shrinkPatches_flag, shrinkPatches_flag);
}

void HyPerConn::ioParam_shrinkPatchesThresh(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "shrinkPatches"));
   if (shrinkPatches_flag) {
      parent->parameters()->ioParamValue(
            ioFlag, name, "shrinkPatchesThresh", &shrinkPatchesThresh, shrinkPatchesThresh);
   }
}

void HyPerConn::ioParam_updateGSynFromPostPerspective(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag,
         name,
         "updateGSynFromPostPerspective",
         &updateGSynFromPostPerspective,
         updateGSynFromPostPerspective);
}

void HyPerConn::ioParam_dWMax(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag) {
      parent->parameters()->ioParamValueRequired(ioFlag, name, "dWMax", &dWMax);
   }
}

void HyPerConn::ioParam_normalizeMethod(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamString(
         ioFlag, name, "normalizeMethod", &normalizeMethod, NULL, true /*warnIfAbsent*/);
   if (ioFlag == PARAMS_IO_READ) {
      if (normalizeMethod == NULL) {
         if (parent->columnId() == 0) {
            Fatal().printf(
                  "%s: specifying a normalizeMethod string is required.\n", getDescription_c());
         }
      }
      if (!strcmp(normalizeMethod, "")) {
         free(normalizeMethod);
         normalizeMethod = strdup("none");
      }
      if (strcmp(normalizeMethod, "none")) {
         int status = setWeightNormalizer();
         if (status != PV_SUCCESS) {
            Fatal().printf(
                  "%s: Rank %d process unable to construct weight normalizer\n",
                  getDescription_c(),
                  parent->columnId());
         }
      }
   }
}

void HyPerConn::ioParam_weightSparsity(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "weightSparsity", &mWeightSparsity, 0.0f, false);
}

int HyPerConn::setWeightNormalizer() {
   pvAssert(normalizer == nullptr);
   pvAssert(normalizeMethod != nullptr);
   pvAssertMessage(
         strcmp(normalizeMethod, "none"),
         "setWeightNormalizer() should not be called if normalizeMethod was \"none\"");
   BaseObject *baseObj = Factory::instance()->createByKeyword(normalizeMethod, name, parent);
   if (baseObj == nullptr) {
      if (parent->columnId() == 0) {
         Fatal() << getDescription_c() << ": normalizeMethod \"" << normalizeMethod
                 << "\" is not recognized." << std::endl;
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   normalizer = dynamic_cast<NormalizeBase *>(baseObj);
   if (normalizer == nullptr) {
      pvAssert(baseObj);
      if (parent->columnId() == 0) {
         Fatal() << getDescription_c() << ": normalizeMethod \"" << normalizeMethod
                 << "\" is not a normalization method." << std::endl;
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}

void HyPerConn::ioParam_normalizeDw(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag) {
      parent->parameters()->ioParamValue(
            ioFlag, getName(), "normalizeDw", &normalizeDwFlag, true, false /*warnIfAbsent*/);
   }
}

void HyPerConn::ioParam_useMask(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag) {
      parent->parameters()->ioParamValue(
            ioFlag, getName(), "useMask", &useMask, false, false /*warnIfAbsent*/);
   }
}

void HyPerConn::ioParam_maskLayerName(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag) {
      pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "useMask"));
      if (useMask) {
         parent->parameters()->ioParamStringRequired(ioFlag, name, "maskLayerName", &maskLayerName);
      }
   }
}

void HyPerConn::ioParam_maskFeatureIdx(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag) {
      pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "useMask"));
      if (useMask) {
         parent->parameters()->ioParamValue(
               ioFlag, name, "maskFeatureIdx", &maskFeatureIdx, maskFeatureIdx);
      }
   }
}

int HyPerConn::setPostPatchSize() {
   // If postConn is many-to-one, the transpose connection is one-to-many; then xscaleDiff > 0.
   // Similarly, if postConn is one-to-many, xscaleDiff < 0.

   // Some of the code duplication might be eliminated by adding some functions to convert.h

   pvAssert(pre && post);

   if (pre != NULL && post != NULL) {
      int xscaleDiff = post->getXScale() - pre->getXScale();
      int nxp_orig   = xPatchSize();
      int nyp_orig   = yPatchSize();
      nxpPost        = nxp_orig;
      if (xscaleDiff > 0) {
         nxpPost *= (int)pow(2, xscaleDiff);
      }
      else if (xscaleDiff < 0) {
         nxpPost /= (int)pow(2, -xscaleDiff);
         pvAssert(nxp_orig == nxpPost * pow(2, (float)(-xscaleDiff)));
      }

      int yscaleDiff = post->getYScale() - pre->getYScale();
      nypPost        = nyp_orig;
      if (yscaleDiff > 0) {
         nypPost *= (int)pow(2, yscaleDiff);
      }
      else if (yscaleDiff < 0) {
         nypPost /= (int)pow(2, -yscaleDiff);
         pvAssert(nyp_orig == nypPost * pow(2, (float)(-yscaleDiff)));
      }

      nfpPost = pre->getLayerLoc()->nf;
   }

   return PV_SUCCESS;
}

int HyPerConn::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   // HyPerConns need to tell the parent HyPerCol how many random number
   // seeds they need.  At the start of HyPerCol::run, the parent HyPerCol
   // calls each layer's and each connection's communicateInitInfo() sequentially in
   // a repeatable order (probably the order they appear in the params
   // file) to make sure that the same runs use the same RNG seeds in the
   // same way.
   //
   // HyPerConns need RNGs if they are using stochastic release flag, or if
   // their InitWeights method is random (e.g. UniformRandomWeights or
   // GaussianRandomWeights).
   //
   // HyPerConn also tells:
   // - its pre-synaptic layer how big a margin is needed
   // - its pre-synaptic layer how long a delay is needed in the data store
   // - its post-synaptic layer which channel it will deliver GSyn to.
   //
   // The routine also checks that nxp and nyp are consistent with
   // the relative densities of the pre and post layers, and that nfp is
   // consistent with the number of features of post.
   //
   // Subclasses (e.g. CloneConn) may also need
   // to send messages to related layers and connections before the allocation
   // phase.  These subclasses should override communicateInitInfo(), and the
   // subclass's communicateInitInfo() should call the parent class's communicateInitInfo().

   int status = BaseConnection::communicateInitInfo(message);
   if (status != PV_SUCCESS) {
      if (parent->columnId() == 0) {
         ErrorLog().printf("%s: communicateInitInfo failed.\n", getDescription_c());
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   pvAssert(preSynapticLayer() != NULL && postSynapticLayer() != NULL);
   handleDefaultSelfFlag();

   if (useMask) {
      mask = message->lookup<HyPerLayer>(std::string(maskLayerName));
      if (mask == NULL) {
         if (parent->columnId() == 0) {
            ErrorLog().printf(
                  "%s: maskLayerName \"%s\" does not correspond to a layer in the column.\n",
                  getDescription_c(),
                  maskLayerName);
         }
         status = PV_FAILURE;
         exit(EXIT_FAILURE);
      }
      // Check mask with restricted post layer
      const PVLayerLoc *maskLoc = mask->getLayerLoc();
      const PVLayerLoc *postLoc = post->getLayerLoc();
      if (postLoc->nx != maskLoc->nx || postLoc->ny != maskLoc->ny) {
         if (parent->columnId() == 0) {
            ErrorLog().printf(
                  "%s: Mask \"%s\" (%d, %d, %d) must have the same x and y size as post layer "
                  "\"%s\" (%d, %d, %d).\n",
                  getDescription_c(),
                  maskLayerName,
                  maskLoc->nx,
                  maskLoc->ny,
                  maskLoc->nf,
                  post->getName(),
                  postLoc->nx,
                  postLoc->ny,
                  postLoc->nf);
         }
         status = PV_FAILURE;
         exit(EXIT_FAILURE);
      }
      // Make sure maskFeatureIdx is within bounds
      if (maskFeatureIdx >= maskLoc->nf || maskFeatureIdx < -1) {
         ErrorLog().printf(
               "%s: maskFeatureIdx must be between -1 (inclusive) and mask layer \"%s\" (%d, %d, "
               "%d) nf dimension (exclusive)\n",
               getDescription_c(),
               maskLayerName,
               maskLoc->nx,
               maskLoc->ny,
               maskLoc->nf);
         status = PV_FAILURE;
         exit(EXIT_FAILURE);
      }

      // This check is only required if a maskFeatureIdx is not specified, aka, pointwise masking
      if (maskFeatureIdx == -1) {
         if (postLoc->nf != maskLoc->nf && maskLoc->nf != 1) {
            if (parent->columnId() == 0) {
               ErrorLog().printf(
                     "%s: Mask \"%s\" (%d, %d, %d) nf dimension must be either the same as post "
                     "layer \"%s\" (%d, %d, %d) or 1\n",
                     getDescription_c(),
                     maskLayerName,
                     maskLoc->nx,
                     maskLoc->ny,
                     maskLoc->nf,
                     post->getName(),
                     postLoc->nx,
                     postLoc->ny,
                     postLoc->nf);
            }
            status = PV_FAILURE;
            exit(EXIT_FAILURE);
         }
      }
   }

   if (getPvpatchAccumulateType() == STOCHASTIC
       && (getConvertRateToSpikeCount() || pre->activityIsSpiking())) {
      if (parent->columnId() == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "%s: stochastic accumulation function is not consistent with ", getDescription_c());
         if (getConvertRateToSpikeCount()) {
            errorMessage.printf("setting convertRateToSpikeCount to true.\n");
         }
         else {
            pvAssert(pre->activityIsSpiking());
            errorMessage.printf("a spiking presynaptic layer \"%s\".\n", pre->getName());
         }
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      status = PV_FAILURE;
      exit(EXIT_FAILURE);
   }

   status = setPatchSize();
   status = checkPatchDimensions();

   PVLayerLoc const *preLoc  = pre->getLayerLoc();
   PVLayerLoc const *postLoc = post->getLayerLoc();
   if (nfp == -1) {
      nfp = postLoc->nf;
      if (warnDefaultNfp && parent->columnId() == 0) {
         InfoLog().printf(
               "%s setting nfp to number of postsynaptic features = %d.\n",
               getDescription_c(),
               nfp);
      }
   }
   if (nfp != postLoc->nf) {
      if (parent->columnId() == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "Params file specifies %d features for %s,\n", nfp, getDescription_c());
         errorMessage.printf(
               "but %d features for post-synaptic layer %s\n", postLoc->nf, post->getName());
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(PV_FAILURE);
   }
   // Currently, the only acceptable number for nfp is the number of post-synaptic features.
   // However, we may add flexibility on this score in the future, e.g. MPI in feature space
   // with each feature connecting to only a few nearby features.
   // Accordingly, we still keep readNfp.

   int xmargin         = requiredConvolveMargin(preLoc->nx, postLoc->nx, nxp);
   int receivedxmargin = 0;
   int statusx         = pre->requireMarginWidth(xmargin, &receivedxmargin, 'x');
   if (statusx != PV_SUCCESS) {
      ErrorLog().printf(
            "Margin Failure for layer %s.  Received x-margin is %d, but %s requires margin of at "
            "least %d\n",
            pre->getDescription_c(),
            receivedxmargin,
            name,
            xmargin);
      status = PV_MARGINWIDTH_FAILURE;
   }
   int ymargin         = requiredConvolveMargin(preLoc->ny, postLoc->ny, nyp);
   int receivedymargin = 0;
   int statusy         = pre->requireMarginWidth(ymargin, &receivedymargin, 'y');
   if (statusy != PV_SUCCESS) {
      ErrorLog().printf(
            "Margin Failure for layer %s.  Received y-margin is %d, but %s requires margin of at "
            "least %d\n",
            pre->getDescription_c(),
            receivedymargin,
            name,
            ymargin);
      status = PV_MARGINWIDTH_FAILURE;
   }

   status = setPostPatchSize();

   // Trigger stuff
   if (triggerLayerName) {
      triggerLayer = message->lookup<HyPerLayer>(std::string(triggerLayerName));
      if (triggerLayer == NULL) {
         if (parent->columnId() == 0) {
            ErrorLog().printf(
                  "%s error: triggerLayer \"%s\" is not a layer in the HyPerCol.\n",
                  getDescription_c(),
                  triggerLayerName);
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }

      // Although weightUpdatePeriod and weightUpdateTime is being set here, if trigger flag is set,
      // they are not being used
      // Only updating for backwards compatibility
      weightUpdatePeriod = triggerLayer->getDeltaUpdateTime();
      if (weightUpdatePeriod <= 0) {
         if (plasticityFlag == true) {
            WarnLog() << "Connection " << name << "triggered layer " << triggerLayerName
                      << " never updates, turning plasticity flag off\n";
            plasticityFlag = false;
         }
      }
      if (weightUpdatePeriod != -1 && triggerOffset >= weightUpdatePeriod) {
         Fatal().printf(
               "%s, rank %d process: TriggerOffset (%f) must be lower than the change in update "
               "time (%f) of the attached trigger layer\n",
               getDescription_c(),
               parent->columnId(),
               triggerOffset,
               weightUpdatePeriod);
      }
      weightUpdateTime = parent->getDeltaTime();
   }

   if (weightInitializer) {
      status = weightInitializer->respond(message);
      // // TODO: try to make this even more clumsy a hack.
      // status = weightInitializer->respond(
      //       std::make_shared<CommunicateInitInfoMessage>(message->mHierarchy));
      FatalIf(
            status != PV_SUCCESS,
            "%s failed CommunicateInitInfo stage.\n",
            weightInitializer->getDescription_c());
   }

   if (sharedWeights) {
      fileType = PVP_KERNEL_FILE_TYPE;
   }
   else {
      fileType = PVP_WGT_FILE_TYPE;
   }

   if (normalizer) {
      normalizer->communicateInitInfo(message);
   }

   // Check if need transpose
   if (updateGSynFromPostPerspective) {
      needPost = true;
   }

// GPU stuff
#ifdef PV_USE_CUDA
   // Here, the connection tells all participating recev layers to allocate memory on gpu
   // if receive from gpu is set. These buffers should be set in allocate
   if (receiveGpu) {
      if (mGpuGroupIdx >= 0) {
         // Scan all the connections to see if any with this group index have set
         // mGpuGroupHead. If so, set mGpuGroupHead to the same thing; otherwise,
         // set mGpuGroupHead to itself (this depends on communicateInitInfo running serially).
         for (auto &obj : message->mHierarchy) {
            auto c = dynamic_cast<HyPerConn *>(obj.second);
            if (c == nullptr) {
               continue;
            }
            if (c == this) {
               continue;
            }
            if (c->getGpuGroupIdx() == mGpuGroupIdx) {
               HyPerConn *groupHead = c->getGpuGroupHead();
               if (groupHead != nullptr) {
                  mGpuGroupHead = groupHead;
                  break;
               }
            }
         }
         if (mGpuGroupHead == nullptr) {
            mGpuGroupHead == this;
         }
      }

      // we need pre datastore, this conn's weights, and post gsyn on the channel of this connection
      pre->setAllocDeviceDatastore();
      if (updateGSynFromPostPerspective) {
         setAllocPostDeviceWeights();
         // Increment number of postKernels for workspace memory
         parent->getDevice()->incrementConvKernels();
      }
      else {
         setAllocDeviceWeights();
      }
      post->setAllocDeviceGSyn();

      // If recv from pre and pre layer is sparse, allocate activeIndices
      if (!updateGSynFromPostPerspective && pre->getSparseFlag()) {
         pre->setAllocDeviceActiveIndices();
      }
   }
#endif

   return status;
}

int HyPerConn::allocatePostToPreBuffer() {
   if (postToPreActivity) {
      return PV_SUCCESS;
   }
   // update conn to original connection
   const PVLayerLoc *sourceLoc = preSynapticLayer()->getLayerLoc();
   const PVLayerLoc *targetLoc = postSynapticLayer()->getLayerLoc();

   const int sourceNx = sourceLoc->nx;
   const int sourceNy = sourceLoc->ny;
   const int sourceNf = sourceLoc->nf;
   const int targetNx = targetLoc->nx;
   const int targetNy = targetLoc->ny;
   const int targetNf = targetLoc->nf;

   float sourceToTargetScaleX = (float)sourceNx / targetNx;
   float sourceToTargetScaleY = (float)sourceNy / targetNy;

   const PVHalo *sourceHalo = &sourceLoc->halo;

   const int numRestricted = postSynapticLayer()->getNumNeurons();

   postToPreActivity = (long *)pvMalloc(sizeof(long) * numRestricted);

   // origpre many, origpost one
   if (sourceToTargetScaleX >= 1 && sourceToTargetScaleY >= 1) {
      for (int kTargetRes = 0; kTargetRes < numRestricted; kTargetRes++) {
         int kTargetXRes = kxPos(kTargetRes, targetNx, targetNy, targetNf);
         int kTargetYRes = kyPos(kTargetRes, targetNx, targetNy, targetNf);

         int xVal = (sourceToTargetScaleX * kTargetXRes)
                    - ((postConn->xPatchSize() - sourceToTargetScaleX) / 2);
         int yVal = (sourceToTargetScaleY * kTargetYRes)
                    - ((postConn->yPatchSize() - sourceToTargetScaleY) / 2);

         postToPreActivity[kTargetRes] = kIndex(
               xVal + sourceHalo->lt,
               yVal + sourceHalo->up,
               0,
               sourceNx + sourceHalo->lt + sourceHalo->rt,
               sourceNy + sourceHalo->up + sourceHalo->dn,
               sourceNf);
      }
   }

   // origpost many, origpre one
   else if (sourceToTargetScaleX <= 1 && sourceToTargetScaleY <= 1) {
      int targetToSourceScaleX = (float)1 / sourceToTargetScaleX;
      int targetToSourceScaleY = (float)1 / sourceToTargetScaleY;
      for (int kTargetRes = 0; kTargetRes < numRestricted; kTargetRes++) {
         int kTargetXRes = kxPos(kTargetRes, targetNx, targetNy, targetNf);
         int kTargetYRes = kyPos(kTargetRes, targetNx, targetNy, targetNf);

         int centerX = floor((float)kTargetXRes / ((float)targetToSourceScaleX / 2));
         int centerY = floor((float)kTargetYRes / ((float)targetToSourceScaleY / 2));
         int offsetX = postConn->xPatchSize() - 1;
         int offsetY = postConn->yPatchSize() - 1;

         int xVal                      = floor(((float)centerX - offsetX) / 2);
         int yVal                      = floor(((float)centerY - offsetY) / 2);
         postToPreActivity[kTargetRes] = kIndex(
               xVal + sourceHalo->lt,
               yVal + sourceHalo->up,
               0,
               sourceNx + sourceHalo->lt + sourceHalo->rt,
               sourceNy + sourceHalo->up + sourceHalo->dn,
               sourceNf);
      }
   }
   else {
      Fatal().printf(
            "sourceToTargetScaleX= %f, sourceToTargetScaleY= %f: the case of many-to-one in one "
            "dimension and one-to-many in the other has not yet been implemented.\n",
            (double)sourceToTargetScaleX,
            (double)sourceToTargetScaleY);
   }

   return PV_SUCCESS;
}

void HyPerConn::handleDefaultSelfFlag() {
   if (!parent->parameters()->present(name, "selfFlag")) {
      selfFlag = (pre == post);
   }
}

int HyPerConn::setPatchSize() {
   int status = PV_SUCCESS;
   // Some subclasses determine some of {nxp, nyp, nfp} from other layers or connections (e.g.
   // TransposeConn, CloneConn) instead of reading them from params.
   // They should override setPatchSize() to set those params.
   return status;
}

// returns handle to initialized weight patches
PVPatch ***HyPerConn::initializeWeights(PVPatch ***patches, float **dataStart) {
   if (weightInitializer) {
      PVPatch ***patches_arg = sharedWeights ? nullptr : patches;
      weightInitializer->initializeWeights(patches_arg, dataStart);
   }
   return patches;
}

int HyPerConn::allocatePostConn() {
   int status = PV_SUCCESS;
   // Allocate private transpose conn
   if (needPost) {
      char privateConnName[PV_PATH_MAX];
      sprintf(privateConnName, "&%s_privatePostConn", name);
      postConn = new privateTransposeConn(privateConnName, parent, this);
      pvAssert(postConn);
      status = postConn->allocateDataStructures();
   }
   // Can't do this with shrink patches flag
   if (needPost && !shrinkPatches_flag) {
      status = allocatePostToPreBuffer();
      postConn->allocatePostToPreBuffer();
   }
   return status;
}

int HyPerConn::allocateDataStructures() {
   int status = BaseConnection::allocateDataStructures();
   pvAssert(status == PV_SUCCESS);
#ifdef PV_USE_CUDA
   if (receiveGpu) {
      if (!pre->getDataStructuresAllocatedFlag()) {
         if (parent->getCommunicator()->commRank() == 0) {
            InfoLog() << getDescription() << " must wait until presynaptic layer \""
                      << pre->getName() << "\" has finished its allocateDataStructures stage.\n";
         }
         status = PV_POSTPONE;
      }
      if (!post->getDataStructuresAllocatedFlag()) {
         if (parent->getCommunicator()->commRank() == 0) {
            InfoLog() << getDescription() << " must wait until postsynaptic layer \""
                      << post->getName() << "\" has finished its allocateDataStructures stage.\n";
         }
         status = PV_POSTPONE;
      }
      if (status == PV_POSTPONE) {
         return status;
      }
   }
   if (status == 0) {
      status = PV_SUCCESS;
   }
   else {
      Fatal().printf(
            "%s: unable to allocate device memory in rank %d process: %s\n",
            getDescription_c(),
            parent->columnId(),
            strerror(errno));
   }
#endif // PV_USE_CUDA
   initNumWeightPatches();
   initNumDataPatches();
   initPatchToDataLUT();

   if (pvpatchAccumulateType == STOCHASTIC) {
      bool from_post = getUpdateGSynFromPostPerspective();
      if (from_post) {
         randState = new Random(postSynapticLayer()->getLayerLoc(), false /*isExtended*/);
      }
      else {
         randState = new Random(preSynapticLayer()->getLayerLoc(), true /*isExtended*/);
      }
   }

   if (plasticityFlag && !triggerLayer) {
      if (weightUpdateTime < parent->simulationTime()) {
         while (weightUpdateTime <= parent->simulationTime()) {
            weightUpdateTime += weightUpdatePeriod;
         }
         if (parent->columnId() == 0) {
            WarnLog().printf(
                  "initialWeightUpdateTime of %s less than simulation start time.  Adjusting "
                  "weightUpdateTime to %f\n",
                  getDescription_c(),
                  weightUpdateTime);
         }
      }
      lastUpdateTime = weightUpdateTime - parent->getDeltaTime();
   }
   lastTimeUpdateCalled = parent->simulationTime();

   status = constructWeights();

   allocatePostConn();

#ifdef PV_USE_CUDA
   status = allocateDeviceBuffers();
   if (receiveGpu) {
      if (updateGSynFromPostPerspective) {
         status = initializeReceivePostKernelArgs();
      }
      else {
         status = initializeReceivePreKernelArgs();
      }
   }
   if (status == 0) {
      status = PV_SUCCESS;
   }
   else {
      Fatal().printf(
            "%s: unable to allocate device memory in rank %d process: %s\n",
            getDescription_c(),
            parent->columnId(),
            strerror(errno));
   }
#endif // PV_USE_CUDA

   // Allocate temp buffers if needed, 1 for each thread
   // Only allocate for recv from pre, and not threading over batches
   if (!getUpdateGSynFromPostPerspective() && parent->getNumThreads() > 1) {
      // thread_gSyn is only a buffer for one batch, as if we're not threading over batches, batches
      // will be sequential
      thread_gSyn = (float **)pvMalloc(sizeof(float *) * parent->getNumThreads());

      // Assign thread_gSyn to different points of tempMem
      for (int i = 0; i < parent->getNumThreads(); i++) {
         thread_gSyn[i] = (float *)pvMallocError(
               sizeof(float) * post->getNumNeurons(),
               "%s: rank %d unable to allocate %zu memory for thread_gSyn",
               getDescription_c(),
               parent->columnId(),
               sizeof(float) * post->getNumNeurons());
      }
   }

   // Allocate batchSkip buffer
   batchSkip = (bool *)pvMallocError(
         parent->getNBatch() * sizeof(bool),
         "%s: rank %d unable to allocate %zu memory for batchSkip",
         getDescription_c(),
         parent->columnId(),
         sizeof(bool) * parent->getNBatch());

   for (int i = 0; i < parent->getNBatch(); i++) {
      batchSkip[i] = false;
   }

   if (mDWMaxDecayInterval > 0) {
      mDWMaxDecayTimer = mDWMaxDecayInterval;
   }

   return status;
}

void HyPerConn::initPatchToDataLUT() {
   pvAssert(patch2datalookuptable == NULL);
   if (sharedWeights) {
      int numWeightPatches = getNumWeightPatches();

      patch2datalookuptable = (int *)pvCalloc(numWeightPatches, sizeof(int));

      for (int i = 0; i < numWeightPatches; i++) {
         int kernelindex          = patchIndexToDataIndex(i);
         patch2datalookuptable[i] = kernelindex;
      }
   }
   else {
      // lookuptable just returns the patchindex
   }
}

taus_uint4 *HyPerConn::getRandState(int index) {
   taus_uint4 *state = NULL;
   if (pvpatchAccumulateType == STOCHASTIC) {
      state = randState->getRNG(index);
   }
   return state;
}

#ifdef PV_USE_CUDA

int HyPerConn::allocatePostDeviceWeights() {
   pvAssert(postConn);
   postConn->allocateDeviceWeights();
   return PV_SUCCESS;
}

int HyPerConn::allocateDeviceWeights() {
   PVCuda::CudaDevice *device = parent->getDevice();
   const size_t size          = numberOfAxonalArborLists() * getNumDataPatches() * xPatchSize()
                       * yPatchSize() * fPatchSize() * sizeof(float);
   d_WData = device->createBuffer(size, &description);
   pvAssert(d_WData);
#ifdef PV_USE_CUDNN
   cudnn_WData = device->createBuffer(size, &description);
#endif
   return PV_SUCCESS;
}

int HyPerConn::allocateDeviceBuffers() {
   int status = 0;

   PVCuda::CudaDevice *device = parent->getDevice();

   bool needAlloc = true;
   if (allocDeviceWeights || allocPostDeviceWeights) {
      // Check group here
      if (mGpuGroupIdx >= 0) {
         pvAssert(mGpuGroupHead);
         // If this connection is NOT the "base" group conn that allocates
         // check dims and don't allocate
         if (mGpuGroupHead != this) {
            // Different num arbors is okay, since GPU mem holds only one arbor at a time
            // nxp, nyp, nfp, numKernels all have to be the same
            if (mGpuGroupHead->xPatchSize() != xPatchSize()
                || mGpuGroupHead->yPatchSize() != yPatchSize()
                || mGpuGroupHead->fPatchSize() != fPatchSize()
                || mGpuGroupHead->getNumDataPatches() != getNumDataPatches()
                || mGpuGroupHead->numberOfAxonalArborLists() != numberOfAxonalArborLists()) {
               Fatal() << "Connection " << getName() << " of size (" << numberOfAxonalArborLists()
                       << ", " << getNumDataPatches() << ", " << xPatchSize() << ", "
                       << yPatchSize() << ", " << fPatchSize()
                       << ") does not match the gpuGroupConnection " << mGpuGroupHead->getName()
                       << " of size (" << mGpuGroupHead->numberOfAxonalArborLists() << ", "
                       << mGpuGroupHead->getNumDataPatches() << ", " << mGpuGroupHead->xPatchSize()
                       << ", " << mGpuGroupHead->yPatchSize() << ", " << mGpuGroupHead->fPatchSize()
                       << ").\n";
            }
            // set d_WData to the group's d_WData
            d_WData = mGpuGroupHead->getDeviceWData();
            pvAssert(d_WData);
#ifdef PV_USE_CUDNN
            cudnn_WData = mGpuGroupHead->getCudnnWData();
            pvAssert(cudnn_WData);
#endif
            needAlloc = false;
         }
      }

      if (needAlloc) {
         if (allocPostDeviceWeights) {
            allocatePostDeviceWeights();
         }
         if (allocDeviceWeights) {
            allocateDeviceWeights();
         }
      }
   }

   if (receiveGpu) {
      if (updateGSynFromPostPerspective) {
         int numPostRes      = post->getNumNeurons();
         d_PostToPreActivity = device->createBuffer(numPostRes * sizeof(long), &description);
         if (sharedWeights) {
            int numWeightPatches = postConn->getNumWeightPatches();
            d_Patch2DataLookupTable =
                  device->createBuffer(numWeightPatches * sizeof(int), &description);
         }
      }
      else {
         // Calculate local pre size here
         const PVLayerLoc *preLoc  = pre->getLayerLoc();
         const PVLayerLoc *postLoc = post->getLayerLoc();
         PVHalo haloPre;
         PVHalo haloPost;

         // Set local sizes here
         float preToPostScaleX = (float)preLoc->nx / ((float)postLoc->nx);
         float preToPostScaleY = (float)preLoc->ny / ((float)postLoc->ny);

         int preNf  = preLoc->nf;
         int postNf = postLoc->nf;

         // This should be the case with petavision restrictions
         pvAssert(postNf == nfp);

         int numWeightPatches = pre->getNumExtended();
         int patchSize        = numWeightPatches * sizeof(PVPatch);
         d_Patches            = device->createBuffer(patchSize, &description);

         // Need a buffer for gsynpatch start for one arbor
         int gsynPatchStartIndexSize = numWeightPatches * sizeof(size_t);
         d_GSynPatchStart            = device->createBuffer(gsynPatchStartIndexSize, &description);

         if (numberOfAxonalArborLists() == 1) {
            PVPatch *h_patches            = weights(0)[0]; // 0 beacuse it's one block of memory
            PVCuda::CudaBuffer *d_patches = getDevicePatches();
            pvAssert(d_patches);
            d_patches->copyToDevice(h_patches);
            size_t *h_GSynPatchStart             = getGSynPatchStart()[0];
            PVCuda::CudaBuffer *d_GSynPatchStart = getDeviceGSynPatchStart();
            pvAssert(d_GSynPatchStart);
            d_GSynPatchStart->copyToDevice(h_GSynPatchStart);
         }

         if (sharedWeights) {
            int numWeightPatches = getNumWeightPatches();
            d_Patch2DataLookupTable =
                  device->createBuffer(numWeightPatches * sizeof(int), &description);
         }
      }
   }
   return status;
}

int HyPerConn::initializeReceivePreKernelArgs() {
   int status                 = 0;
   PVCuda::CudaDevice *device = parent->getDevice();
   krRecvPre                  = new PVCuda::CudaRecvPre(device);

   const PVLayerLoc *preLoc  = pre->getLayerLoc();
   const PVLayerLoc *postLoc = post->getLayerLoc();
   const PVHalo *preHalo     = &pre->getLayerLoc()->halo;
   const PVHalo *postHalo    = &post->getLayerLoc()->halo;

   PVCuda::CudaBuffer *d_PreData  = pre->getDeviceDatastore();
   PVCuda::CudaBuffer *d_PostGSyn = post->getDeviceGSyn();

   pvAssert(d_PreData);
   pvAssert(d_PostGSyn);

   pvAssert(getDeviceWData());
   pvAssert(d_Patches);
   pvAssert(d_GSynPatchStart);

   int nxp             = xPatchSize();
   int nyp             = yPatchSize();
   int nfp             = fPatchSize();
   float dtFactor      = getConvertToRateDeltaTimeFactor();
   int i_sharedWeights = sharedWeights;

   int sy  = getPostNonextStrides()->sy;
   int syw = yPatchStride();

   bool isSparse = pre->getSparseFlag();

   int numPreExt  = pre->getNumExtended();
   int numPostRes = post->getNumNeurons();

   int nbatch = postLoc->nbatch;

   PVCuda::CudaBuffer *d_activeIndices = NULL;
   PVCuda::CudaBuffer *d_numActive     = NULL;
   if (isSparse) {
      d_numActive = pre->getDeviceNumActive();
      pvAssert(d_numActive);
      d_activeIndices = pre->getDeviceActiveIndices();
      pvAssert(d_activeIndices);
   }

   // Since it never changes, set this buffer here
   d_Patch2DataLookupTable->copyToDevice(getPatchToDataLUT());

   krRecvPre->setArgs(
         nbatch,
         numPreExt,
         numPostRes,
         nxp,
         nyp,
         nfp,

         sy,
         syw,
         dtFactor,
         i_sharedWeights,
         d_Patches,
         d_GSynPatchStart,

         d_PreData,
         getDeviceWData(),
         d_PostGSyn,
         d_Patch2DataLookupTable,

         isSparse,
         d_numActive,
         d_activeIndices);
   return status;
}

int HyPerConn::initializeReceivePostKernelArgs() {
   InfoLog() << name << " setting up post kernel\n";
   int status                 = 0;
   PVCuda::CudaDevice *device = parent->getDevice();
   krRecvPost                 = new PVCuda::CudaRecvPost(device);

   const PVLayerLoc *preLoc  = pre->getLayerLoc();
   const PVLayerLoc *postLoc = post->getLayerLoc();
   const PVHalo *preHalo     = &pre->getLayerLoc()->halo;
   const PVHalo *postHalo    = &post->getLayerLoc()->halo;

   PVCuda::CudaBuffer *d_PreData   = pre->getDeviceDatastore();
   PVCuda::CudaBuffer *d_PostGSyn  = post->getDeviceGSyn();
   PVCuda::CudaBuffer *d_origWData = postConn->getDeviceWData();

#ifdef PV_USE_CUDNN
   PVCuda::CudaBuffer *cudnn_preData   = pre->getCudnnDatastore();
   PVCuda::CudaBuffer *cudnn_gSyn      = post->getCudnnGSyn();
   PVCuda::CudaBuffer *cudnn_origWData = postConn->getCudnnWData();
   pvAssert(cudnn_preData);
   pvAssert(cudnn_gSyn);
   pvAssert(cudnn_origWData);
#endif

   pvAssert(d_PreData);
   pvAssert(d_PostGSyn);
   pvAssert(d_origWData);

   int sy              = (preLoc->nx + preHalo->rt + preHalo->lt) * preLoc->nf;
   int syp             = postConn->yPatchStride();
   int numPerStride    = postConn->xPatchSize() * postConn->fPatchSize();
   float dtFactor      = getConvertToRateDeltaTimeFactor();
   int i_sharedWeights = sharedWeights;

   const PVHalo *oHalo = &postConn->preSynapticLayer()->getLayerLoc()->halo;
   int oNblt           = oHalo->lt;
   int oNbrt           = oHalo->rt;
   int oNbup           = oHalo->up;
   int oNbdn           = oHalo->dn;

   // nxp, nyp, and nfp should be orig conn's
   int oNxp   = postConn->xPatchSize();
   int oNyp   = postConn->yPatchSize();
   int oNfp   = postConn->fPatchSize();
   int postNx = postLoc->nx;
   int postNy = postLoc->ny;
   int postNf = postLoc->nf;

   int preNx   = preLoc->nx;
   int preNy   = preLoc->ny;
   int preNf   = preLoc->nf;
   int preNblt = preHalo->lt;
   int preNbrt = preHalo->rt;
   int preNbup = preHalo->up;
   int preNbdn = preHalo->dn;

   int nbatch = preLoc->nbatch;

   // Set local sizes here
   float preToPostScaleX = (float)preLoc->nx / ((float)postLoc->nx);
   float preToPostScaleY = (float)preLoc->ny / ((float)postLoc->ny);

   // Since it never changes, set this buffer here
   // Need to set orig connection's patch2datalookuptable
   d_PostToPreActivity->copyToDevice(getPostToPreActivity());

   d_Patch2DataLookupTable->copyToDevice(postConn->getPatchToDataLUT());

   // See the size of buffer needed based on x and y
   // oNxp is the patch size from the post point of view

   if (parent->columnId() == 0) {
      InfoLog() << "preToPostScale: (" << preToPostScaleX << "," << preToPostScaleY << ")\n";
   }

   krRecvPost->setArgs(
         nbatch,
         postNx, // num post neurons
         postNy,
         postNf,

         oNblt, // Border of orig
         oNbrt, // Border of orig
         oNbdn, // Border of orig
         oNbup, // Border of orig

         preNx,
         preNy,
         preNf,
         preNblt,
         preNbrt,
         preNbup,
         preNbdn,

         oNxp,
         oNyp,
         oNfp,

         preToPostScaleX,
         preToPostScaleY,

         sy,
         syp,
         numPerStride,
         dtFactor,
         i_sharedWeights,

         d_PostToPreActivity,
         d_PreData,
         d_origWData,
         d_PostGSyn,
#ifdef PV_USE_CUDNN
         cudnn_preData,
         cudnn_origWData,
         cudnn_gSyn,
#endif
         d_Patch2DataLookupTable);
   return status;
}
#endif

int HyPerConn::writeWeights(double timed) {
   PVPatch ***patches_arg = sharedWeights ? NULL : wPatches;
   return writeWeights(
         patches_arg,
         get_wDataStart(),
         getNumDataPatches(),
         mOutputStateStream,
         timed,
         writeCompressedWeights,
         false);
}

int HyPerConn::writeWeights(const char *filename, bool verifyWrites) {
   PVPatch ***patches_arg = sharedWeights ? NULL : wPatches;
   FileStream *fileStream = nullptr;
   if (getMPIBlock()->getRank() == 0) {
      fileStream = new FileStream(filename, std::ios_base::out, verifyWrites);
   }

   int status = writeWeights(
         patches_arg,
         get_wDataStart(),
         getNumDataPatches(),
         fileStream,
         parent->simulationTime(),
         writeCompressedWeights,
         true);
   delete fileStream;
   return 0;
}

int HyPerConn::writeWeights(
      PVPatch ***patches,
      float **dataStart,
      int numPatches,
      FileStream *fileStream,
      double timed,
      bool compressWeights,
      bool last) {

   float minVal = FLT_MAX;
   float maxVal = -FLT_MAX;
   for (int arbor = 0; arbor < numberOfAxonalArborLists(); arbor++) {
      float minVal1 = minWeight(arbor);
      if (minVal1 < minVal)
         minVal     = minVal1;
      float maxVal1 = maxWeight(arbor);
      if (maxVal1 > maxVal)
         maxVal = maxVal1;
   }

   const PVLayerLoc *preLoc  = pre->getLayerLoc();
   const PVLayerLoc *postLoc = post->getLayerLoc();

   if (sharedWeights) {
      writeSharedWeights(
            timed,
            fileStream,
            getMPIBlock(),
            preLoc,
            nxp,
            nyp,
            nfp,
            numberOfAxonalArborLists(),
            dataStart,
            compressWeights,
            minVal,
            maxVal,
            mNumDataPatchesX,
            mNumDataPatchesY,
            mNumDataPatchesF);
   }
   else {
      writeNonsharedWeights(
            timed,
            fileStream,
            getMPIBlock(),
            preLoc,
            nxp,
            nyp,
            nfp,
            numberOfAxonalArborLists(),
            dataStart,
            compressWeights,
            true /*extended*/,
            postLoc,
            wPatches);
   }

   return PV_SUCCESS;
}

int HyPerConn::writeTextWeights(const char *path, bool verifyWrites, int k) {
   if (parent->getCommunicator()->commSize() > 1) {
      Fatal().printf(
            "writeTextWeights error for %s: writeTextWeights is not compatible with MPI",
            getDescription_c());
      // NOTE : if run under MPI when more than one process sees the same file system, the
      // contending processes will clobber each other.
   }
   PrintStream *outStream = nullptr;

   if (path != nullptr) {
      outStream = new FileStream(path, std::ios_base::out, verifyWrites);
   }
   else {
      outStream = new PrintStream(getOutputStream());
   }

   outStream->printf("Weights for %s, neuron %d\n", getDescription_c(), k);
   outStream->printf(
         "   (kxPre,kyPre,kfPre)   = (%i,%i,%i)\n",
         kxPos(k,
               pre->getLayerLoc()->nx + pre->getLayerLoc()->halo.lt + pre->getLayerLoc()->halo.rt,
               pre->getLayerLoc()->ny + pre->getLayerLoc()->halo.dn + pre->getLayerLoc()->halo.up,
               pre->getLayerLoc()->nf),
         kyPos(k,
               pre->getLayerLoc()->nx + pre->getLayerLoc()->halo.lt + pre->getLayerLoc()->halo.rt,
               pre->getLayerLoc()->ny + pre->getLayerLoc()->halo.dn + pre->getLayerLoc()->halo.up,
               pre->getLayerLoc()->nf),
         featureIndex(
               k,
               pre->getLayerLoc()->nx + pre->getLayerLoc()->halo.lt + pre->getLayerLoc()->halo.rt,
               pre->getLayerLoc()->ny + pre->getLayerLoc()->halo.dn + pre->getLayerLoc()->halo.up,
               pre->getLayerLoc()->nf));
   outStream->printf("   (nxp,nyp,nfp)   = (%i,%i,%i)\n", (int)nxp, (int)nyp, (int)nfp);
   outStream->printf(
         "   pre  (nx,ny,nf) = (%i,%i,%i)\n",
         pre->getLayerLoc()->nx,
         pre->getLayerLoc()->ny,
         pre->getLayerLoc()->nf);
   outStream->printf(
         "   post (nx,ny,nf) = (%i,%i,%i)\n",
         post->getLayerLoc()->nx,
         post->getLayerLoc()->ny,
         post->getLayerLoc()->nf);
   outStream->printf("\n");

   for (int arbor = 0; arbor < numberOfAxonalArborLists(); arbor++) {
      outStream->printf("displaying arbor %1.1d\n", arbor);
      // give a chance for derived classes to add extra information
      //
      writeTextWeightsExtra(outStream, k, arbor);
      pv_text_write_patch(outStream, wPatches[arbor][k], get_wData(arbor, k), nfp, sxp, syp, sfp);
      outStream->printf("----------------------------\n");
   }

   delete outStream;

   return 0;
}

int HyPerConn::readStateFromCheckpoint(Checkpointer *checkpointer) {
   if (initializeFromCheckpointFlag) {
      checkpointer->readNamedCheckpointEntry(std::string(name), std::string("W"), !plasticityFlag);
      if (plasticityFlag and !mImmediateWeightUpdate) {
         checkpointer->readNamedCheckpointEntry(
               std::string(name), std::string("dW"), false /*not constant*/);
      }
   }
   return PV_SUCCESS;
}

void HyPerConn::checkpointWeightPvp(
      Checkpointer *checkpointer,
      char const *bufferName,
      float **weightDataBuffer) {
   bool registerSucceeded = checkpointer->registerCheckpointEntry(
         std::make_shared<CheckpointEntryWeightPvp>(
               getName(),
               bufferName,
               checkpointer->getMPIBlock(),
               numberOfAxonalArborLists(),
               usingSharedWeights(),
               get_wPatches(),
               weightDataBuffer,
               mNumDataPatchesX,
               mNumDataPatchesY,
               mNumDataPatchesF,
               nxp,
               nyp,
               nfp,
               pre->getLayerLoc(),
               post->getLayerLoc(),
               writeCompressedWeights),
         !plasticityFlag);
   FatalIf(
         !registerSucceeded,
         "%s failed to register %s for checkpointing.\n",
         getDescription_c(),
         bufferName);
}

int HyPerConn::registerData(Checkpointer *checkpointer) {
   int status = BaseConnection::registerData(checkpointer);
   checkpointWeightPvp(checkpointer, "W", get_wDataStart());
   if (plasticityFlag and !mImmediateWeightUpdate) {
      checkpointWeightPvp(checkpointer, "dW", get_dwDataStart());
      // If we checkpoint dW, we have to get PrepareCheckpointRead messages,
      // in order to call blockingNormalize_dW() before the checkpoint.
   }
   if (writeStep >= 0) {
      std::string nameString = std::string(name);
      checkpointer->registerCheckpointData(
            nameString,
            "lastUpdateTime",
            &lastUpdateTime,
            (std::size_t)1,
            true /*broadcast*/,
            false /*not constant*/);
      if (plasticityFlag && !triggerLayerName) {
         checkpointer->registerCheckpointData(
               nameString,
               "weightUpdateTime",
               &weightUpdateTime,
               (std::size_t)1,
               true /*broadcast*/,
               false /*not constant*/);
      }
      checkpointer->registerCheckpointData(
            nameString,
            "nextWrite",
            &writeTime,
            (std::size_t)1,
            true /*broadcast*/,
            false /*not constant*/);

      openOutputStateFile(checkpointer);
   }
   registerTimers(checkpointer);

   return status;
}

void HyPerConn::openOutputStateFile(Checkpointer *checkpointer) {
   if (writeStep >= 0) {

      if (checkpointer->getMPIBlock()->getRank() == 0) {
         std::string outputStatePath(getName());
         outputStatePath.append(".pvp");

         std::string checkpointLabel(getName());
         checkpointLabel.append("_filepos");

         bool createFlag    = checkpointer->getCheckpointReadDirectory().empty();
         mOutputStateStream = new CheckpointableFileStream(
               outputStatePath.c_str(), createFlag, checkpointer, checkpointLabel);
      }
   }
}

void HyPerConn::registerTimers(Checkpointer *checkpointer) {
   io_timer = new Timer(getName(), "conn", "io     ");
   checkpointer->registerTimer(io_timer);

   update_timer = new Timer(getName(), "conn", "update ");
   checkpointer->registerTimer(update_timer);
}

float HyPerConn::minWeight(int arborId) {
   const int num_data_patches = getNumDataPatches();
   float min_weight           = FLT_MAX;
   if (sharedWeights) {
      const int numWeights = nxp * nyp * nfp;
      for (int iKernel = 0; iKernel < num_data_patches; iKernel++) {
         float *kernelWeights = get_wDataHead(arborId, iKernel);
         for (int iWeight = 0; iWeight < numWeights; iWeight++) {
            min_weight =
                  (min_weight < kernelWeights[iWeight]) ? min_weight : kernelWeights[iWeight];
         }
      }
   }
   else {
      for (int i_patch = 0; i_patch < num_data_patches; i_patch++) {
         float *w_data    = get_wData(arborId, i_patch);
         PVPatch *w_patch = getWeights(i_patch, arborId);
         int num_weights  = fPatchSize() * w_patch->nx * w_patch->ny;
         for (int iWeight = 0; iWeight < num_weights; iWeight++) {
            min_weight = (min_weight < w_data[iWeight]) ? min_weight : w_data[iWeight];
         }
      }
   }
   return min_weight;
}

float HyPerConn::maxWeight(int arborId) {
   const int num_data_patches = getNumDataPatches();
   float max_weight           = -FLT_MAX;
   if (sharedWeights) {
      const int numWeights = nxp * nyp * nfp;
      for (int iKernel = 0; iKernel < num_data_patches; iKernel++) {
         float *kernelWeights = get_wDataHead(arborId, iKernel);
         for (int iWeight = 0; iWeight < numWeights; iWeight++) {
            max_weight =
                  (max_weight > kernelWeights[iWeight]) ? max_weight : kernelWeights[iWeight];
         }
      }
   }
   else {
      for (int i_weight = 0; i_weight < num_data_patches; i_weight++) {
         float *w_data    = get_wData(arborId, i_weight);
         PVPatch *w_patch = getWeights(i_weight, arborId);
         int num_weights  = fPatchSize() * w_patch->nx * w_patch->ny;
         for (int iWeight = 0; iWeight < num_weights; iWeight++) {
            max_weight = (max_weight > w_data[iWeight]) ? max_weight : w_data[iWeight];
         }
      }
   }
   return max_weight;
}

int HyPerConn::insertProbe(BaseConnectionProbe *p) {
   if (p->getTargetConn() != this) {
      WarnLog().printf(
            "HyPerConn \"%s\": insertProbe called with probe %p, whose targetConn is not this "
            "connection.  Probe was not inserted.\n",
            name,
            p);
      return numProbes;
   }
   for (int i = 0; i < numProbes; i++) {
      if (p == probes[i]) {
         WarnLog().printf(
               "HyPerConn \"%s\": insertProbe called with probe %p, which has already been "
               "inserted as probe %d.\n",
               name,
               p,
               i);
         return numProbes;
      }
   }

   BaseConnectionProbe **tmp;
   // malloc'ing a new buffer, copying data over, and freeing the old buffer could be replaced by
   // malloc
   tmp = (BaseConnectionProbe **)pvMalloc((numProbes + 1) * sizeof(BaseConnectionProbe *));

   for (int i = 0; i < numProbes; i++) {
      tmp[i] = probes[i];
   }
   delete probes;

   probes            = tmp;
   probes[numProbes] = p;

   return ++numProbes;
}

int HyPerConn::setInitialValues() {
   initializeWeights(wPatches, wDataStart);
   int status = PV_SUCCESS;
   return status;
}

int HyPerConn::outputProbeParams() {
   int status = PV_SUCCESS;
   for (int p = 0; p < numProbes; p++) {
      int status1 = probes[p]->writeParams();
      if (status1 != PV_SUCCESS) {
         status = PV_FAILURE;
      }
   }
   return status;
}

int HyPerConn::outputState(double timef) {
   int status = 0;
   io_timer->start();

   for (int i = 0; i < numProbes; i++) {
      probes[i]->outputStateWrapper(timef, parent->getDeltaTime());
   }

   if ((writeStep >= 0) && (timef >= writeTime)) {
      writeTime += writeStep;

      status = writeWeights(timef);
      pvAssert(status == 0);
   }
   else if (writeStep < 0) { // If writeStep is negative, we never call writeWeights, but someone
      // might restart from a checkpoint with a different writeStep, so we
      // should still maintain writeTime
      writeTime = timef;
   }

   io_timer->stop();
   return status;
}

bool HyPerConn::needUpdate(double simTime, double dt) {
   if (!plasticityFlag) {
      return false;
   }
   if (triggerLayer) {
      return triggerLayer->needUpdate(simTime + triggerOffset, dt);
   }
   return simTime >= weightUpdateTime;
}

int HyPerConn::updateState(double simTime, double dt) {
   int status = PV_SUCCESS;
   if (needUpdate(simTime, dt)) {
      pvAssert(plasticityFlag);
      update_timer->start();
      if (mImmediateWeightUpdate) {
         updateWeightsImmediate(simTime, dt);
      }
      else {
         updateWeightsDelayed(simTime, dt);
      }

      decay_dWMax();

      lastUpdateTime = simTime;
      computeNewWeightUpdateTime(simTime, weightUpdateTime);
      needFinalize = true;
      update_timer->stop();
   }
   lastTimeUpdateCalled = simTime;
   return status;
}

void HyPerConn::updateWeightsImmediate(double simTime, double dt) {
   updateLocal_dW();
   reduce_dW();
   blockingNormalize_dW();
   updateArbors();
}

void HyPerConn::updateWeightsDelayed(double simTime, double dt) {
   blockingNormalize_dW();
   updateArbors();
   updateLocal_dW();
   reduce_dW();
}

void HyPerConn::updateLocal_dW() {
   pvAssert(plasticityFlag);
   int status;
   for (int arborId = 0; arborId < numberOfAxonalArborLists(); arborId++) {
      status = initialize_dW(arborId);
      if (status == PV_BREAK) {
         status = PV_SUCCESS;
         break;
      }
   }
   pvAssert(status == PV_SUCCESS);

   for (int arborId = 0; arborId < numberOfAxonalArborLists(); arborId++) {
      status = update_dW(arborId);
      if (status == PV_BREAK) {
         break;
      }
   }
   pvAssert(status == PV_SUCCESS or status == PV_BREAK);
}

void HyPerConn::reduce_dW() {
   int status;
   for (int arborId = 0; arborId < numberOfAxonalArborLists(); arborId++) {
      status = reduce_dW(arborId);
      if (status == PV_BREAK) {
         break;
      }
   }
   pvAssert(status == PV_SUCCESS or status == PV_BREAK);
   mReductionPending = true;
}

void HyPerConn::blockingNormalize_dW() {
   if (mReductionPending) {
      wait_dWReduceRequests();
      normalize_dW();
      mReductionPending = false;
   }
}

void HyPerConn::wait_dWReduceRequests() {
   MPI_Waitall(m_dWReduceRequests.size(), m_dWReduceRequests.data(), MPI_STATUSES_IGNORE);
   m_dWReduceRequests.clear();
}

void HyPerConn::normalize_dW() {
   int status = PV_SUCCESS;
   if (normalizeDwFlag) {
      for (int arborId = 0; arborId < numberOfAxonalArborLists(); arborId++) {
         status = normalize_dW(arborId);
         if (status == PV_BREAK) {
            break;
         }
      }
   }
   pvAssert(status == PV_SUCCESS or status == PV_BREAK);
}

void HyPerConn::updateArbors() {
   int status = PV_SUCCESS;
   for (int arborId = 0; arborId < numberOfAxonalArborLists(); arborId++) {
      status = updateWeights(arborId); // Apply changes in weights
      if (status == PV_BREAK) {
         status = PV_SUCCESS;
         break;
      }
   }
   pvAssert(status == PV_SUCCESS or status == PV_BREAK);
}

void HyPerConn::decay_dWMax() {
   if (mDWMaxDecayInterval > 0) {
      if (--mDWMaxDecayTimer < 0) {
         float oldDWMax   = dWMax;
         mDWMaxDecayTimer = mDWMaxDecayInterval;
         dWMax *= 1.0f - mDWMaxDecayFactor;
         InfoLog() << getName() << ": dWMax decayed from " << oldDWMax << " to " << dWMax << "\n";
      }
   }
}

int HyPerConn::clear_numActivations(int arborId) {
   // zero out all dW.
   // This also zeroes out the unused parts of shrunken patches
   for (int kArbor = 0; kArbor < numberOfAxonalArborLists(); kArbor++) {
      for (int kKernel = 0; kKernel < getNumDataPatches(); kKernel++) {
         int syPatch       = syp;
         int nkPatch       = nfp * nxp;
         long *activations = get_activationsHead(kArbor, kKernel);
         for (int kyPatch = 0; kyPatch < nyp; kyPatch++) {
            for (int kPatch = 0; kPatch < nkPatch; kPatch++) {
               activations[kPatch] = 0.0f;
            }
            activations += syPatch;
         }
      }
   }
   return PV_BREAK;
}

int HyPerConn::clear_dW(int arborId) {
   // zero out all dW.
   // This also zeroes out the unused parts of shrunken patches
   int const syPatch = syp;
   int const nkPatch = nfp * nxp;
   for (int kArbor = 0; kArbor < numberOfAxonalArborLists(); kArbor++) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (int kKernel = 0; kKernel < getNumDataPatches(); kKernel++) {
         float *dWeights = get_dwDataHead(kArbor, kKernel);
         for (int kyPatch = 0; kyPatch < nyp; kyPatch++) {
            for (int kPatch = 0; kPatch < nkPatch; kPatch++) {
               dWeights[kyPatch * syPatch + kPatch] = 0.0f;
            }
         }
      }
   }
   return PV_BREAK;
}

int HyPerConn::initialize_dW(int arborId) {
   if (!combine_dW_with_W_flag) {
      clear_dW(arborId);
   }
   if (numKernelActivations) {
      clear_numActivations(arborId);
   }
   // default initialize_dW returns PV_BREAK
   return PV_BREAK;
}

int HyPerConn::finalizeUpdate(double timed, double dt) {
   if (!needFinalize) {
      return PV_SUCCESS;
   }

#if defined(PV_USE_CUDA)
   if (allocDeviceWeights) {
      updateDeviceWeights();
   }
#endif

   // Update postConn if needed
   if (needPost && postConn) {
      int status = postConn->finalizeUpdate(timed, dt);
      pvAssert(status == PV_SUCCESS);
   }

   needFinalize = false;
   return PV_SUCCESS;
}

int HyPerConn::reduce_dW(int arborId) {
   int kernel_status = PV_BREAK;
   if (sharedWeights) {
      kernel_status = reduceKernels(arborId); // combine partial changes in each column
      if (normalizeDwFlag) {
         int activation_status = reduceActivations(arborId);
         pvAssert(kernel_status == activation_status);
      }
   }
   else {
      reduceAcrossBatch(arborId);
   }
   return kernel_status;
}

int HyPerConn::reduceActivations(int arborID) {
   pvAssert(sharedWeights && plasticityFlag);
   Communicator *comm = parent->getCommunicator();
   const int nxProcs  = comm->numCommColumns();
   const int nyProcs  = comm->numCommRows();
   const int nbProcs  = comm->numCommBatches();
   const int nProcs   = nxProcs * nyProcs * nbProcs;
   if (numKernelActivations && nProcs != 1) {
      const MPI_Comm mpi_comm = comm->globalCommunicator();
      const int numPatches    = getNumDataPatches();
      const size_t patchSize  = (size_t)nxp * (size_t)nyp * (size_t)nfp;
      const size_t localSize  = numPatches * patchSize;
      const size_t arborSize  = localSize * numberOfAxonalArborLists();

      auto sz = m_dWReduceRequests.size();
      m_dWReduceRequests.resize(sz + 1);
      MPI_Iallreduce(
            MPI_IN_PLACE,
            get_activations(arborID),
            arborSize,
            MPI_LONG,
            MPI_SUM,
            mpi_comm,
            &(m_dWReduceRequests.data())[sz]);
   }

   return PV_BREAK;
}

int HyPerConn::reduceKernels(int arborID) {
   pvAssert(sharedWeights && plasticityFlag);
   Communicator *comm = parent->getCommunicator();
   const int nxProcs  = comm->numCommColumns();
   const int nyProcs  = comm->numCommRows();
   const int nbProcs  = comm->numCommBatches();
   const int nProcs   = nxProcs * nyProcs * nbProcs;
   if (nProcs != 1) {
      const MPI_Comm mpi_comm = comm->globalCommunicator();
      const int numPatches    = getNumDataPatches();
      const size_t patchSize  = (size_t)nxp * (size_t)nyp * (size_t)nfp;
      const size_t localSize  = (size_t)numPatches * (size_t)patchSize;
      const size_t arborSize  = localSize * (size_t)numberOfAxonalArborLists();

      auto sz = m_dWReduceRequests.size();
      m_dWReduceRequests.resize(sz + 1);
      MPI_Iallreduce(
            MPI_IN_PLACE,
            get_dwDataStart(arborID),
            arborSize,
            MPI_FLOAT,
            MPI_SUM,
            mpi_comm,
            &(m_dWReduceRequests.data())[sz]);
   }

   return PV_BREAK;
}

void HyPerConn::reduceAcrossBatch(int arborID) {
   pvAssert(!sharedWeights && plasticityFlag);
   if (parent->getCommunicator()->numCommBatches() != 1) {
      float *dwArborStart      = get_dwDataStart(arborID);
      size_t const patchSize   = (size_t)nxp * (size_t)nyp * (size_t)nfp;
      size_t const localSize   = (size_t)getNumDataPatches() * (size_t)patchSize;
      size_t const arborSize   = localSize * (size_t)numberOfAxonalArborLists();
      MPI_Comm const batchComm = parent->getCommunicator()->batchCommunicator();

      auto sz = m_dWReduceRequests.size();
      m_dWReduceRequests.resize(sz + 1);
      MPI_Iallreduce(
            MPI_IN_PLACE,
            get_dwDataStart(arborID),
            arborSize,
            MPI_FLOAT,
            MPI_SUM,
            batchComm,
            &(m_dWReduceRequests.data())[sz]);
   }
}

int HyPerConn::update_dW(int arborID) {
   // compute dW but don't add them to the weights yet.
   // That takes place in reduceKernels, so that the output is
   // independent of the number of processors.
   int nExt              = preSynapticLayer()->getNumExtended();
   const PVLayerLoc *loc = preSynapticLayer()->getLayerLoc();
   int nbatch            = loc->nbatch;

   float const *preactbufHead  = preSynapticLayer()->getLayerData(getDelay(arborID));
   float const *postactbufHead = postSynapticLayer()->getLayerData();

   if (sharedWeights) {
      // Calculate x and y cell size
      int xCellSize  = zUnitCellSize(pre->getXScale(), post->getXScale());
      int yCellSize  = zUnitCellSize(pre->getYScale(), post->getYScale());
      int nxExt      = loc->nx + loc->halo.lt + loc->halo.rt;
      int nyExt      = loc->ny + loc->halo.up + loc->halo.dn;
      int nf         = loc->nf;
      int numKernels = getNumDataPatches();

      for (int b = 0; b < nbatch; b++) {
         if (batchSkip[b])
            continue;
// Shared weights done in parallel, parallel in numkernels
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int kernelIdx = 0; kernelIdx < numKernels; kernelIdx++) {

            // Calculate xCellIdx, yCellIdx, and fCellIdx from kernelIndex
            int kxCellIdx = kxPos(kernelIdx, xCellSize, yCellSize, nf);
            int kyCellIdx = kyPos(kernelIdx, xCellSize, yCellSize, nf);
            int kfIdx     = featureIndex(kernelIdx, xCellSize, yCellSize, nf);
            // Loop over all cells in pre ext
            int kyIdx    = kyCellIdx;
            int yCellIdx = 0;
            while (kyIdx < nyExt) {
               int kxIdx    = kxCellIdx;
               int xCellIdx = 0;
               while (kxIdx < nxExt) {
                  // Calculate kExt from ky, kx, and kf
                  int kExt = kIndex(kxIdx, kyIdx, kfIdx, nxExt, nyExt, nf);
                  updateInd_dW(arborID, b, preactbufHead, postactbufHead, kExt);
                  xCellIdx++;
                  kxIdx = kxCellIdx + xCellIdx * xCellSize;
               }
               yCellIdx++;
               kyIdx = kyCellIdx + yCellIdx * yCellSize;
            }
         }
      }
   }
   else {
      for (int b = 0; b < nbatch; b++) {
// Shared weights done in parallel, parallel in numkernels
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int kExt = 0; kExt < nExt; kExt++) {
            updateInd_dW(arborID, b, preactbufHead, postactbufHead, kExt);
         }
      }
   }

   // If update from clones, update dw here as well
   // Updates on all PlasticClones
   for (int clonei = 0; clonei < clones.size(); clonei++) {
      pvAssert(clones[clonei]->preSynapticLayer()->getNumExtended() == nExt);
      float const *clonePre  = clones[clonei]->preSynapticLayer()->getLayerData(getDelay(arborID));
      float const *clonePost = clones[clonei]->postSynapticLayer()->getLayerData();
      for (int b = 0; b < parent->getNBatch(); b++) {
         for (int kExt = 0; kExt < nExt; kExt++) {
            this->updateInd_dW(arborID, b, clonePre, clonePost, kExt);
         }
      }
   }

   return PV_SUCCESS;
}

int HyPerConn::updateInd_dW(
      int arborID,
      int batchID,
      float const *preLayerData,
      float const *postLayerData,
      int kExt) {
   const PVLayerLoc *postLoc = post->getLayerLoc();

   const float *maskactbuf = NULL;
   if (useMask) {
      const float *maskactbufHead = mask->getLayerData();
      maskactbuf                  = maskactbufHead + batchID * mask->getNumExtended();
   }
   const float *preactbuf  = preLayerData + batchID * preSynapticLayer()->getNumExtended();
   const float *postactbuf = postLayerData + batchID * postSynapticLayer()->getNumExtended();

   int sya =
         (post->getLayerLoc()->nf * (post->getLayerLoc()->nx + post->getLayerLoc()->halo.lt
                                     + post->getLayerLoc()->halo.rt));

   float preact = preactbuf[kExt];
   if (skipPre(preact))
      return PV_CONTINUE;

   PVPatch *weights = getWeights(kExt, arborID);
   int ny           = weights->ny;
   int nk           = weights->nx * nfp;
   if (ny == 0 || nk == 0) {
      return PV_SUCCESS;
   }

   size_t offset           = getAPostOffset(kExt, arborID);
   const float *postactRef = &postactbuf[offset];

   int sym                 = 0;
   const float *maskactRef = NULL;
   if (useMask) {
      const PVLayerLoc *maskLoc = mask->getLayerLoc();
      // Calculate mask offset, must account for different size margins and the num features
      // offsetX and Y are restricted indices into post
      size_t offsetX, offsetY;
      offsetX = kxPos(offset,
                      postLoc->nx + postLoc->halo.lt + postLoc->halo.rt,
                      postLoc->ny + postLoc->halo.up + postLoc->halo.dn,
                      postLoc->nf)
                - postLoc->halo.lt;
      offsetY = kyPos(offset,
                      postLoc->nx + postLoc->halo.lt + postLoc->halo.rt,
                      postLoc->ny + postLoc->halo.up + postLoc->halo.dn,
                      postLoc->nf)
                - postLoc->halo.up;
      // Sanity check, offset should be in restricted
      pvAssert(offsetX < postLoc->nx + postLoc->halo.lt);
      pvAssert(offsetY < postLoc->ny + postLoc->halo.up);
      // Convert to maskOffsetX and Y, extended (in mask)
      size_t maskOffsetX, maskOffsetY;
      maskOffsetX = offsetX + maskLoc->halo.lt;
      maskOffsetY = offsetY + maskLoc->halo.up;
      // Convert to extIndex into mask

      size_t maskOffset = kIndex(
            maskOffsetX,
            maskOffsetY,
            0,
            maskLoc->nx + maskLoc->halo.lt + maskLoc->halo.rt,
            maskLoc->ny + maskLoc->halo.up + maskLoc->halo.dn,
            maskLoc->nf); // This should take into account if maskLoc's nf is either 1 or the size
      // of post

      maskactRef = &maskactbuf[maskOffset];
      sym        = (maskLoc->nf * (maskLoc->nx + maskLoc->halo.lt + maskLoc->halo.rt));
   }

   float *dwdata     = get_dwData(arborID, kExt);
   long *activations = NULL;
   if (sharedWeights && normalizeDwFlag) {
      activations = get_activations(arborID, kExt);
   }

   int lineoffsetw = 0;
   int lineoffseta = 0;
   int lineoffsetm = 0;
   for (int y = 0; y < ny; y++) {
      for (int k = 0; k < nk; k++) {
         float aPost = postactRef[lineoffseta + k];
         // calculate contribution to dw unless masked out
         pvAssert(
               !useMask || maskactRef != NULL); // if useMask is true, maskactRef must not be null
         float maskVal = 1;
         if (useMask) {
            if (mask->getLayerLoc()->nf == 1) {
               maskVal = maskactRef[lineoffsetm + ((int)k / postLoc->nf)];
            }
            else {
               // If a maskFeatureIdx was specified
               if (maskFeatureIdx >= 0) {
                  // k is an index into x/f space. Convert back to x space, and find the 0 feature
                  // index
                  int startingMaskK = ((int)k / postLoc->nf) * postLoc->nf;
                  // Offset into maskFeatureIdx
                  maskVal = maskactRef[lineoffsetm + startingMaskK + maskFeatureIdx];
               }
               else {
                  maskVal = maskactRef[lineoffsetm + k];
               }
            }
         }
         if (maskVal != 0) {
            // Note: this is a hack, as batching calls this function, but overwrites to allocate
            // numKernelActivations with non-shared weights
            if (activations) {
               // Offset in the case of a shrunken patch, where dwdata is applying when calling
               // get_dwData
               activations[lineoffsetw + k]++;
            }
            dwdata[lineoffsetw + k] += updateRule_dW(preact, aPost);
         }
      }
      lineoffsetw += syp;
      lineoffseta += sya;
      lineoffsetm += sym;
   }
   return PV_SUCCESS;
}

void HyPerConn::addClone(PlasticCloneConn *conn) {
   // Make sure that the origional conn is indeed this
   pvAssert(conn->getOriginalConn() == this);

   // CloneConn's communicateInitInfo makes sure the pre layers' borders are in sync,
   // but for PlasticCloneConns to apply the update rules correctly, we need the
   // post layers' borders to be equal as well.

   conn->postSynapticLayer()->synchronizeMarginWidth(this->postSynapticLayer());
   this->postSynapticLayer()->synchronizeMarginWidth(conn->postSynapticLayer());

   // Add the new PlasticCloneConn to the list of clones.
   clones.push_back(conn);
}

int HyPerConn::normalize_dW(int arbor_ID) {
   // This is here in case other classes overwrite the outer class calling this function
   if (!normalizeDwFlag) {
      return PV_SUCCESS;
   }
   if (sharedWeights) {
      pvAssert(numKernelActivations);
      int numKernelIndices = getNumDataPatches();
      for (int loop_arbor = 0; loop_arbor < numberOfAxonalArborLists(); loop_arbor++) {
// Divide by numKernelActivations in this timestep
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int kernelindex = 0; kernelindex < numKernelIndices; kernelindex++) {
            // Calculate pre feature index from patch index
            int numpatchitems  = nxp * nyp * nfp;
            float *dwpatchdata = get_dwDataHead(loop_arbor, kernelindex);
            long *activations  = get_activationsHead(loop_arbor, kernelindex);
            for (int n = 0; n < numpatchitems; n++) {
               long divisor = activations[n];

               if (divisor != 0) {
                  dwpatchdata[n] /= divisor;
               }
               else {
                  dwpatchdata[n] = 0;
               }
            }
         }
      }
   }
   // TODO: non-shared weights should divide by batch period if applicable
   return PV_BREAK;
}

float HyPerConn::updateRule_dW(float pre, float post) { return dWMax * pre * post; }

int HyPerConn::updateWeights(int arborId) {
   // add dw to w
   for (int kArbor = 0; kArbor < numberOfAxonalArborLists(); kArbor++) {
      float *w_data_start = get_wDataStart(kArbor);
      for (long int k = 0; k < patchStartIndex(getNumDataPatches()); k++) {
         w_data_start[k] += get_dwDataStart(kArbor)[k];
      }
   }
   return PV_BREAK;
}

double HyPerConn::computeNewWeightUpdateTime(double simTime, double currentUpdateTime) {
   // Only called if plasticity flag is set
   if (!triggerLayer) {
      while (simTime >= weightUpdateTime) {
         weightUpdateTime += weightUpdatePeriod;
      }
   }
   return weightUpdateTime;
}

PVPatch *HyPerConn::getWeights(int k, int arbor) {
   // a separate arbor/patch of weights for every neuron
   return wPatches[arbor][k];
}

int HyPerConn::deliver() {
   int status = PV_SUCCESS;

   // Check if updating from post perspective
   HyPerLayer *pre = preSynapticLayer();
   int numArbors   = numberOfAxonalArborLists();

   for (int arbor = 0; arbor < numArbors; arbor++) {
      int delay        = getDelay(arbor);
      PVLayerCube cube = pre->getPublisher()->createCube(delay);
      cube.numItems /= cube.loc.nbatch;
      // hack; should make sure deliver*Perspective* methods expect numItems to include batching.
      if (!getUpdateGSynFromPostPerspective()) {
#ifdef PV_USE_CUDA
         if (getReceiveGpu()) {
            status = deliverPresynapticPerspectiveGPU(&cube, arbor);
            // No need to update GSyn since it's already living on gpu
            post->setUpdatedDeviceGSynFlag(false);
         }
         else
#endif
         {
            status = deliverPresynapticPerspective(&cube, arbor);
#ifdef PV_USE_CUDA
            // CPU updated gsyn, need to update gsyn
            post->setUpdatedDeviceGSynFlag(true);
#endif
         }
      }
      else {
#ifdef PV_USE_CUDA
         if (getReceiveGpu()) {
            status = deliverPostsynapticPerspectiveGPU(&cube, arbor);
            // GSyn already living on GPU
            post->setUpdatedDeviceGSynFlag(false);
         }
         else
#endif
         {
            status = deliverPostsynapticPerspective(&cube, arbor);
#ifdef PV_USE_CUDA
            // CPU updated gsyn, need to update on GPU
            post->setUpdatedDeviceGSynFlag(true);
#endif
         }
      }
      pvAssert(status == PV_SUCCESS || status == PV_BREAK);
      if (status == PV_BREAK) {
         break; // Breaks out of arbor loop
      }
   }
   return PV_SUCCESS;
}

int HyPerConn::deliverPresynapticPerspectiveConvolve(PVLayerCube const *activity, int arbor) {
   // Check if we need to update based on connection's channel
   if (getChannel() == CHANNEL_NOUPDATE) {
      return PV_SUCCESS;
   }
   pvAssert(post->getChannel(getChannel()));

   float dtFactor = getConvertToRateDeltaTimeFactor();
   if (getPvpatchAccumulateType() == STOCHASTIC) {
      dtFactor = parent->getDeltaTime();
   }

   if (mWeightSparsity > 0.0f && !mSparseWeightsAllocated[arbor]) {
      allocateSparseWeightsPre(activity, arbor);
   }

   const PVLayerLoc *preLoc  = preSynapticLayer()->getLayerLoc();
   const PVLayerLoc *postLoc = postSynapticLayer()->getLayerLoc();

   pvAssert(arbor >= 0);
   const int numExtended = activity->numItems;

   int nbatch    = parent->getNBatch();
   const int sy  = getPostNonextStrides()->sy; // stride in layer
   const int syw = yPatchStride(); // stride in patch

   for (int b = 0; b < nbatch; b++) {
      size_t batchOffset = b * (preLoc->nx + preLoc->halo.rt + preLoc->halo.lt)
                           * (preLoc->ny + preLoc->halo.up + preLoc->halo.dn) * preLoc->nf;
      float *activityBatch = activity->data + batchOffset;
      float *gSynPatchHeadBatch =
            post->getChannel(getChannel()) + b * postLoc->nx * postLoc->ny * postLoc->nf;
      SparseList<float>::Entry const *activeIndicesBatch = NULL;
      if (activity->isSparse) {
         activeIndicesBatch = (SparseList<float>::Entry *)activity->activeIndices + batchOffset;
      }

      int numNeurons = activity->isSparse ? activity->numActive[b] : numExtended;

#ifdef PV_USE_OPENMP_THREADS
      // Clear all thread gsyn buffer
      if (thread_gSyn) {
         int numNeurons = post->getNumNeurons();
#pragma omp parallel for schedule(static)
         for (int ti = 0; ti < parent->getNumThreads(); ++ti) {
            for (int ni = 0; ni < numNeurons; ++ni) {
               thread_gSyn[ti][ni] = 0.0;
            }
         }
      }
#endif

      if (!activity->isSparse) {
         for (int y = 0; y < nyp; y++) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(guided)
#endif
            for (int idx = 0; idx < numNeurons; idx++) {
               int kPreExt = idx;

               // Weight
               PVPatch *weights = getWeights(kPreExt, arbor);

               if (y >= weights->ny) {
                  continue;
               }

               // Activity
               float a = activityBatch[kPreExt] * dtFactor;
               if (a == 0.0f) {
                  continue;
               }

               // gSyn
               float *gSynPatchHead = gSynPatchHeadBatch;

#ifdef PV_USE_OPENMP_THREADS
               if (thread_gSyn) {
                  gSynPatchHead = thread_gSyn[omp_get_thread_num()];
               }
#endif // PV_USE_OPENMP_THREADS

               float *postPatchStart = gSynPatchHead + getGSynPatchStart(kPreExt, arbor);

               const int nk           = weights->nx * fPatchSize();
               float *weightDataStart = get_wData(arbor, kPreExt); // make this a float const *?

               float *v = postPatchStart + y * sy;
               float *w = weightDataStart + y * syw;
               for (int k = 0; k < nk; k++) {
                  v[k] += a * w[k];
               }
            }
         }
      }
      else { // Sparse, use the stored activity / index pairs
         for (int y = 0; y < nyp; y++) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(guided)
#endif
            for (int idx = 0; idx < numNeurons; idx++) {
               int kPreExt = activeIndicesBatch[idx].index;

               // Weight
               PVPatch *weights = getWeights(kPreExt, arbor);

               if (y >= weights->ny) {
                  continue;
               }

               // Activity
               float a = activeIndicesBatch[idx].value;
               if (a == 0.0f) {
                  continue;
               }

               // gSyn
               float *gSynPatchHead = gSynPatchHeadBatch;

#ifdef PV_USE_OPENMP_THREADS
               if (thread_gSyn) {
                  gSynPatchHead = thread_gSyn[omp_get_thread_num()];
               }
#endif // PV_USE_OPENMP_THREADS

               float *postPatchStart = gSynPatchHead + getGSynPatchStart(kPreExt, arbor);

               const int nk           = weights->nx * fPatchSize();
               float *weightDataStart = get_wData(arbor, kPreExt); // make this a float const *?
               a *= dtFactor;
               float *v = postPatchStart + y * sy;
               float *w = weightDataStart + y * syw;
               for (int k = 0; k < nk; k++) {
                  v[k] += a * w[k];
               }
            }
         }
      }
#ifdef PV_USE_OPENMP_THREADS
      // Accumulate back into gSyn // Should this be done in HyPerLayer where it can be done once,
      // as opposed to once per connection?
      if (thread_gSyn) {
         float *gSynPatchHead = gSynPatchHeadBatch;
         int numNeurons       = post->getNumNeurons();
         for (int ti = 0; ti < parent->getNumThreads(); ti++) {
            float *onethread = thread_gSyn[ti];
// Looping over neurons is thread safe
#pragma omp parallel for
            for (int ni = 0; ni < numNeurons; ni++) {
               gSynPatchHead[ni] += onethread[ni];
            }
         }
      }
#endif // PV_USE_OPENMP_THREADS
   }
   return PV_SUCCESS;
}

// TODO: Use templating to replace deliverPresynapticPerspectiveStochastic and
// delivePresynapticPerspective.  These two functions differ only in the innermost
// loop and some variable initializations the innermost loop uses.  We want to avoid
// the inefficiency of an if-statement or dereferencing a function pointer in the
// innermost loop.
int HyPerConn::deliverPresynapticPerspectiveStochastic(PVLayerCube const *activity, int arbor) {
   // Check if we need to update based on connection's channel
   if (getChannel() == CHANNEL_NOUPDATE) {
      return PV_SUCCESS;
   }
   pvAssert(post->getChannel(getChannel()));

   float dtFactor = getConvertToRateDeltaTimeFactor();
   if (getPvpatchAccumulateType() == STOCHASTIC) {
      dtFactor = parent->getDeltaTime();
   }

   if (mWeightSparsity > 0.0f && !mSparseWeightsAllocated[arbor]) {
      allocateSparseWeightsPre(activity, arbor);
   }

   const PVLayerLoc *preLoc  = preSynapticLayer()->getLayerLoc();
   const PVLayerLoc *postLoc = postSynapticLayer()->getLayerLoc();

   pvAssert(arbor >= 0);
   const int numExtended = activity->numItems;

   int nbatch    = parent->getNBatch();
   const int sy  = getPostNonextStrides()->sy; // stride in layer
   const int syw = yPatchStride(); // stride in patch

   for (int b = 0; b < nbatch; b++) {
      size_t batchOffset = b * (preLoc->nx + preLoc->halo.rt + preLoc->halo.lt)
                           * (preLoc->ny + preLoc->halo.up + preLoc->halo.dn) * preLoc->nf;
      float *activityBatch = activity->data + batchOffset;
      float *gSynPatchHeadBatch =
            post->getChannel(getChannel()) + b * postLoc->nx * postLoc->ny * postLoc->nf;
      SparseList<taus_uint4>::Entry const *activeIndicesBatch = NULL;
      if (activity->isSparse) {
         activeIndicesBatch =
               (SparseList<taus_uint4>::Entry *)activity->activeIndices + batchOffset;
      }

      int numNeurons = activity->isSparse ? activity->numActive[b] : numExtended;

#ifdef PV_USE_OPENMP_THREADS
      // Clear all thread gsyn buffer
      if (thread_gSyn) {
         int numNeurons = post->getNumNeurons();
#pragma omp parallel for
         for (int i = 0; i < parent->getNumThreads() * numNeurons; i++) {
            int ti              = i / numNeurons;
            int ni              = i % numNeurons;
            thread_gSyn[ti][ni] = 0;
         }
      }

#pragma omp parallel for schedule(guided)
#endif
      for (int idx = 0; idx < numNeurons; idx++) {
         int kPreExt = activity->isSparse ? activeIndicesBatch[idx].index : idx;

         // Activity
         float a = activityBatch[kPreExt] * dtFactor;
         if (a == 0.0f)
            continue;

         // gSyn
         float *gSynPatchHead = gSynPatchHeadBatch;

#ifdef PV_USE_OPENMP_THREADS
         if (thread_gSyn) {
            gSynPatchHead = thread_gSyn[omp_get_thread_num()];
         }
#endif // PV_USE_OPENMP_THREADS

         float *postPatchStart = gSynPatchHead + getGSynPatchStart(kPreExt, arbor);

         // Weight
         PVPatch *weights       = getWeights(kPreExt, arbor);
         const int nk           = weights->nx * fPatchSize();
         const int ny           = weights->ny;
         float *weightDataStart = get_wData(arbor, kPreExt); // make this a float const *?
         taus_uint4 *rng        = randState->getRNG(kPreExt);
         long along             = (long)((double)a * cl_random_max());

         for (int y = 0; y < ny; y++) {
            float *v = postPatchStart + y * sy;
            float *w = weightDataStart + y * syw;
            for (int k = 0; k < nk; k++) {
               *rng = cl_random_get(*rng);
               v[k] += (rng->s0 < along) * w[k];
            }
         }
      }

#ifdef PV_USE_OPENMP_THREADS
      // Accumulate back into gSyn // Should this be done in HyPerLayer where it can be done once,
      // as opposed to once per connection?
      if (thread_gSyn) {
         float *gSynPatchHead = gSynPatchHeadBatch;
         int numNeurons       = post->getNumNeurons();
// Looping over neurons first to be thread safe
#pragma omp parallel for
         for (int ni = 0; ni < numNeurons; ni++) {
            for (int ti = 0; ti < parent->getNumThreads(); ti++) {
               gSynPatchHead[ni] += thread_gSyn[ti][ni];
            }
         }
      }
#endif // PV_USE_OPENMP_THREADS
   }
   return PV_SUCCESS;
}

int HyPerConn::deliverPostsynapticPerspectiveConvolve(
      PVLayerCube const *activity,
      int arbor,
      int *numActive,
      int **activeList) {
   // Make sure numActive and activeList are either both null or both valid pointers
   if (numActive) {
      assert(activeList);
   }
   else {
      assert(!activeList);
   }

   // Check channel number for noupdate
   if (getChannel() == CHANNEL_NOUPDATE) {
      return PV_SUCCESS;
   }
   pvAssert(post->getChannel(getChannel()));

   pvAssert(arbor >= 0);
   // Get number of neurons restricted target
   const int numPostRestricted = post->getNumNeurons();

   float dtFactor = getConvertToRateDeltaTimeFactor();
   if (getPvpatchAccumulateType() == STOCHASTIC) {
      dtFactor = parent->getDeltaTime();
   }

   if (mWeightSparsity > 0.0f && !mSparseWeightsAllocated[arbor]) {
      allocateSparseWeightsPost(activity, arbor);
   }

   const PVLayerLoc *sourceLoc = preSynapticLayer()->getLayerLoc();
   const PVLayerLoc *targetLoc = post->getLayerLoc();

   const int sourceNx = sourceLoc->nx;
   const int sourceNy = sourceLoc->ny;
   const int sourceNf = sourceLoc->nf;
   const int targetNx = targetLoc->nx;
   const int targetNy = targetLoc->ny;
   const int targetNf = targetLoc->nf;
   const int nbatch   = targetLoc->nbatch;

   const PVHalo *sourceHalo = &sourceLoc->halo;
   const PVHalo *targetHalo = &targetLoc->halo;

   // get source layer's extended y stride
   int sy = (sourceNx + sourceHalo->lt + sourceHalo->rt) * sourceNf;

   // The start of the gsyn buffer
   float *gSynPatchHead = post->getChannel(getChannel());

   long *startSourceExtBuf = getPostToPreActivity();
   if (!startSourceExtBuf) {
      Fatal() << "HyPerLayer::recvFromPost: unable to get preToPostActivity from connection. Is "
                 "shrink_patches on?\n";
   }

   // If numActive is a valid pointer, we're recv from post sparse
   bool recvPostSparse = numActive;

   // Get source layer's patch y stride
   int syp               = postConn->yPatchStride();
   int yPatchSize        = postConn->yPatchSize();
   int numPerStride      = postConn->xPatchSize() * postConn->fPatchSize();
   int neuronIndexStride = nfp < 4 ? 1 : nfp / 4;

   if (sharedWeights) {
      // The differences between the sharedWeights and non-sharedWeights parts of the code are:
      // sharedWeights splits the loop over neurons using neuronIndexStride
      // sharedWeights calls patchToDataLUT, while non-sharedWeights just uses the neuron index.
      // If you change one side or the other of this if-statement, please evaluate whether
      // the same change makes sense in the other part, or document the difference between them.
      for (int b = 0; b < nbatch; b++) {
         int numNeurons  = recvPostSparse ? numActive[b] : numPostRestricted;
         int sourceNxExt = sourceNx + sourceHalo->rt + sourceHalo->lt;
         int sourceNyExt = sourceNy + sourceHalo->dn + sourceHalo->up;

         float *activityBatch      = activity->data + b * sourceNxExt * sourceNyExt * sourceNf;
         float *gSynPatchHeadBatch = gSynPatchHead + b * targetNx * targetNy * targetNf;

         // Iterate over each line in the y axis, the goal is to keep weights in the cache
         for (int ky = 0; ky < yPatchSize; ky++) {
// Threading over feature was the important change that improved cache performance by
// 5-10x. dynamic scheduling also gave another performance increase over static.
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
            for (int feature = 0; feature < neuronIndexStride; feature++) {
               for (int idx = feature; idx < numNeurons; idx += neuronIndexStride) {
                  int kTargetRes = recvPostSparse ? activeList[b][idx] : idx;
                  // gSyn
                  float *gSyn = gSynPatchHeadBatch + kTargetRes;

                  // Activity
                  float *a = activityBatch + startSourceExtBuf[kTargetRes] + ky * sy;

                  // Weight
                  int kTargetExt = kIndexExtended(
                        kTargetRes,
                        targetNx,
                        targetNy,
                        targetNf,
                        targetHalo->lt,
                        targetHalo->rt,
                        targetHalo->dn,
                        targetHalo->up);
                  int kernelIndex  = postConn->patchToDataLUT(kTargetExt);
                  float *weightBuf = postConn->get_wDataHead(arbor, kernelIndex);
                  float *weights   = weightBuf + ky * syp;

                  float dv = 0.0f;
                  for (int k = 0; k < numPerStride; ++k) {
                     dv += a[k] * weights[k];
                  }
                  *gSyn += dtFactor * dv;
               }
            }
         }
      }
   }
   else {
      // The differences between the sharedWeights and non-sharedWeights parts of the code are:
      // sharedWeights splits the loop over neurons using neuronIndexStride
      // sharedWeights calls patchToDataLUT, while non-sharedWeights just uses the neuron index.
      // If you change one side or the other of this if-statement, please evaluate whether
      // the same change makes sense in the other part, or document the difference between them.
      for (int b = 0; b < nbatch; b++) {
         int numNeurons  = recvPostSparse ? numActive[b] : numPostRestricted;
         int sourceNxExt = sourceNx + sourceHalo->rt + sourceHalo->lt;
         int sourceNyExt = sourceNy + sourceHalo->dn + sourceHalo->up;

         float *activityBatch      = activity->data + b * sourceNxExt * sourceNyExt * sourceNf;
         float *gSynPatchHeadBatch = gSynPatchHead + b * targetNx * targetNy * targetNf;

         // Iterate over each line in the y axis, the goal is to keep weights in the cache
         for (int ky = 0; ky < yPatchSize; ky++) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
            for (int idx = 0; idx < numNeurons; idx++) {
               int kTargetRes = recvPostSparse ? activeList[b][idx] : idx;
               // gSyn
               float *gSyn = gSynPatchHeadBatch + kTargetRes;

               // Activity
               float *a = activityBatch + startSourceExtBuf[kTargetRes] + ky * sy;

               // Weight
               int kTargetExt = kIndexExtended(
                     kTargetRes,
                     targetNx,
                     targetNy,
                     targetNf,
                     targetHalo->lt,
                     targetHalo->rt,
                     targetHalo->dn,
                     targetHalo->up);
               int kernelIndex  = kTargetExt;
               float *weightBuf = postConn->get_wDataHead(arbor, kernelIndex);
               float *weights   = weightBuf + ky * syp;

               float dv = 0.0f;
               for (int k = 0; k < numPerStride; ++k) {
                  dv += a[k] * weights[k];
               }
               *gSyn += dtFactor * dv;
            }
         }
      }
   }
   return PV_SUCCESS;
}

// TODO: Use templating to replace deliverPostsynapticPerspectiveStochastic and
// delivePostsynapticPerspective.  These two functions differ only in the innermost
// loop and some variable initializations the innermost loop uses.  We want to avoid
// the inefficiency of an if-statement or dereferencing a function pointer in the
// innermost loop.
int HyPerConn::deliverPostsynapticPerspectiveStochastic(
      PVLayerCube const *activity,
      int arbor,
      int *numActive,
      int **activeList) {
   // Make sure numActive and activeList are either both null or both valid pointers
   if (numActive) {
      assert(activeList);
   }
   else {
      assert(!activeList);
   }

   // Check channel number for noupdate
   if (getChannel() == CHANNEL_NOUPDATE) {
      return PV_SUCCESS;
   }
   pvAssert(post->getChannel(getChannel()));

   pvAssert(arbor >= 0);
   // Get number of neurons restricted target
   const int numPostRestricted = post->getNumNeurons();

   double dtFactor = getConvertToRateDeltaTimeFactor();
   if (getPvpatchAccumulateType() == STOCHASTIC) {
      dtFactor = parent->getDeltaTime();
   }

   if (mWeightSparsity > 0.0f && !mSparseWeightsAllocated[arbor]) {
      allocateSparseWeightsPost(activity, arbor);
   }

   const PVLayerLoc *sourceLoc = preSynapticLayer()->getLayerLoc();
   const PVLayerLoc *targetLoc = post->getLayerLoc();

   const int sourceNx = sourceLoc->nx;
   const int sourceNy = sourceLoc->ny;
   const int sourceNf = sourceLoc->nf;
   const int targetNx = targetLoc->nx;
   const int targetNy = targetLoc->ny;
   const int targetNf = targetLoc->nf;
   const int nbatch   = targetLoc->nbatch;

   const PVHalo *sourceHalo = &sourceLoc->halo;
   const PVHalo *targetHalo = &targetLoc->halo;

   // get source layer's extended y stride
   int sy = (sourceNx + sourceHalo->lt + sourceHalo->rt) * sourceNf;

   // The start of the gsyn buffer
   float *gSynPatchHead = post->getChannel(getChannel());

   long *startSourceExtBuf = getPostToPreActivity();
   if (!startSourceExtBuf) {
      Fatal() << "HyPerLayer::recvFromPost: unable to get preToPostActivity from connection. Is "
                 "shrink_patches on?\n";
   }

   // If numActive is a valid pointer, we're recv from post sparse
   bool recvPostSparse = numActive;

   // Get source layer's patch y stride
   int syp          = postConn->yPatchStride();
   int yPatchSize   = postConn->yPatchSize();
   int numPerStride = postConn->xPatchSize() * postConn->fPatchSize();

   for (int b = 0; b < nbatch; b++) {
      int numNeurons = recvPostSparse ? numActive[b] : numPostRestricted;

      float *activityBatch = activity->data
                             + b * (sourceNx + sourceHalo->rt + sourceHalo->lt)
                                     * (sourceNy + sourceHalo->up + sourceHalo->dn) * sourceNf;
      float *gSynPatchHeadBatch = gSynPatchHead + b * targetNx * targetNy * targetNf;

      // Iterate over each line in the y axis, the goal is to keep weights in the cache
      for (int ky = 0; ky < yPatchSize; ky++) {
// Threading over feature was the important change that improved cache performance by
// 5-10x. dynamic scheduling also gave another performance increase over static.
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
         for (int feature = 0; feature < nfp; feature++) {
            for (int idx = feature; idx < numNeurons; idx += nfp) {
               int kTargetRes = recvPostSparse ? activeList[b][idx] : idx;

               // gSyn
               float *gSyn = gSynPatchHeadBatch + kTargetRes;

               // Activity
               long startSourceExt     = startSourceExtBuf[kTargetRes];
               float *activityStartBuf = activityBatch + startSourceExt;
               float *a                = activityStartBuf + ky * sy;
               taus_uint4 *rng         = randState->getRNG(kTargetRes);

               // Weight
               int kTargetExt = kIndexExtended(
                     kTargetRes,
                     targetNx,
                     targetNy,
                     targetNf,
                     targetHalo->lt,
                     targetHalo->rt,
                     targetHalo->dn,
                     targetHalo->up);
               int kernelIndex       = postConn->patchToDataLUT(kTargetExt);
               float *weightStartBuf = postConn->get_wDataHead(arbor, kernelIndex);
               float *w              = weightStartBuf + ky * syp;
               float dv              = 0.0f;
               for (int k = 0; k < numPerStride; k++) {
                  *rng     = cl_random_get(*rng);
                  double p = (double)rng->s0 / cl_random_max(); // 0.0 < p < 1.0
                  dv += (p < (double)a[k] * dtFactor) * w[k];
               }
               *gSyn += dv;
            }
         }
      }
   }

   return PV_SUCCESS;
}

#ifdef PV_USE_CUDA
void HyPerConn::updateDeviceWeights() {
   // wDataStart is one big buffer, so this should grab everything
   float *h_weights              = get_wDataStart(0);
   PVCuda::CudaBuffer *d_weights = getDeviceWData();
   pvAssert(d_weights);
   d_weights->copyToDevice(h_weights);

   // Need barrier here?
   parent->getDevice()->syncDevice();

#ifdef PV_USE_CUDNN
   // Set local sizes here
   const PVLayerLoc *preLoc  = pre->getLayerLoc();
   const PVLayerLoc *postLoc = post->getLayerLoc();

   pvAssert(cudnn_WData);
   cudnn_WData->permuteWeightsPVToCudnn(
         d_weights->getPointer(), numberOfAxonalArborLists(), getNumDataPatches(), nxp, nyp, nfp);
#endif
}

int HyPerConn::deliverPresynapticPerspectiveGPU(PVLayerCube const *activity, int arborID) {
   pvAssert(krRecvPre);
   // Check if we need to update based on connection's channel
   if (getChannel() == CHANNEL_NOUPDATE) {
      return PV_SUCCESS;
   }
   pvAssert(post->getChannel(getChannel()));

   float dtFactor;
   if (getPvpatchAccumulateType() == STOCHASTIC) {
      dtFactor = parent->getDeltaTime();
   }
   else if (getPvpatchAccumulateType() == CONVOLVE) {
      dtFactor = getConvertToRateDeltaTimeFactor();
   }
   else {
      Fatal() << "Pooling accumulate not implemented for GPUs";
   }

   krRecvPre->set_dt_factor(dtFactor);

   // Post layer receives synaptic input
   // Only with respect to post layer
   const PVLayerLoc *preLoc  = preSynapticLayer()->getLayerLoc();
   const PVLayerLoc *postLoc = postSynapticLayer()->getLayerLoc();
   // If the connection uses gpu to receive, update all buffers

   // TODO see if you can avoid this step of transferring patches to gpu
   // Based on arborId
   // Other way would be to just allocate all arbors to gpu

   // If more than 1 arbor, need to update patches and GSynPatchStart.
   // If one arbor, done in allocatePreKernel in HyPerConn
   if (numberOfAxonalArborLists() > 1) {
      PVPatch *h_patches            = weights(arborID)[0]; // 0 because it's one block of memory
      PVCuda::CudaBuffer *d_patches = getDevicePatches();
      pvAssert(d_patches);

      d_patches->copyToDevice(h_patches);

      size_t *h_GSynPatchStart             = getGSynPatchStart()[arborID];
      PVCuda::CudaBuffer *d_GSynPatchStart = getDeviceGSynPatchStart();
      pvAssert(d_GSynPatchStart);
      d_GSynPatchStart->copyToDevice(h_GSynPatchStart);
   }

   // Update pre datastore, post gsyn, and conn weights
   // Only if their updated
   if (preSynapticLayer()->getUpdatedDeviceDatastoreFlag()) {
      float *h_preDatastore              = activity->data;
      PVCuda::CudaBuffer *d_preDatastore = preSynapticLayer()->getDeviceDatastore();
      pvAssert(d_preDatastore);
      d_preDatastore->copyToDevice(h_preDatastore);

      // Copy active indices and num active if needed
      if (activity->isSparse) {
         PVCuda::CudaBuffer *d_ActiveIndices;
         PVCuda::CudaBuffer *d_numActive;
         d_ActiveIndices = preSynapticLayer()->getDeviceActiveIndices();
         d_numActive     = preSynapticLayer()->getDeviceNumActive();
         pvAssert(d_ActiveIndices);
         SparseList<float>::Entry const *h_ActiveIndices =
               (SparseList<float>::Entry *)activity->activeIndices;
         long const *h_numActive = activity->numActive;
         pvAssert(h_ActiveIndices);
         d_numActive->copyToDevice(h_numActive);
         d_ActiveIndices->copyToDevice(h_ActiveIndices);
      }
      // Device now has updated
      preSynapticLayer()->setUpdatedDeviceDatastoreFlag(false);
   }

   // X direction is active neuron
   // Y direction is post patch size
   long totActiveNeuron[parent->getNBatch()];
   long maxTotalActiveNeuron = 0;
   for (int b = 0; b < parent->getNBatch(); b++) {
      if (activity->isSparse) {
         totActiveNeuron[b] = activity->numActive[b];
      }
      else {
         totActiveNeuron[b] = preSynapticLayer()->getNumExtended();
      }
      if (totActiveNeuron[b] > maxTotalActiveNeuron) {
         maxTotalActiveNeuron = totActiveNeuron[b];
      }
   }

   long totPatchSize   = xPatchSize() * yPatchSize() * fPatchSize();
   long totThreads     = maxTotalActiveNeuron * totPatchSize;
   int maxThreads      = parent->getDevice()->get_max_threads();
   int numLocalThreads = totPatchSize < maxThreads ? totPatchSize : maxThreads;

   krRecvPre->run_nocheck(totThreads, numLocalThreads);

   return PV_SUCCESS;
}

int HyPerConn::deliverPostsynapticPerspectiveGPU(PVLayerCube const *activity, int arborID) {

   // Check channel number for noupdate
   if (getChannel() == CHANNEL_NOUPDATE) {
      return PV_SUCCESS;
   }
   pvAssert(post->getChannel(getChannel()));

   pvAssert(arborID >= 0);
   // Get number of neurons restricted target
   const int numRestricted = post->getNumNeurons();

   float dtFactor;
   if (getPvpatchAccumulateType() == STOCHASTIC) {
      dtFactor = parent->getDeltaTime();
   }
   else if (getPvpatchAccumulateType() == CONVOLVE) {
      dtFactor = getConvertToRateDeltaTimeFactor();
   }
   else {
      pvAssert(0); // Only STOCHASTIC and CONVOLVE are defined in HyPerConn; other methods should be
      // handled in subclasses.
   }

   pvAssert(krRecvPost);
   krRecvPost->set_dt_factor(dtFactor);

   const PVLayerLoc *sourceLoc = pre->getLayerLoc();
   const PVLayerLoc *targetLoc = post->getLayerLoc();
   const PVHalo *sourceHalo    = &sourceLoc->halo;

   const int sourceNx = sourceLoc->nx;
   const int sourceNy = sourceLoc->ny;
   const int sourceNf = sourceLoc->nf;
   const int targetNx = targetLoc->nx;
   const int targetNy = targetLoc->ny;
   const int targetNf = targetLoc->nf;

   // get source layer's extended y stride
   int sy = (sourceNx + sourceHalo->rt + sourceHalo->lt) * sourceNf;
   // get source layer's patch y stride
   int syp = postConn->yPatchStride();
   // Iterate through y patch
   int numPerStride = postConn->xPatchSize() * postConn->fPatchSize();

   long *startSourceExtBuf = getPostToPreActivity();
   if (!startSourceExtBuf) {
      Fatal() << "HyPerLayer::recvFromPost unable to get preToPostActivity from connection. Is "
                 "shrink_patches on?\n";
   }

   bool updatePreAct = false;
   // Update pre activity, post gsyn, and conn weights
   // Only if they're updated
   if (pre->getUpdatedDeviceDatastoreFlag()) {
      float *h_preDatastore              = activity->data;
      PVCuda::CudaBuffer *d_preDatastore = pre->getDeviceDatastore();
      pvAssert(d_preDatastore);
      d_preDatastore->copyToDevice(h_preDatastore);
      // Device now has updated
      pre->setUpdatedDeviceDatastoreFlag(false);
      updatePreAct = true;
   }

#ifdef PV_USE_CUDNN
   // Permutation buffer is local to the kernel, NOT the layer
   // Therefore, we must permute Datastore every time
   krRecvPost->permuteDatastorePVToCudnn();
   //}

   // Permute GSyn
   krRecvPost->permuteGSynPVToCudnn(getChannel());
#endif // PV_USE_CUDA

   int totF = targetNf;
   int totX = targetNx;
   int totY = targetNy;
   // Make sure local sizes are divisible by f, x, and y
   krRecvPost->run(totX, totY, totF, 1L, 1L, 1L);

#ifdef PV_USE_CUDNN
   krRecvPost->permuteGSynCudnnToPV(getChannel());
#endif

   return PV_SUCCESS;
}
#endif // PV_USE_CUDA

void HyPerConn::deliverOnePostNeuronActivity(
      int arborID,
      int kTargetExt,
      int inSy,
      float *activityStartBuf,
      float *gSynPatchPos,
      float dtFactor,
      taus_uint4 *rngPtr) {
   // get source layer's patch y stride
   int syp        = postConn->yPatchStride();
   int yPatchSize = postConn->yPatchSize();
   // Iterate through y patch
   int numPerStride = postConn->xPatchSize() * postConn->fPatchSize();
   int kernelIndex  = postConn->patchToDataLUT(kTargetExt);

   float *weightStartBuf = postConn->get_wDataHead(arborID, kernelIndex);
   int sf                = 1;
   int offset            = 0;
   for (int ky = 0; ky < yPatchSize; ky++) {
      float *activityY = &(activityStartBuf[ky * inSy + offset]);
      float *weightY   = weightStartBuf + ky * syp;
      // TODO add sf here
      (accumulateFunctionFromPostPointer)(
            0, numPerStride, gSynPatchPos, activityY, weightY, dtFactor, rngPtr, sf);
   }
}

void HyPerConn::deliverOnePreNeuronActivity(
      int kPreExt,
      int arbor,
      float a,
      float *postBufferStart,
      void *auxPtr) {
   PVPatch *weights       = getWeights(kPreExt, arbor);
   const int nk           = weights->nx * fPatchSize();
   const int ny           = weights->ny;
   const int sy           = getPostNonextStrides()->sy; // stride in layer
   const int syw          = yPatchStride(); // stride in patch
   float *weightDataStart = NULL;
   float *postPatchStart  = postBufferStart + getGSynPatchStart(kPreExt, arbor);
   int offset             = 0;
   int sf                 = 1;
   weightDataStart        = get_wData(arbor, kPreExt); // make this a float const *?
   for (int y = 0; y < ny; y++) {
      (accumulateFunctionPointer)(
            0,
            nk,
            postPatchStart + y * sy + offset,
            a,
            weightDataStart + y * syw + offset,
            auxPtr,
            sf);
   }
}

int HyPerConn::createWeights(
      PVPatch ***patches,
      int nWeightPatches,
      int nDataPatches,
      int nxPatch,
      int nyPatch,
      int nfPatch,
      int arborId) {
   pvAssert(patches[arborId] == NULL);

   if (shrinkPatches_flag || arborId == 0) {
      patches[arborId] = createPatches(nWeightPatches, nxPatch, nyPatch);
      pvAssert(patches[arborId] != NULL);
   }
   else {
      patches[arborId] = patches[0];
   }
   return PV_SUCCESS;
}

double HyPerConn::getConvertToRateDeltaTimeFactor() {
   double dtFactor = 1.0;
   // if (preActivityIsNotRate) { // preActivityIsNotRate was replaced with convertRateToSpikeCount
   // on Dec 31, 2014
   if (convertRateToSpikeCount && !pre->activityIsSpiking()) {
      enum ChannelType channel_type = getChannel();
      float dt                      = parent->getDeltaTime();
      float tau                     = post->getChannelTimeConst(channel_type);
      if (tau > 0) {
         double exp_dt_tau = exp(-dt / tau);
         dtFactor          = (1.0 - exp_dt_tau) / exp_dt_tau;
         // the above factor was chosen so that for a constant input of G_SYN to an excitatory
         // conductance G_EXC,
         // then G_EXC -> G_SYN as t -> inf
      }
      else {
         dtFactor = dt;
      }
   }
   return dtFactor;
}

/**
 * Create a separate patch of weights for every neuron
 */
int HyPerConn::createWeights(PVPatch ***patches, int arborId) {
   int nWeightPatches = getNumWeightPatches();
   int nDataPatches   = getNumDataPatches();
   int nxPatch        = nxp;
   int nyPatch        = nyp;
   int nfPatch        = nfp;

   int status =
         createWeights(patches, nWeightPatches, nDataPatches, nxPatch, nyPatch, nfPatch, arborId);
   return status;
}

int HyPerConn::clearWeights(float **dataStart, int numPatches, int nxp, int nyp, int nfp) {
   int status = PV_SUCCESS;
   for (int arborID = 0; arborID < numAxonalArborLists; arborID++) {
      if (clearWeights(dataStart[arborID], numPatches, nxp, nyp, nfp) != PV_SUCCESS)
         status = PV_FAILURE;
   }
   return status;
}

int HyPerConn::clearWeights(float *arborDataStart, int numPatches, int nxp, int nyp, int nfp) {
   for (long int w = 0; w < patchStartIndex(numPatches); w++) {
      arborDataStart[w] = 0.0f;
   }
   return PV_SUCCESS;
}

int HyPerConn::deleteWeights() {
   // to be used if createPatches is used above
   if (wPatches != NULL) {
      for (int arbor = 0; arbor < numAxonalArborLists; arbor++) {
         if (wPatches[arbor] != NULL) {
            if (shrinkPatches_flag || arbor == 0) {
               deletePatches(wPatches[arbor]);
            }
            wPatches[arbor] = NULL;
         }
      } // arbor
      free(wPatches);
      wPatches = NULL;
   } // wPatches != NULL

   if (wDataStart != NULL) {
      for (int arbor = 0; arbor < numAxonalArborLists; arbor++) {
         // entire arbor allocated as single block
         if (arbor == 0) {
            if (wDataStart[arbor] != NULL) {
               free(wDataStart[arbor]);
            }
         } // arbor == 0
         wDataStart[arbor] = NULL;
         if (!combine_dW_with_W_flag) {
            if (dwDataStart != NULL && dwDataStart[arbor] != NULL) {
               free(dwDataStart[arbor]);
               dwDataStart[arbor] = NULL;
            }
         }
      } // arbor
      free(wDataStart);
      wDataStart = NULL;
      if (!combine_dW_with_W_flag) {
         free(dwDataStart);
      }
      dwDataStart = NULL;
   } // wDataStart != NULL

   if (numKernelActivations != NULL) {
      free(numKernelActivations[0]);
      free(numKernelActivations);
   }

   if (wPostPatches != NULL) {
      for (int arborID = 0; arborID < numberOfAxonalArborLists(); arborID++) {
         if (wPostPatches[arborID] != NULL) {
            if (shrinkPatches_flag || arborID == 0) {
               deletePatches(wPostPatches[arborID]);
            }
            wPostPatches[arborID] = NULL;
         }

         if (wPostDataStart != NULL) {
            free(wPostDataStart[arborID]);
            wPostDataStart[arborID] = NULL;
         }
      }
      free(wPostPatches);
      wPostPatches = NULL;
      free(wPostDataStart);
      wPostDataStart = NULL;
   } // wPostPatches != NULL

   if (gSynPatchStart != NULL) {
      free(gSynPatchStart[0]); // All gSynPatchStart[k]'s were allocated together in a single malloc
      // call.
      free(gSynPatchStart);
   }
   if (aPostOffset != NULL) {
      free(aPostOffset[0]); // All aPostOffset[k]'s were allocated together in a single malloc call.
      free(aPostOffset);
   }
   free(patch2datalookuptable);
   patch2datalookuptable = NULL;

   return PV_SUCCESS;
}

// This function is doing what adjust axonal arbors was doing before, but generalized to not use the
// pre/post layer's pre/post for use with gpu post groups
int HyPerConn::adjustAllPatches(
      int nxPre,
      int nyPre,
      int nfPre,
      const PVHalo *haloPre,
      int nxPost,
      int nyPost,
      int nfPost,
      const PVHalo *haloPost,
      PVPatch ***inWPatches,
      size_t **inGSynPatchStart,
      size_t **inAPostOffset,
      int arborId) {

   const int xScaleDiff               = pre->getXScale() - post->getXScale();
   const int xPostNeuronsPerPreNeuron = xScaleDiff > 0 ? (int)pow(2, xScaleDiff) : 1;
   const int xPreNeuronsPerPostNeuron = xScaleDiff < 0 ? (int)pow(2, -xScaleDiff) : 1;

   const int yScaleDiff               = pre->getYScale() - post->getYScale();
   const int yPostNeuronsPerPreNeuron = yScaleDiff > 0 ? (int)pow(2, yScaleDiff) : 1;
   const int yPreNeuronsPerPostNeuron = yScaleDiff < 0 ? (int)pow(2, -yScaleDiff) : 1;

   // can't use member variable numWeightPatches because GPUs might call this routine with a smaller
   // block.  Calculate from input arguments
   int numWeightPatches =
         (nxPre + haloPre->lt + haloPre->rt) * (nyPre + haloPre->up + haloPre->dn) * nfPre;
   for (int kex = 0; kex < numWeightPatches; kex++) {
      // calculate start of patch in postsynaptic restricted coordinates, and width of patch in
      // postsynaptic restricted coordinates
      int xPre =
            kxPos(kex, nxPre + haloPre->lt + haloPre->rt, nyPre + haloPre->dn + haloPre->up, nfPre)
            - haloPre->lt; // x-coordinate of presynaptic neuron tied to patch kex, in presynaptic
      // restricted coordinates.
      int xPostStart, xPatchStart, nxPatch;
      int status = adjustedPatchDimension(
            xPre,
            xPreNeuronsPerPostNeuron,
            xPostNeuronsPerPreNeuron,
            nxPost,
            nxp,
            &xPostStart,
            &xPatchStart,
            &nxPatch);
      int yPre =
            kyPos(kex, nxPre + haloPre->lt + haloPre->rt, nyPre + haloPre->dn + haloPre->up, nfPre)
            - haloPre->up; // y-coordinate of presynaptic neuron tied to patch kex, in presynaptic
      // restricted coordinates.
      int yPostStart, yPatchStart, nyPatch;
      status = adjustedPatchDimension(
            yPre,
            yPreNeuronsPerPostNeuron,
            yPostNeuronsPerPreNeuron,
            nyPost,
            nyp,
            &yPostStart,
            &yPatchStart,
            &nyPatch);

      if (inAPostOffset) {
         inAPostOffset[arborId][kex] = (size_t)kIndex(
               xPostStart + haloPost->lt,
               yPostStart + haloPost->up,
               0,
               nxPost + haloPost->lt + haloPost->rt,
               nyPost + haloPost->dn + haloPost->up,
               nfPost);
      }

      inGSynPatchStart[arborId][kex] =
            (size_t)kIndex(xPostStart, yPostStart, 0, nxPost, nyPost, nfPost);

      PVPatch *w = inWPatches[arborId][kex];
      pvAssert(w->offset == 0);
      pvpatch_adjust(w, sxp, syp, nxPatch, nyPatch, xPatchStart, yPatchStart);
   } // loop over patches

   return PV_SUCCESS;
}

//!
/*!
 *
 *      - Each neuron in the pre-synaptic layer projects a number of axonal
 *      arbors to the post-synaptic layer (Can they be projected accross columns too?).
 *      - numAxons is the number of axonal arbors projected by each neuron.
 *      - Each axonal arbor (PVAxonalArbor) connects to a patch of neurons in the post-synaptic
 * layer.
 *      - The PVAxonalArbor structure contains STDP P variable.
 *      -
 *      .
 *
 * REMARKS:
 *      - numArbors = (nxPre + 2*prePad)*(nyPre+2*prePad) = nxexPre * nyexPre
 *      This is the total number of weight patches for a given arbor.
 *      Is the number of pre-synaptic neurons including margins.
 *      - activity and STDP M variable are extended into margins
 *      .
 *
 */
int HyPerConn::adjustAxonalArbors(int arborId) {

   const int nxPre        = pre->getLayerLoc()->nx;
   const int nyPre        = pre->getLayerLoc()->ny;
   const int nfPre        = pre->getLayerLoc()->nf;
   const PVHalo *haloPre  = &pre->getLayerLoc()->halo;
   const int nxPost       = post->getLayerLoc()->nx;
   const int nyPost       = post->getLayerLoc()->ny;
   const int nfPost       = post->getLayerLoc()->nf;
   const PVHalo *haloPost = &post->getLayerLoc()->halo;

   return adjustAllPatches(
         nxPre,
         nyPre,
         nfPre,
         haloPre,
         nxPost,
         nyPost,
         nfPost,
         haloPost,
         wPatches,
         gSynPatchStart,
         aPostOffset,
         arborId);
}

PVPatch ***HyPerConn::convertPreSynapticWeights(double simTime) {
   if (simTime <= wPostTime) {
      return wPostPatches;
   }
   wPostTime = simTime;

   const PVLayerLoc *preLoc  = pre->getLayerLoc();
   const PVLayerLoc *postLoc = post->getLayerLoc();

   const int xScale       = post->getXScale() - pre->getXScale();
   const int yScale       = post->getYScale() - pre->getYScale();
   const double powXScale = pow(2.0f, (double)xScale);
   const double powYScale = pow(2.0f, (double)yScale);

   // pre-synaptic weights are in extended layer reference frame
   const int nxPre = preLoc->nx + preLoc->halo.lt + preLoc->halo.rt;
   const int nyPre = preLoc->ny + preLoc->halo.dn + preLoc->halo.up;
   const int nfPre = preLoc->nf;

   const int nxPost  = postLoc->nx;
   const int nyPost  = postLoc->ny;
   const int nfPost  = postLoc->nf;
   const int numPost = post->getNumNeurons();

   nxpPost = (int)(nxp * powXScale);
   nypPost = (int)(nyp * powYScale);
   nfpPost = preLoc->nf;

   int sxPost = nfpPost;
   int syPost = sxPost * nxpPost;
   int spPost = syPost * nypPost;

   // the number of features is the end-point value (normally post-synaptic)
   const int numPostPatch = nxpPost * nypPost * nfpPost; // Post-synaptic weights are never shrunken

   if (wPostPatches == NULL) {
      wPostPatches = (PVPatch ***)pvCalloc(numAxonalArborLists, sizeof(PVPatch **));
      pvAssert(wPostDataStart == NULL);
      wPostDataStart    = (float **)pvCalloc(numAxonalArborLists, sizeof(float *));
      wPostDataStart[0] = allocWeights(numPost, nxpPost, nypPost, nfpPost);
      pvAssert(wPostDataStart[0] != NULL);
      for (int arborID = 0; arborID < numberOfAxonalArborLists(); arborID++) {
         int status =
               createWeights(wPostPatches, numPost, numPost, nxpPost, nypPost, nfpPost, arborID);
         pvAssert(status == PV_SUCCESS);
         if (arborID > 0) { // wDataStart already allocated
            wPostDataStart[arborID] = wPostDataStart[0] + spPost * numPost * arborID;
            pvAssert(wPostDataStart[arborID] != NULL);
         }
      }
   }

   // loop through all axons:
   for (int arborID = 0; arborID < numberOfAxonalArborLists(); arborID++) {

      // loop through post-synaptic neurons (non-extended indices)

      for (int kPost = 0; kPost < numPost; kPost++) {
         int kxPost = kxPos(kPost, nxPost, nyPost, nfPost);
         int kyPost = kyPos(kPost, nxPost, nyPost, nfPost);
         int kfPost = featureIndex(kPost, nxPost, nyPost, nfPost);

         int kxPreHead = zPatchHead(kxPost, nxpPost, post->getXScale(), pre->getXScale());
         int kyPreHead = zPatchHead(kyPost, nypPost, post->getYScale(), pre->getYScale());

         // convert kxPreHead and kyPreHead to extended indices
         kxPreHead += preLoc->halo.lt;
         kyPreHead += preLoc->halo.up;

         float *postData = wPostDataStart[arborID] + nxpPost * nypPost * nfpPost * kPost
                           + wPostPatches[arborID][kPost]->offset;
         for (int kp = 0; kp < numPostPatch; kp++) {

            // calculate extended indices of presynaptic neuron {kPre, kzPre}
            int kxPostPatch = (int)kxPos(kp, nxpPost, nypPost, nfPre);
            int kyPostPatch = (int)kyPos(kp, nxpPost, nypPost, nfPre);
            int kfPostPatch = (int)featureIndex(kp, nxpPost, nypPost, nfPre);

            int kxPre = kxPreHead + kxPostPatch;
            int kyPre = kyPreHead + kyPostPatch;
            int kfPre = kfPostPatch;
            int kPre  = kIndex(kxPre, kyPre, kfPre, nxPre, nyPre, nfPre);

            // if {kPre, kzPre} out of bounds, set post weight to zero
            if (kxPre < 0 || kyPre < 0 || kxPre >= nxPre || kyPre >= nyPre) {
               pvAssert(kxPre < 0 || kyPre < 0 || kxPre >= nxPre || kyPre >= nyPre);
               postData[kp] = 0.0;
            }
            else {
               // {kzPostHead} store the restricted indices of the postsynaptic patch head
               int kxPostHead, kyPostHead, kfPostHead;
               int nxp_post, nyp_post; // shrunken patch dimensions
               int dx_nxp, dy_nyp; // shrinkage

               postSynapticPatchHead(
                     kPre,
                     &kxPostHead,
                     &kyPostHead,
                     &kfPostHead,
                     &dx_nxp,
                     &dy_nyp,
                     &nxp_post,
                     &nyp_post);

               int kxPrePatch, kyPrePatch; // relative index in shrunken patch
               kxPrePatch     = kxPost - kxPostHead;
               kyPrePatch     = kyPost - kyPostHead;
               int kPrePatch  = kfPost * sfp + kxPrePatch * sxp + kyPrePatch * syp;
               float *preData = get_wDataStart(arborID) + patchStartIndex(kPre)
                                + getWeights(kPre, arborID)->offset;
               postData[kp] = preData[kPrePatch];
            }
         }
      }
   }
   return wPostPatches;
}

PVPatch ****HyPerConn::point2PreSynapticWeights() {

   const PVLayerLoc *preLoc  = pre->getLayerLoc();
   const PVLayerLoc *postLoc = post->getLayerLoc();

   const int xScale       = post->getXScale() - pre->getXScale();
   const int yScale       = post->getYScale() - pre->getYScale();
   const double powXScale = pow(2.0f, (double)xScale);
   const double powYScale = pow(2.0f, (double)yScale);

   // pre-synaptic weights are in extended layer reference frame
   const int nxPre = preLoc->nx + preLoc->halo.lt + preLoc->halo.rt;
   const int nyPre = preLoc->ny + preLoc->halo.dn + preLoc->halo.up;
   const int nfPre = preLoc->nf;

   const int nxPost  = postLoc->nx;
   const int nyPost  = postLoc->ny;
   const int nfPost  = postLoc->nf;
   const int numPost = post->getNumNeurons();

   nxpPost = (int)(nxp * powXScale);
   nypPost = (int)(nyp * powYScale);
   nfpPost = preLoc->nf;
   float z = 0;

   // the number of features is the end-point value (normally post-synaptic)
   const int numPostPatch = nxpPost * nypPost * nfpPost; // Post-synaptic weights are never shrunken

   if (wPostPatchesp == NULL) {

      // Return data structure
      wPostPatchesp = (PVPatch ****)pvCalloc(numAxonalArborLists, sizeof(PVPatch ***));
      pvAssert(wPostDataStartp == NULL);
      wPostDataStartp = (float ***)pvCalloc(numAxonalArborLists, sizeof(float **));

      for (int arborID = 0; arborID < numberOfAxonalArborLists(); arborID++) {

         wPostPatchesp[arborID] = (PVPatch ***)pvCalloc(numPost, sizeof(PVPatch **));

         int sx = nfpPost;
         int sy = sx * nxpPost;
         int sp = sy * nypPost;

         size_t patchSize = sp * sizeof(float);
         size_t dataSize  = numPost * patchSize;

         wPostDataStartp[arborID] = (float **)pvCalloc(dataSize, sizeof(char *));

         PVPatch **patcharray = (PVPatch **)pvCalloc(numPost, sizeof(PVPatch *));
         PVPatch **curpatch   = patcharray;
         for (int i = 0; i < numPost; i++) {
            wPostPatchesp[arborID][i] = curpatch;
            curpatch++;
         }
      }
   }

   // loop through all arbors:
   for (int arborID = 0; arborID < numberOfAxonalArborLists(); arborID++) {

      // loop through post-synaptic neurons (non-extended indices)
      for (int kPost = 0; kPost < numPost; kPost++) {
         int kxPost = kxPos(kPost, nxPost, nyPost, nfPost);
         int kyPost = kyPos(kPost, nxPost, nyPost, nfPost);
         int kfPost = featureIndex(kPost, nxPost, nyPost, nfPost);

         int kxPreHead = zPatchHead(kxPost, nxpPost, post->getXScale(), pre->getXScale());
         int kyPreHead = zPatchHead(kyPost, nypPost, post->getYScale(), pre->getYScale());

         // convert kxPreHead and kyPreHead to extended indices
         kxPreHead += preLoc->halo.lt;
         kyPreHead += preLoc->halo.up;

         // Accessing by patch offset through wPostDataStart by x,y,and feature of a patch
         float **postData = wPostDataStartp[arborID] + nxpPost * nypPost * nfpPost * kPost + 0;
         for (int kp = 0; kp < numPostPatch; kp++) {

            // calculate extended indices of presynaptic neuron {kPre, kzPre}
            int kxPostPatch = (int)kxPos(kp, nxpPost, nypPost, nfPre);
            int kyPostPatch = (int)kyPos(kp, nxpPost, nypPost, nfPre);
            int kfPostPatch = (int)featureIndex(kp, nxpPost, nypPost, nfPre);

            int kxPre = kxPreHead + kxPostPatch;
            int kyPre = kyPreHead + kyPostPatch;
            int kfPre = kfPostPatch;
            int kPre  = kIndex(kxPre, kyPre, kfPre, nxPre, nyPre, nfPre);

            // if {kPre, kzPre} out of bounds, set post weight to zero
            if (kxPre < 0 || kyPre < 0 || kxPre >= nxPre || kyPre >= nyPre) {
               pvAssert(kxPre < 0 || kyPre < 0 || kxPre >= nxPre || kyPre >= nyPre);
               postData[kp] = &z;
            }
            else {
               // {kzPostHead} store the restricted indices of the postsynaptic patch head
               int kxPostHead, kyPostHead, kfPostHead;
               int nxp_post, nyp_post; // shrunken patch dimensions
               int dx_nxp, dy_nyp; // shrinkage

               postSynapticPatchHead(
                     kPre,
                     &kxPostHead,
                     &kyPostHead,
                     &kfPostHead,
                     &dx_nxp,
                     &dy_nyp,
                     &nxp_post,
                     &nyp_post);

               int kxPrePatch, kyPrePatch; // relative index in shrunken patch
               kxPrePatch     = kxPost - kxPostHead;
               kyPrePatch     = kyPost - kyPostHead;
               int kPrePatch  = kfPost * sfp + kxPrePatch * sxp + kyPrePatch * syp;
               float *preData = get_wDataStart(arborID) + patchStartIndex(kPre)
                                + getWeights(kPre, arborID)->offset;
               postData[kp] = &(preData[kPrePatch]);
            }
         }
      }
   }
   return wPostPatchesp;
}

/**
 * Returns the head (kxPre, kyPre) of a pre-synaptic patch given post-synaptic layer indices.
 * @kxPost the post-synaptic kx index (non-extended units)
 * @kyPost the post-synaptic ky index (non-extended units)
 * @kfPost the post-synaptic kf index
 * @kxPre address of the kx index in the pre-synaptic layer (non-extended units) on output
 * @kyPre address of the ky index in the pre-synaptic layer (non-extended units) on output
 *
 * NOTE: kxPre and kyPre may be in the border region
 */
int HyPerConn::preSynapticPatchHead(int kxPost, int kyPost, int kfPost, int *kxPre, int *kyPre) {
   int status = 0;

   const int xScale       = post->getXScale() - pre->getXScale();
   const int yScale       = post->getYScale() - pre->getYScale();
   const double powXScale = pow(2, (double)xScale);
   const double powYScale = pow(2, (double)yScale);

   const int nxPostPatch = (int)(nxp * powXScale);
   const int nyPostPatch = (int)(nyp * powYScale);

   int kxPreHead = zPatchHead(kxPost, nxPostPatch, post->getXScale(), pre->getXScale());
   int kyPreHead = zPatchHead(kyPost, nyPostPatch, post->getYScale(), pre->getYScale());

   *kxPre = kxPreHead;
   *kyPre = kyPreHead;

   return status;
}

/**
 * Returns the head (kxPostOut, kyPostOut) of the post-synaptic patch plus other
 * patch information.
 * @kPreEx the pre-synaptic k index (extended units)
 * @kxPostOut address of the kx index in post layer (non-extended units) on output
 * @kyPostOut address of the ky index in post layer (non-extended units) on output
 * @kfPostOut address of the kf index in post layer (non-extended units) on output
 * @dxOut address of the change in x dimension size of patch (to fit border) on output
 * @dyOut address of the change in y dimension size of patch (to fit border) on output
 * @nxpOut address of x dimension patch size (includes border reduction) on output
 * @nypOut address of y dimension patch size (includes border reduction) on output
 *
 * NOTE: kxPostOut and kyPostOut are always within the post-synaptic
 * non-extended layer because the patch size is reduced at borders
 */
int HyPerConn::postSynapticPatchHead(
      int kPreEx,
      int *kxPostOut,
      int *kyPostOut,
      int *kfPostOut,
      int *dxOut,
      int *dyOut,
      int *nxpOut,
      int *nypOut) {
   int status = 0;

   const PVLayerLoc *preLoc  = pre->getLayerLoc();
   const PVLayerLoc *postLoc = post->getLayerLoc();

   const int kx0Pre = preLoc->kx0;
   const int ky0Pre = preLoc->ky0;
   const int nfPre  = preLoc->nf;

   const int nxexPre = preLoc->nx + preLoc->halo.lt + preLoc->halo.rt;
   const int nyexPre = preLoc->ny + preLoc->halo.dn + preLoc->halo.up;

   const int nxPost  = postLoc->nx;
   const int nyPost  = postLoc->ny;
   const int kx0Post = postLoc->kx0;
   const int ky0Post = postLoc->ky0;

   // kPreEx is in extended frame, this makes transformations more difficult
   //

   // local indices in extended frame
   //
   int kxPre = kxPos(kPreEx, nxexPre, nyexPre, nfPre);
   int kyPre = kyPos(kPreEx, nxexPre, nyexPre, nfPre);

   // convert to global non-extended frame
   //
   kxPre += kx0Pre - preLoc->halo.lt;
   kyPre += ky0Pre - preLoc->halo.up;

   // global non-extended post-synaptic frame
   //
   int kxPost = zPatchHead(kxPre, nxp, pre->getXScale(), post->getXScale());
   int kyPost = zPatchHead(kyPre, nyp, pre->getYScale(), post->getYScale());

   // TODO - can get nf from weight patch but what about kf0?
   // weight patch is actually a pencil and so kfPost is always 0?
   int kfPost = 0;

   // convert to local non-extended post-synaptic frame
   kxPost = kxPost - kx0Post;
   kyPost = kyPost - ky0Post;

   // adjust location so patch is in bounds
   int dx      = 0;
   int dy      = 0;
   int nxPatch = nxp;
   int nyPatch = nyp;

   if (kxPost < 0) {
      nxPatch -= -kxPost;
      kxPost = 0;
      if (nxPatch < 0)
         nxPatch = 0;
      dx         = nxp - nxPatch;
   }
   else if (kxPost + nxp > nxPost) {
      nxPatch -= kxPost + nxp - nxPost;
      if (nxPatch <= 0) {
         nxPatch = 0;
         kxPost  = nxPost - 1;
      }
   }

   if (kyPost < 0) {
      nyPatch -= -kyPost;
      kyPost = 0;
      if (nyPatch < 0)
         nyPatch = 0;
      dy         = nyp - nyPatch;
   }
   else if (kyPost + nyp > nyPost) {
      nyPatch -= kyPost + nyp - nyPost;
      if (nyPatch <= 0) {
         nyPatch = 0;
         kyPost  = nyPost - 1;
      }
   }

   // if out of bounds in x (y), also out in y (x)
   //
   if (nxPatch == 0 || nyPatch == 0) {
      dx      = 0;
      dy      = 0;
      nxPatch = 0;
      nyPatch = 0;
      WarnLog().printf("HyPerConn::postSynapticPatchHead: patch size is zero\n");
   }

   *kxPostOut = kxPost;
   *kyPostOut = kyPost;
   *kfPostOut = kfPost;

   *dxOut  = dx;
   *dyOut  = dy;
   *nxpOut = nxPatch;
   *nypOut = nyPatch;

   return status;
}

// writePostSynapticWeights was removed Apr 28, 2017. Create a TransposeConn if needed.

int HyPerConn::sumWeights(
      int nx,
      int ny,
      int offset,
      float *dataStart,
      double *sum,
      double *sum2,
      float *maxVal) {
   // TODO CER - should make volatile conditional on GPU usage (this could be slow otherwise)?
   volatile float *w = dataStart + offset;
   float sum_tmp     = 0;
   float sum2_tmp    = 0;
   float max_tmp     = -FLT_MAX;
   for (int ky = 0; ky < ny; ky++) {
      for (int iWeight = 0; iWeight < syp; iWeight++) {
         sum_tmp += w[iWeight];
         sum2_tmp += w[iWeight] * w[iWeight];
         max_tmp = (max_tmp > w[iWeight]) ? max_tmp : w[iWeight];
      }
      w += syp;
   }
   *sum    = sum_tmp;
   *sum2   = sum2_tmp;
   *maxVal = max_tmp;
   return PV_SUCCESS;
} // sumWeights

int HyPerConn::checkPatchDimensions() {
   int statusx = checkPatchSize(nxp, pre->getXScale(), post->getXScale(), 'x');
   int statusy = checkPatchSize(nyp, pre->getYScale(), post->getYScale(), 'y');
   int status  = statusx == PV_SUCCESS && statusy == PV_SUCCESS ? PV_SUCCESS : PV_FAILURE;
   return status;
}

int HyPerConn::checkPatchSize(int patchSize, int scalePre, int scalePost, char dim) {
   int scaleDiff = scalePre - scalePost;
   bool goodsize;

   if (scaleDiff == 0) {
      // complain if patchSize is not an odd number
      goodsize = patchSize > 0 && patchSize % 2 == 1;
   }
   else if (scaleDiff > 0) {
      // complain if patchSize is not a multiple of 2^scaleDiff
      int scaleFactor           = (int)pow(2, (double)scaleDiff);
      int multipleOfScaleFactor = patchSize / scaleFactor;
      goodsize = multipleOfScaleFactor > 0 && patchSize == multipleOfScaleFactor * scaleFactor;
   }
   else {
      pvAssert(scaleDiff < 0);
      // any patch size is allowed
      goodsize = true;
   }
   if (!goodsize) {
      Fatal(errorMessage);
      errorMessage.printf("Error:  Connection: %s\n", name);
      errorMessage.printf("Presynaptic layer:  %s\n", pre->getName());
      errorMessage.printf("Postsynaptic layer: %s\n", post->getName());
      errorMessage.printf(
            "Patch size n%cp=%d is not compatible with presynaptic n%cScale %f\n",
            dim,
            patchSize,
            dim,
            pow(2, -scalePre));
      errorMessage.printf("and postsynaptic n%cScale %f.\n", dim, pow(2, -scalePost));
      if (scaleDiff == 0) {
         errorMessage.printf("(presynaptic scale) == (postsynaptic scale);\n");
         errorMessage.printf("therefore patch size must be odd\n");
      }
      if (scaleDiff > 0) {
         int scaleFactor = (int)pow(2, (float)scaleDiff);
         errorMessage.printf("(postsynaptic scale) = %d * (presynaptic scale);\n", scaleFactor);
         errorMessage.printf("therefore compatible sizes are multiples of %d.\n", scaleFactor);
      }
      else {
         // If scaleDiff < 0 any patch size is acceptable
         pvAssert(0);
      }
      errorMessage.printf("Exiting.\n");
      // errorMessage declared using Fatal, so program exits here.
   }
   return PV_SUCCESS;
}

int HyPerConn::setPatchStrides() {
   // these strides are for the weight patches
   sfp = 1;
   sxp = nfp;
   syp = nfp * nxp;

   // these strides are for a post-synaptic non-extended layer variable.
   // HyPerLayer::recvSynapticInput uses these strides for GSyn, which is nonextended
   postNonextStrides.sf = 1;
   postNonextStrides.sx = nfp;
   postNonextStrides.sy = nfp * post->getLayerLoc()->nx;

   // these strides are for a post-synaptic extended layer variable.
   postExtStrides.sf = 1;
   postExtStrides.sx = nfp;
   postExtStrides.sy = nfp * (post->getLayerLoc()->nx + post->getLayerLoc()->halo.lt
                              + post->getLayerLoc()->halo.rt);

   return PV_SUCCESS;
}

float *HyPerConn::allocWeights(int nPatches, int nxPatch, int nyPatch, int nfPatch) {
   bool overflow = false; // Do sanity checking on the size of the weight allocation.

   int sx = nfPatch;
   int sy = sx * nxPatch;
   if (sy / sx != nxPatch) {
      overflow = true;
   }
   int sp = sy * nyPatch;
   if (sp / sy != nyPatch) {
      overflow = true;
   }

   size_t patchSize = sp * sizeof(float);
   if (patchSize / sp != sizeof(float)) {
      overflow = true;
   }
   size_t dataSize = nPatches * patchSize;
   if (dataSize / nPatches != patchSize) {
      overflow = true;
   }
   size_t arborSize = dataSize * numberOfAxonalArborLists();
   if (arborSize / dataSize != numberOfAxonalArborLists()) {
      overflow = true;
   }

   if (overflow) {
      Fatal().printf(
            "%s is too big (%d patches of size nxPatch=%d by nyPatch=%d by nfPatch=%d; %d arbors, "
            "weight size=%zu bytes).  Exiting.\n",
            getDescription_c(),
            nPatches,
            nxPatch,
            nyPatch,
            nfPatch,
            numberOfAxonalArborLists(),
            sizeof(float));
   }

   return (float *)pvCallocError(
         arborSize, sizeof(char), "Error allocating weights for %s", getDescription_c());
}

int HyPerConn::patchToDataLUT(int patchIndex) {
   return sharedWeights ? patch2datalookuptable[patchIndex] : patchIndex;
}

int HyPerConn::patchIndexToDataIndex(
      int patchIndex,
      int *kx /*default=NULL*/,
      int *ky /*default=NULL*/,
      int *kf /*default=NULL*/) {
   int dataIndex;
   if (sharedWeights) {
      dataIndex = calcUnitCellIndex(patchIndex, kx, ky, kf);
   }
   else {
      const PVLayerLoc *preLoc = pre->getLayerLoc();
      if (kx)
         *kx =
               kxPos(patchIndex,
                     preLoc->nx + preLoc->halo.lt + preLoc->halo.rt,
                     preLoc->ny + preLoc->halo.dn + preLoc->halo.up,
                     preLoc->nf);
      if (ky)
         *ky =
               kyPos(patchIndex,
                     preLoc->nx + preLoc->halo.lt + preLoc->halo.rt,
                     preLoc->ny + preLoc->halo.dn + preLoc->halo.up,
                     preLoc->nf);
      if (kf)
         *kf = featureIndex(
               patchIndex,
               preLoc->nx + preLoc->halo.lt + preLoc->halo.rt,
               preLoc->ny + preLoc->halo.dn + preLoc->halo.up,
               preLoc->nf);
      dataIndex = patchIndex;
   }
   return dataIndex;
}

int HyPerConn::dataIndexToUnitCellIndex(
      int dataIndex,
      int *kx /*default=NULL*/,
      int *ky /*default=NULL*/,
      int *kf /*default=NULL*/) {
   int unitCellIndex;
   if (sharedWeights) {
      int nfUnitCell = pre->getLayerLoc()->nf;
      int nxUnitCell = zUnitCellSize(pre->getXScale(), post->getXScale());
      int nyUnitCell = zUnitCellSize(pre->getYScale(), post->getYScale());
      pvAssert(dataIndex >= 0 && dataIndex < nxUnitCell * nyUnitCell * nfUnitCell);
      if (kx)
         *kx = kxPos(dataIndex, nxUnitCell, nyUnitCell, nfUnitCell);
      if (ky)
         *ky = kyPos(dataIndex, nxUnitCell, nyUnitCell, nfUnitCell);
      if (kf)
         *kf        = featureIndex(dataIndex, nxUnitCell, nyUnitCell, nfUnitCell);
      unitCellIndex = dataIndex;
   }
   else {
      unitCellIndex = calcUnitCellIndex(dataIndex, kx, ky, kf);
   }
   return unitCellIndex;
}

int HyPerConn::calcUnitCellIndex(
      int patchIndex,
      int *kxUnitCellIndex /*default=NULL*/,
      int *kyUnitCellIndex /*default=NULL*/,
      int *kfUnitCellIndex /*default=NULL*/) {
   const PVLayerLoc *preLoc = pre->getLayerLoc();
   int nxUnitCell           = zUnitCellSize(pre->getXScale(), post->getXScale());
   int nyUnitCell           = zUnitCellSize(pre->getYScale(), post->getYScale());
   int unitCellIndex        = layerIndexToUnitCellIndex(
         patchIndex,
         preLoc,
         nxUnitCell,
         nyUnitCell,
         kxUnitCellIndex,
         kyUnitCellIndex,
         kfUnitCellIndex);
   return unitCellIndex;
}

/**
 * Find the weight value that that is in the nth percentile
 */
SparseWeightInfo HyPerConn::findPercentileThreshold(
      float percentile,
      float **wDataStart,
      size_t numAxonalArborLists,
      size_t numPatches,
      size_t patchSize) const {
   pvAssert(percentile >= 0.0f);
   pvAssert(percentile <= 1.0f);

   size_t fullWeightSize = numAxonalArborLists * numPatches * patchSize;
   SparseWeightInfo info;
   info.percentile = percentile;

   if (percentile >= 1.0f) {
      info.size            = fullWeightSize;
      info.thresholdWeight = 0.0f;
      return info;
   }

   std::vector<float> weights;
   weights.reserve(fullWeightSize);

   for (int ar = 0; ar < numAxonalArborLists; ar++) {
      for (int pt = 0; pt < numPatches; pt++) {
         float *weight = &wDataStart[ar][pt * patchSize];
         for (int k = 0; k < patchSize; k++) {
            weights.push_back(fabs(weight[k]));
         }
      }
   }

   std::sort(weights.begin(), weights.end());
   int index = weights.size() * info.percentile;

   info.thresholdWeight = weights[index];
   info.size            = weights.size() - index;
   return info;
}

SparseWeightInfo HyPerConn::calculateSparseWeightInfo() const {
   size_t patchSize = nfp * nxp * nyp;
   return findPercentileThreshold(
         mWeightSparsity, wDataStart, numAxonalArborLists, numDataPatches, patchSize);
}

void HyPerConn::allocateSparseWeightsPre(PVLayerCube const *activity, int arbor) {
   mSparseWeightInfo = calculateSparseWeightInfo();

   std::map<const WeightType *const, int> sizes;

   for (int kPreExt = 0; kPreExt < activity->numItems; kPreExt++) {
      PVPatch *patch                          = getWeights(kPreExt, arbor);
      const int nk                            = patch->nx * fPatchSize();
      const int nyp                           = patch->ny;
      const WeightType *const weightDataStart = get_wData(arbor, kPreExt);

      for (int y = 0; y < nyp; y++) {
         const WeightType *const weightPtr = weightDataStart + y * yPatchStride();

         // Don't re-sparsify something that's already been put thru the sparsfication grinder
         bool shouldSparsify = false;

         // Find the weight pointers for this nk sized patch
         WeightMapType::iterator sparseWeightValuesNk = mSparseWeightValues.find(nk);
         IndexMapType::iterator sparseWeightIndexesNk = mSparseWeightIndices.find(nk);

         if (sparseWeightValuesNk == mSparseWeightValues.end()) {
            // Weight pointers don't exist for this sized nk. Allocate a map for this nk
            mSparseWeightValues.insert(make_pair(nk, WeightPtrMapType()));
            mSparseWeightIndices.insert(make_pair(nk, WeightIndexMapType()));
            // Get references
            sparseWeightValuesNk  = mSparseWeightValues.find(nk);
            sparseWeightIndexesNk = mSparseWeightIndices.find(nk);
            shouldSparsify        = true;
         }
         else if (
               sparseWeightValuesNk->second.find(weightPtr) == sparseWeightValuesNk->second.end()) {
            // This nk group exists, but no weight pointer.
            shouldSparsify = true;
         }

         if (shouldSparsify) {
            WeightListType sparseWeight;
            IndexListType idx;

            // Equivalent to inner loop accumulate
            for (int k = 0; k < nk; k++) {
               WeightType weight = weightPtr[k];
               if (std::abs(weight) >= mSparseWeightInfo.thresholdWeight) {
                  sparseWeight.push_back(weight);
                  idx.push_back(k);
               }
            }

            sparseWeightValuesNk->second.insert(make_pair(weightPtr, sparseWeight));
            sparseWeightIndexesNk->second.insert(make_pair(weightPtr, idx));
         }
      }

      mKPreExtWeightSparsified.insert(kPreExt);
   }

   mSparseWeightsAllocated[arbor] = true;
}

void HyPerConn::allocateSparseWeightsPost(PVLayerCube const *activity, int arbor) {
   mSparseWeightInfo           = calculateSparseWeightInfo();
   const PVLayerLoc *targetLoc = post->getLayerLoc();
   const PVHalo *targetHalo    = &targetLoc->halo;
   const int targetNx          = targetLoc->nx;
   const int targetNy          = targetLoc->ny;
   const int targetNf          = targetLoc->nf;

   for (int kTargetRes = 0; kTargetRes < post->getNumNeurons(); kTargetRes++) {
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
      // get source layer's patch y stride
      int syp        = postConn->yPatchStride();
      int yPatchSize = postConn->yPatchSize();
      // Iterate through y patch
      int nk          = postConn->xPatchSize() * postConn->fPatchSize();
      int kernelIndex = postConn->patchToDataLUT(kTargetExt);

      const WeightType *const weightDataStart = postConn->get_wDataHead(arbor, kernelIndex);

      for (int ky = 0; ky < yPatchSize; ky++) {
         const WeightType *const weightPtr = weightDataStart + ky * syp;

         // Don't re-sparsify something that's already been put thru the sparsfication grinder
         bool shouldSparsify = false;

         // Find the weight pointers for this nk sized patch
         // Find the weight pointers for this nk sized patch
         WeightMapType::iterator sparseWeightValuesNk = mSparseWeightValues.find(nk);
         IndexMapType::iterator sparseWeightIndexesNk = mSparseWeightIndices.find(nk);

         if (mSparseWeightValues.find(nk) == mSparseWeightValues.end()) {
            // Weight pointers don't exist for this sized nk. Allocate a map for this nk
            mSparseWeightValues.insert(make_pair(nk, WeightPtrMapType()));
            mSparseWeightIndices.insert(make_pair(nk, WeightIndexMapType()));
            // Get references
            sparseWeightValuesNk  = mSparseWeightValues.find(nk);
            sparseWeightIndexesNk = mSparseWeightIndices.find(nk);
            shouldSparsify        = true;
         }
         else if (
               sparseWeightValuesNk->second.find(weightPtr) == sparseWeightValuesNk->second.end()) {
            // This nk group exists, but no weight pointer.
            shouldSparsify = true;
         }

         if (shouldSparsify) {
            WeightListType sparseWeight;
            IndexListType idx;

            for (int k = 0; k < nk; k++) {
               WeightType weight = weightPtr[k];
               if (std::abs(weight) >= mSparseWeightInfo.thresholdWeight) {
                  sparseWeight.push_back(weight);
                  idx.push_back(k);
               }
            }

            sparseWeightValuesNk->second.insert(make_pair(weightPtr, sparseWeight));
            sparseWeightIndexesNk->second.insert(make_pair(weightPtr, idx));
         }
      }

      mKPreExtWeightSparsified.insert(kTargetRes);
   }

   mSparseWeightsAllocated[arbor] = true;
}

void HyPerConn::deliverOnePreNeuronActivitySparseWeights(
      int kPreExt,
      int arbor,
      float a,
      float *postBufferStart,
      void *auxPtr) {
   pvAssert(mSparseWeightsAllocated[arbor] == true);
   pvAssert(mKPreExtWeightSparsified.find(kPreExt) != mKPreExtWeightSparsified.end());

   PVPatch *patch        = getWeights(kPreExt, arbor);
   const int nk          = patch->nx * fPatchSize();
   const int nyp         = patch->ny;
   const int sy          = getPostNonextStrides()->sy; // stride in layer
   auto weightDataStart  = get_wData(arbor, kPreExt);
   float *postPatchStart = postBufferStart + getGSynPatchStart(kPreExt, arbor);
   int offset            = 0;

   for (int y = 0; y < nyp; y++) {
      WeightType *weightPtr = weightDataStart + y * yPatchStride();
      float *post           = postPatchStart + y * sy + offset;

      pvAssert(mSparseWeightValues.find(nk) != mSparseWeightValues.end());
      pvAssert(
            mSparseWeightValues.find(nk)->second.find(weightPtr)
            != mSparseWeightValues.find(nk)->second.end());

      const WeightListType &sparseWeights =
            mSparseWeightValues.find(nk)->second.find(weightPtr)->second;
      const IndexListType &idx = mSparseWeightIndices.find(nk)->second.find(weightPtr)->second;

      for (int k = 0; k < sparseWeights.size(); k++) {
         int outIdx = idx[k];
         post[outIdx] += a * sparseWeights[k];
      }
   }
}

void HyPerConn::deliverOnePostNeuronActivitySparseWeights(
      int arborID,
      int kTargetExt,
      int inSy,
      float *activityStartBuf,
      float *gSynPatchPos,
      float dtFactor,
      taus_uint4 *rngPtr) {
   // get source layer's patch y stride
   int syp        = postConn->yPatchStride();
   int yPatchSize = postConn->yPatchSize();
   // Iterate through y patch
   int nk          = postConn->xPatchSize() * postConn->fPatchSize();
   int kernelIndex = postConn->patchToDataLUT(kTargetExt);

   float *weightStartBuf = postConn->get_wDataHead(arborID, kernelIndex);
   int offset            = 0;
   for (int ky = 0; ky < yPatchSize; ky++) {
      float *activityY = &(activityStartBuf[ky * inSy + offset]);

      float *weightPtr = weightStartBuf + ky * syp;

      pvAssert(mSparseWeightValues.find(nk) != mSparseWeightValues.end());
      pvAssert(
            mSparseWeightValues.find(nk)->second.find(weightPtr)
            != mSparseWeightValues.find(nk)->second.end());

      const WeightListType &sparseWeight =
            mSparseWeightValues.find(nk)->second.find(weightPtr)->second;
      const IndexListType &idx = mSparseWeightIndices.find(nk)->second.find(weightPtr)->second;

      float dv = 0.0;
      for (int k = 0; k < sparseWeight.size(); k++) {
         dv += activityY[idx[k]] * sparseWeight[k];
      }
      *gSynPatchPos += dtFactor * dv;
   }
}

int HyPerConn::prepareCheckpointWrite() {
   blockingNormalize_dW();
   pvAssert(m_dWReduceRequests.empty());
   return PV_SUCCESS;
}

int HyPerConn::cleanup() {
   if (!m_dWReduceRequests.empty()) {
      wait_dWReduceRequests();
   }
   return PV_SUCCESS;
}

} // namespace PV
