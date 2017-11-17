/*
 * HyPerConn.cpp
 *
 *  Created on: Oct 21, 2008
 *      Author: Craig Rasmussen
 */

#include "HyPerConn.hpp"
#include "PlasticCloneConn.hpp"
#include "checkpointing/CheckpointEntryWeightPvp.hpp"
#include "columns/Factory.hpp"
#include "components/PostWeights.hpp"
#include "include/default_params.h"
#include "io/FileStream.hpp"
#include "io/PrintStream.hpp"
#include "io/WeightsFileIO.hpp"
#include "io/fileio.hpp"
#include "io/io.hpp"
#include "normalizers/NormalizeBase.hpp"
#include "privateTransposeConn.hpp"
#include "utils/TransposeWeights.hpp"
#include "utils/conversions.h"
#include "weightinit/InitWeights.hpp"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <float.h>
#include <iostream>
#include <limits.h>
#include <limits>
#include <stdlib.h>
#include <string.h>

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
   // connections can set to false if no warning is necessary.
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

   nxpPost                    = 0;
   nypPost                    = 0;
   nfpPost                    = 0;
   writeCompressedWeights     = false;
   writeCompressedCheckpoints = false;
   fileType = PVP_WGT_FILE_TYPE; // Subclass's initialize_base() gets called after HyPerConn's
   // initialize_base(), so this can be changed in subclasses.

   selfFlag =
         false; // specifies whether connection is from a layer to itself (i.e. a self-connection)
   combine_dW_with_W_flag      = false;
   normalizeMethod             = NULL;
   normalizer                  = NULL;
   plasticityFlag              = false;
   dWMax                       = std::numeric_limits<float>::quiet_NaN();
   strengthParamHasBeenWritten = false;

   updateGSynFromPostPerspective = false;
   thread_gSyn                   = NULL;

   pvpatchAccumulateTypeString = NULL;

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

   postToPreActivity = NULL;
   needFinalize      = true;

   lastUpdateTime        = 0.0;
   lastTimeUpdateCalled  = 0.0;
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
#ifdef PV_USE_CUDNN
   cudnn_WData = NULL;
#endif
#endif

   return PV_SUCCESS;
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

// set member variables specified by user
int HyPerConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   BaseConnection::ioParamsFillGroup(ioFlag);
   ioParam_sharedWeights(ioFlag);
   ioParam_weightInitType(ioFlag);
   if (weightInitializer != nullptr) {
      weightInitializer->ioParams(ioFlag, false, false);
   }
   ioParam_triggerLayerName(ioFlag);
   ioParam_triggerOffset(ioFlag);
   ioParam_weightUpdatePeriod(ioFlag);
   ioParam_initialWeightUpdateTime(ioFlag);
   ioParam_immediateWeightUpdate(ioFlag);
   if (ioFlag == PARAMS_IO_READ) {
      // Will be written by the HyPerDeliver component, but not all
      // delivery methods are componentized yet, so we still need to read here.
      ioParam_updateGSynFromPostPerspective(ioFlag);
      ioParam_pvpatchAccumulateType(ioFlag);
      ioParam_convertRateToSpikeCount(ioFlag);
   }
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

   // Weight sparsity
   ioParam_weightSparsity(ioFlag);
   return PV_SUCCESS;
}

void HyPerConn::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "receiveGpu"));
   parent->parameters()->ioParamValue(
         ioFlag, name, "sharedWeights", &sharedWeights, true /*default*/, true /*warn if absent*/);
   if (sharedWeights == false and receiveGpu == true) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: sharedWeights must be true in order to receive on the GPU.\n",
               getDescription_c());
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
       && parent->getCommunicator()->globalCommRank() == 0) {
      InfoLog().printf(
            "%s: nfp will be set in the communicateInitInfo() stage.\n", getDescription_c());
   }
}

void HyPerConn::ioParam_shrinkPatches(enum ParamsIOFlag ioFlag) {
   // shrinkPatches was marked obsolete Jul 25, 2017
   if (ioFlag == PARAMS_IO_READ and parent->parameters()->present(name, "shrinkPatches")) {
      WarnLog() << "shrinkPatches flag is no longer used.\n";
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

void HyPerConn::ioParam_convertRateToSpikeCount(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag,
         this->getName(),
         "convertRateToSpikeCount",
         &convertRateToSpikeCount,
         false /*default value*/);
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
   if (needPost) {
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
   setPatchStrides();
   allocateWeights();
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

void HyPerConn::allocateWeights() {
   setWeights(
         new Weights(
               name,
               nxp,
               nyp,
               nfp,
               pre->getLayerLoc(),
               post->getLayerLoc(),
               numAxonalArborLists,
               sharedWeights,
               0.0));
   addObserver(&mWeightsPair, BaseMessage{});
   mWeightsPair.mPreWeights->allocateDataStructures();
   if (plasticityFlag) {
      if (combine_dW_with_W_flag) {
         mDeltaWeights = mWeightsPair.mPreWeights;
      }
      else {
         setDeltaWeights(new Weights(name, mWeightsPair.mPreWeights));
      }
      if (!combine_dW_with_W_flag) {
         mDeltaWeights->allocateDataStructures();
      }
   }
   if (sharedWeights && normalizeDwFlag) {
      int const nPatches      = mWeightsPair.mPreWeights->getNumDataPatches();
      numKernelActivations    = (long **)pvCalloc(numAxonalArborLists, sizeof(long *));
      int const sp            = nxp * nyp * nfp;
      std::size_t numWeights  = (std::size_t)(sp) * (std::size_t)nPatches;
      numKernelActivations[0] = (long *)pvCalloc(numWeights, sizeof(long));
      for (int arborId = 0; arborId < numAxonalArborLists; arborId++) {
         numKernelActivations[arborId] = (numKernelActivations[0] + sp * nPatches * arborId);
      } // loop over arbors
   }
   if (needPost) {
      mWeightsPair.mPostWeights = new PostWeights(getName(), getPreWeights());
      mWeightsPair.mPostWeights->allocateDataStructures();
   }
}

void HyPerConn::initPatchToDataLUT() {
   pvAssert(patch2datalookuptable == NULL);
   if (sharedWeights) {
      int numGeometryPatches = getNumGeometryPatches();

      patch2datalookuptable = (int *)pvCalloc(numGeometryPatches, sizeof(int));

      for (int i = 0; i < numGeometryPatches; i++) {
         int kernelindex          = patchIndexToDataIndex(i);
         patch2datalookuptable[i] = kernelindex;
      }
   }
   else {
      // lookuptable just returns the patchindex
   }
}

void HyPerConn::createDeliveryObject() {
   BaseObject *baseObject =
         Factory::instance()->createByKeyword("HyPerDeliveryFacade", name, parent);
   HyPerDeliveryFacade *deliveryObject = dynamic_cast<HyPerDeliveryFacade *>(baseObject);
   pvAssert(deliveryObject);
   setDeliveryObject(deliveryObject);
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
            int numGeometryPatches = postConn->getNumGeometryPatches();
            d_Patch2DataLookupTable =
                  device->createBuffer(numGeometryPatches * sizeof(int), &description);
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

         int numGeometryPatches = getNumGeometryPatches();
         int patchSize          = numGeometryPatches * sizeof(Patch);
         d_Patches              = device->createBuffer(patchSize, &description);

         // Need a buffer for gsynpatch start for one arbor
         int gsynPatchStartIndexSize = numGeometryPatches * sizeof(size_t);
         d_GSynPatchStart            = device->createBuffer(gsynPatchStartIndexSize, &description);

         if (numberOfAxonalArborLists() == 1) {
            // Patches in mPreWeights were allocated as a single block of memory
            Patch const *h_patches        = &mWeightsPair.mPreWeights->getPatch(0);
            PVCuda::CudaBuffer *d_patches = getDevicePatches();
            pvAssert(d_patches);
            d_patches->copyToDevice(h_patches);
            size_t const *h_GSynPatchStart       = getGSynPatchStart();
            PVCuda::CudaBuffer *d_GSynPatchStart = getDeviceGSynPatchStart();
            pvAssert(d_GSynPatchStart);
            d_GSynPatchStart->copyToDevice(h_GSynPatchStart);
         }

         if (sharedWeights) {
            int numGeometryPatches = getNumGeometryPatches();
            d_Patch2DataLookupTable =
                  device->createBuffer(numGeometryPatches * sizeof(int), &description);
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

void HyPerConn::writeWeights(double timestamp) {
   writeWeights(timestamp, mWeightsPair.mPreWeights, writeCompressedWeights, mOutputStateStream);
}

void HyPerConn::writeWeights(
      double timestamp,
      Weights *weights,
      bool compressWeights,
      FileStream *fileStream) {
   WeightsFileIO weightsFileIO(fileStream, getMPIBlock(), weights);
   weightsFileIO.writeWeights(timestamp, compressWeights);
}

void HyPerConn::writeWeights(
      double timestamp,
      Weights *weights,
      bool compressWeights,
      std::string const &path,
      bool appendFlag,
      bool verifyWrites) {
   std::ios_base::openmode mode =
         appendFlag ? std::ios_base::out
                    : (std::ios_base::in | std::ios_base::out | std::ios_base::ate);
   FileStream fileStream(path.c_str(), mode, verifyWrites);
   WeightsFileIO weightsFileIO(&fileStream, getMPIBlock(), weights);
   weightsFileIO.writeWeights(timestamp, compressWeights);
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
      pv_text_write_patch(outStream, getPatch(k), getWeightsData(arbor, k), nfp, sxp, syp, sfp);
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
      Weights *weights) {
   bool registerSucceeded = checkpointer->registerCheckpointEntry(
         std::make_shared<CheckpointEntryWeightPvp>(
               getName(), bufferName, checkpointer->getMPIBlock(), weights, writeCompressedWeights),
         !plasticityFlag);
   FatalIf(
         !registerSucceeded,
         "%s failed to register %s for checkpointing.\n",
         getDescription_c(),
         bufferName);
}

int HyPerConn::registerData(Checkpointer *checkpointer) {
   int status = BaseConnection::registerData(checkpointer);
   checkpointWeightPvp(checkpointer, "W", getPreWeights());
   if (plasticityFlag and !mImmediateWeightUpdate) {
      checkpointWeightPvp(checkpointer, "dW", getDeltaWeights());
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
   if (weightInitializer) {
      weightInitializer->initializeWeights(mWeightsPair.mPreWeights);
   }
   return PV_SUCCESS;
}

int HyPerConn::outputState(double timef) {
   int status = 0;
   io_timer->start();

   for (int i = 0; i < numProbes; i++) {
      probes[i]->outputStateWrapper(timef, parent->getDeltaTime());
   }

   if ((writeStep >= 0) && (timef >= writeTime)) {
      writeTime += writeStep;

      writeWeights(timef);
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
         long *activations = getActivationsHead(kArbor, kKernel);
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
         float *dWeights = getDeltaWeightsDataHead(kArbor, kKernel);
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
   if (needPost) {
      pvAssert(getPostWeights() != nullptr);
      TransposeWeights::transpose(getPreWeights(), getPostWeights(), parent->getCommunicator());
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
            getActivations(arborID),
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
            getDeltaWeightsDataStart(arborID),
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
      float *dwArborStart      = getDeltaWeightsDataStart(arborID);
      size_t const patchSize   = (size_t)nxp * (size_t)nyp * (size_t)nfp;
      size_t const localSize   = (size_t)getNumDataPatches() * (size_t)patchSize;
      size_t const arborSize   = localSize * (size_t)numberOfAxonalArborLists();
      MPI_Comm const batchComm = parent->getCommunicator()->batchCommunicator();

      auto sz = m_dWReduceRequests.size();
      m_dWReduceRequests.resize(sz + 1);
      MPI_Iallreduce(
            MPI_IN_PLACE,
            getDeltaWeightsDataStart(arborID),
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

   Patch const *weights = getPatch(kExt);
   int ny               = weights->ny;
   int nk               = weights->nx * nfp;
   if (ny == 0 || nk == 0) {
      return PV_SUCCESS;
   }

   size_t offset           = getAPostOffset(kExt);
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

   float *dwdata     = getDeltaWeightsData(arborID, kExt);
   long *activations = NULL;
   if (sharedWeights && normalizeDwFlag) {
      activations = getActivations(arborID, kExt);
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
               // getDeltaWeightsData
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
            float *dwpatchdata = getDeltaWeightsDataHead(loop_arbor, kernelindex);
            long *activations  = getActivationsHead(loop_arbor, kernelindex);
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
      float *w_data_start = getWeightsDataStart(kArbor);
      for (long int k = 0; k < patchStartIndex(getNumDataPatches()); k++) {
         w_data_start[k] += getDeltaWeightsDataStart(kArbor)[k];
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

int HyPerConn::deliver() {
   Weights *deliveryWeights = updateGSynFromPostPerspective ? getPostWeights() : getPreWeights();
   pvAssert(deliveryWeights != nullptr);
   if (!getReceiveGpu()) {
      getDeliveryObject()->deliver(deliveryWeights);
      return PV_SUCCESS;
   }

#ifdef PV_USE_CUDA
   // Check if updating from post perspective
   HyPerLayer *pre = preSynapticLayer();
   int numArbors   = numberOfAxonalArborLists();

   for (int arbor = 0; arbor < numArbors; arbor++) {
      int status       = PV_SUCCESS;
      int delay        = getDelay(arbor);
      PVLayerCube cube = pre->getPublisher()->createCube(delay);
      cube.numItems /= cube.loc.nbatch;
      // hack; should make sure deliver*Perspective* methods expect numItems to include batching.
      if (!getUpdateGSynFromPostPerspective()) {
         status = deliverPresynapticPerspectiveGPU(&cube, arbor);
      }
      else {
         status = deliverPostsynapticPerspectiveGPU(&cube, arbor);
      }
      // No need to update GSyn since it's already living on gpu
      post->setUpdatedDeviceGSynFlag(false);
      pvAssert(status == PV_SUCCESS || status == PV_BREAK);
      if (status == PV_BREAK) {
         break; // Breaks out of arbor loop
      }
   }
#endif
   return PV_SUCCESS;
}

void HyPerConn::deliverUnitInput(float *recvBuffer) {
   auto weightsFacade = dynamic_cast<HyPerDeliveryFacade *>(getDeliveryObject());
   bool fromPost      = weightsFacade->getUpdateGSynFromPostPerspective();
   Weights *weights   = fromPost ? mWeightsPair.mPreWeights : mWeightsPair.mPostWeights;
   getDeliveryObject()->deliverUnitInput(weights, recvBuffer);
}

#ifdef PV_USE_CUDA
void HyPerConn::updateDeviceWeights() {
   // wDataStart is one big buffer, so this should grab everything
   float *h_weights              = getWeightsDataStart(0);
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
      // mPreWeights patches were allocated as a single block of memory
      Patch const *h_patches        = &mWeightsPair.mPreWeights->getPatch(0);
      PVCuda::CudaBuffer *d_patches = getDevicePatches();
      pvAssert(d_patches);

      d_patches->copyToDevice(h_patches);

      size_t const *h_GSynPatchStart       = getGSynPatchStart();
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

int HyPerConn::clearWeights(float *arborDataStart, int numPatches, int nxp, int nyp, int nfp) {
   for (long int w = 0; w < patchStartIndex(numPatches); w++) {
      arborDataStart[w] = 0.0f;
   }
   return PV_SUCCESS;
}

int HyPerConn::deleteWeights() {
   delete mWeightsPair.mPreWeights;
   if (!combine_dW_with_W_flag) {
      delete mDeltaWeights;
   }

   if (numKernelActivations != NULL) {
      free(numKernelActivations[0]);
      free(numKernelActivations);
   }

   delete mWeightsPair.mPostWeights;

   free(patch2datalookuptable);
   patch2datalookuptable = NULL;

   return PV_SUCCESS;
}

// convertPreSynapticWeights was marked obsolete Jul 27, 2017.
Patch ***HyPerConn::convertPreSynapticWeights(double simTime) {
   Fatal() << "convertPreSynapticWeights is obsolete.\n";
   return nullptr;
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
SparseWeightInfo HyPerConn::findPercentileThreshold(float percentile, Weights *weights) const {
   pvAssert(percentile >= 0.0f);
   pvAssert(percentile <= 1.0f);

   int numAxonalArborLists = weights->getNumArbors();
   int numPatches          = weights->getNumDataPatches();
   int patchSize = weights->getPatchSizeX() * weights->getPatchSizeY() * weights->getPatchSizeF();
   size_t fullWeightSize = numAxonalArborLists * numPatches * patchSize;
   SparseWeightInfo info;
   info.percentile = percentile;

   if (percentile >= 1.0f) {
      info.size            = fullWeightSize;
      info.thresholdWeight = 0.0f;
      return info;
   }

   std::vector<float> weightVector;
   weightVector.reserve(fullWeightSize);

   for (int arbor = 0; arbor < numAxonalArborLists; arbor++) {
      for (int patchIndex = 0; patchIndex < numPatches; patchIndex++) {
         float *weight = weights->getDataFromDataIndex(arbor, patchIndex);
         for (int k = 0; k < patchSize; k++) {
            weightVector.push_back(fabs(weight[k]));
         }
      }
   }

   std::sort(weightVector.begin(), weightVector.end());
   int index = weightVector.size() * info.percentile;

   info.thresholdWeight = weightVector[index];
   info.size            = weightVector.size() - index;
   return info;
}

SparseWeightInfo HyPerConn::calculateSparseWeightInfo() const {
   size_t patchSize = nfp * nxp * nyp;
   return findPercentileThreshold(mWeightSparsity, mWeightsPair.mPreWeights);
}

void HyPerConn::allocateSparseWeightsPre(PVLayerCube const *activity, int arbor) {
   mSparseWeightInfo = calculateSparseWeightInfo();

   std::map<const WeightType *const, int> sizes;

   for (int kPreExt = 0; kPreExt < activity->numItems; kPreExt++) {
      Patch const *patch                      = getPatch(kPreExt);
      const int nk                            = patch->nx * fPatchSize();
      const int nyp                           = patch->ny;
      const WeightType *const weightDataStart = getWeightsData(arbor, kPreExt);

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

      const WeightType *const weightDataStart = postConn->getWeightsDataHead(arbor, kernelIndex);

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
