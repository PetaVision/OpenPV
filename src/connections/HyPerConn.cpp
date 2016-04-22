/*
 * HyPerConn.cpp
 *
 *  Created on: Oct 21, 2008
 *      Author: Craig Rasmussen
 */

#include "HyPerConn.hpp"
#include "include/default_params.h"
#include "io/io.h"
#include "io/fileio.hpp"
#include "utils/conversions.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>

#include "layers/accumulate_functions.h"
#include "weightinit/InitWeights.hpp"
#include "normalizers/NormalizeBase.hpp"
#include "privateTransposeConn.hpp"
#include "PlasticCloneConn.hpp"
#include "io/CoreParamGroupHandler.hpp"
#include "columns/Factory.hpp"

#include "utils/PVLog.hpp"
#include "utils/PVAlloc.hpp"
#include "utils/PVAssert.hpp"

namespace PV {

HyPerConn::HyPerConn()
{
   initialize_base();
}

HyPerConn::HyPerConn(const char * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer) {
   initialize_base();
   initialize(name, hc, weightInitializer, weightNormalizer);
}

HyPerConn::~HyPerConn()
{
   delete io_timer;      io_timer     = NULL;
   delete update_timer;  update_timer = NULL;

   free(pvpatchAccumulateTypeString);

#if defined(PV_USE_CUDA)
   pvDelete(d_WData);
   pvDelete(d_Patches);
   pvDelete(d_GSynPatchStart);
   pvDelete(d_PostToPreActivity);
   pvDelete(d_Patch2DataLookupTable);
   pvDelete(krRecvPost);
   pvDelete(krRecvPre);

#if defined(PV_USE_CUDNN)
   pvDelete(cudnn_WData);
#endif

#endif // PV_USE_CUDA

   deleteWeights();

   // free the task information
   free(normalizeMethod);

   free(weightInitTypeString);

   pvDelete(weightInitializer);
   pvDelete(randState);

   if(postToPreActivity){
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

   free(normalizeGroupName);

   if(thread_gSyn){
      for(int i = 0; i < parent->getNumThreads(); i++){
         free(thread_gSyn[i]);
         thread_gSyn[i] = NULL;
      }
      free(thread_gSyn);
      thread_gSyn = NULL;
   }

   if(needPost && postConn){
      delete postConn;
   }

   if(batchSkip){
      free(batchSkip);
   }
}

//!
/*!
 *
 *
 *
 */
int HyPerConn::initialize_base()
{
   nxp = 1;
   nyp = 1;
   nfp = -1; // A negative value for nfp will be converted to postsynaptic layer's nf.
   warnDefaultNfp = true;  // Issue a warning if default value of nfp (post's nf) is used.  Derived layers can set to false if only one nfp is allowed (e.g. IdentConn)
   sxp = 1;
   syp = 1;
   sfp = 1;
   parent = NULL;
   ioAppend = false;

   weightInitTypeString = NULL;
   weightInitializer = NULL;

   io_timer     = NULL;
   update_timer = NULL;


   postConn = NULL;
   needPost = false;

   // wMin = 0.0;
   // wMax = 1.0;
   wPostTime = -1.0;
   wPostPatches = NULL;
   wPostDataStart = NULL;
   wPostPatchesp = NULL;
   wPostDataStartp = NULL;
   nxpPost = 0;
   nypPost = 0;
   nfpPost = 0;
   writeCompressedWeights = false;
   writeCompressedCheckpoints = false;
   fileType = PVP_WGT_FILE_TYPE; // Subclass's initialize_base() gets called after HyPerConn's initialize_base(), so this can be changed in subclasses.

   wDataStart = NULL;
   dwDataStart = NULL;
   wPatches=NULL;
   aPostOffset = NULL;
   gSynPatchStart = NULL;

   selfFlag = false;  // specifies whether connection is from a layer to itself (i.e. a self-connection)
   combine_dW_with_W_flag = false;
   normalizeMethod = NULL;
   normalizeGroupName = NULL;
   normalizer = NULL;
   plasticityFlag = false;
   shrinkPatches_flag = false; // default value, overridden by params file parameter "shrinkPatches" in readShrinkPatches()
   shrinkPatchesThresh = 0;
   normalizeArborsIndividually = true;
   normalize_max = false;
   normalize_zero_offset = false;
   normalize_cutoff = 0.0f;
   normalize_RMS_amp = false;
   dWMax            = std::numeric_limits<float>::quiet_NaN();
   strengthParamHasBeenWritten = false;

   updateGSynFromPostPerspective = false;
   thread_gSyn = NULL;

   pvpatchAccumulateTypeString = NULL;
   pvpatchAccumulateType = ACCUMULATE_CONVOLVE;

   initInfoCommunicatedFlag = false;
   dataStructuresAllocatedFlag = false;
   initialValuesSetFlag = false;

   randState = NULL;

   triggerFlag = false; //Default to update every timestamp
   triggerLayer = NULL;
   triggerLayerName = NULL;
   triggerOffset = 0;
   weightUpdatePeriod = 0;
   initialWeightUpdateTime = 0;
   weightUpdateTime = 0;

   clones.clear();

   postToPreActivity = NULL;
   needFinalize = true;
   needAllocPostWeights = true;

   lastUpdateTime = 0.f;
   symmetrizeWeightsFlag = false;
   patch2datalookuptable = NULL;
   numKernelActivations = NULL;
   keepKernelsSynchronized_flag = false;

   normalizeDwFlag = true;
   useMask = false;
   maskLayerName = NULL;
   maskFeatureIdx = -1;
   mask = NULL;

   batchSkip = NULL;

#if defined(PV_USE_CUDA)
   receiveGpu = false;
   allocDeviceWeights = false;
   allocPostDeviceWeights = false;
   d_WData = NULL;
   d_Patches = NULL;
   d_GSynPatchStart = NULL;
   d_PostToPreActivity = NULL;
   d_Patch2DataLookupTable = NULL;
   krRecvPost = NULL;
   krRecvPre = NULL;
   //updatedDeviceWeights = true; //Start off as always updated
   numXLocal= 1;
   numYLocal= 1;
   numFLocal = 1;
   preDataLocal = true;
   gpuGroupIdx = -1;
#if defined(PV_USE_CUDNN)
   cudnn_WData = NULL;
#endif
#endif

   return PV_SUCCESS;
}

int HyPerConn::createArbors() {
   const int shrinkNum = (shrinkPatches_flag ? numAxonalArborLists : 1) * preSynapticLayer()->getNumExtended();

   wPatches = (PVPatch***) pvCalloc(numAxonalArborLists, sizeof(PVPatch**));
   // GTK:  gSynPatchStart is offset from beginning of gSyn buffer for the corresponding channel
   gSynPatchStart = (size_t **) pvCalloc( numAxonalArborLists, sizeof(size_t *) );

   size_t * gSynPatchStartBuffer = (size_t *) pvCalloc(shrinkNum, sizeof(size_t));

   for (int k = 0; k < numAxonalArborLists; k++) {
      gSynPatchStart[k] = gSynPatchStartBuffer
            + shrinkPatches_flag * k * preSynapticLayer()->getNumExtended();
   }

   aPostOffset = (size_t **) pvCalloc(numAxonalArborLists, sizeof(size_t *));

   size_t * aPostOffsetBuffer = (size_t *) pvCalloc(shrinkNum, sizeof(size_t));

   for( int k=0; k<numAxonalArborLists; k++ ) {
      aPostOffset[k] = aPostOffsetBuffer
            + shrinkPatches_flag * k * preSynapticLayer()->getNumExtended();
   }

   wDataStart = (pvwdata_t **) pvCalloc(numAxonalArborLists, sizeof(pvwdata_t *));

   dwDataStart = (pvwdata_t **) pvCalloc(numAxonalArborLists, sizeof(pvwdata_t *));

   if(sharedWeights && normalizeDwFlag) {
      numKernelActivations = (long **) pvCalloc(numAxonalArborLists, sizeof(long * ));
   }

   return PV_SUCCESS;
}

void HyPerConn::createArborsOutOfMemory() {
   connOutOfMemory("HyPerConn::createArbors()");
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
int HyPerConn::constructWeights()
{
   int sx = nfp;
   int sy = sx * nxp;
   int sp = sy * nyp;
   int nPatches = getNumDataPatches();
   int status = PV_SUCCESS;

   //pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "shrinkPatches"));
   
   // createArbors() uses the value of shrinkPatches.  It should have already been read in ioParamsFillGroup.
   //allocate the arbor arrays:
   createArbors();

   setPatchStrides();

   ////allocate weight patches and axonal arbors for each arbor
   ////Allocate all the weights
   wDataStart[0] = allocWeights(nPatches, nxp, nyp, nfp);
   pvAssert(get_wDataStart(0) != NULL);

   for (int arborId=0;arborId<numAxonalArborLists;arborId++) {
      status = createWeights(wPatches, arborId);
      pvAssert(wPatches[arborId] != NULL);

      if (arborId > 0){  // wDataStart already allocated
         wDataStart[arborId] = (get_wDataStart(0) + sp * nPatches * arborId);
         pvAssert(wDataStart[arborId] != NULL);
      }
      if (shrinkPatches_flag || arborId == 0){
         status |= adjustAxonalArbors(arborId);
      }
   }  // arborId

   //call to initializeWeights moved to BaseConnection::initializeState()
   status |= initPlasticityPatches();
   pvAssert(status == 0);
   if (shrinkPatches_flag) {
      for (int arborId=0;arborId<numAxonalArborLists;arborId++) {
         shrinkPatches(arborId);
      }
   }

   return status;
}

int HyPerConn::shrinkPatches(int arborId) {
   int numPatches = getNumWeightPatches();
   for (int kex = 0; kex < numPatches; kex++) {
      shrinkPatch(kex, arborId /* arbor */ );
   } // loop over pre-synaptic neurons

   return 0;
}

int HyPerConn::shrinkPatch(int kExt, int arborId) {

   int kIndex = patchToDataLUT(kExt);

   PVPatch *weights = getWeights(kExt,arborId);

   pvwdata_t * w = &get_wDataStart(arborId)[patchStartIndex(kIndex)+weights->offset];

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
            if(fabs(w[x * sxp + y * syp + f * sfp]) <= shrinkPatchesThresh) {
               nonZeroWeightFound=true;
               maxnx = maxnx < x ? x : maxnx;
               minnx = minnx > x ? x : minnx;
               maxny = maxny < y ? y : maxny;
               minny = minny > y ? y : minny;
            }
         }
      }
   }
   
   if(nonZeroWeightFound) {
      //Plus one to capture all of the patch
      int nxNew = maxnx+1 - minnx;
      int nyNew = maxny+1 - minny;
      int dxNew = minnx;
      int dyNew = minny;

      // adjust patch size (shrink) to fit within interior of post-synaptic layer
      //
      pvpatch_adjust(weights, sxp, syp, nxNew, nyNew, dxNew, dyNew);

      gSynPatchStart[arborId][kExt] += dxNew*getPostNonextStrides()->sx + dyNew*getPostNonextStrides()->sy;
      aPostOffset[arborId][kExt] += dxNew*getPostExtStrides()->sx + dyNew*getPostExtStrides()->sy; // Someone who uses these routines, please check that this is correct.
   }
   return 0;
}


int HyPerConn::initialize(const char * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer) {
   // It is okay for either of weightInitializer or weightNormalizer to be null at this point, either because we're in a subclass that doesn't need it, or because we are allowing for
   // backward compatibility.
   // The two lines needs to be before the call to BaseConnection::initialize, because that function calls ioParamsFillGroup,
   // which will call ioParam_weightInitType and ioParam_normalizeMethod, which for reasons of backward compatibility
   // will create the initializer and normalizer if those member variables are null.
   this->weightInitializer = weightInitializer;
   normalizer = weightNormalizer;

   int status = BaseConnection::initialize(name, hc); // BaseConnection should *NOT* take weightInitializer or weightNormalizer as arguments, as it does not know about InitWeights or NormalizeBase

   pvAssert(parent);
   PVParams * inputParams = parent->parameters();

   //set accumulateFunctionPointer
   pvAssert(!inputParams->presentAndNotBeenRead(name, "pvpatchAccumulateType"));
   switch (pvpatchAccumulateType) {
   case ACCUMULATE_CONVOLVE:
      accumulateFunctionPointer  = &pvpatch_accumulate;
      accumulateFunctionFromPostPointer = &pvpatch_accumulate_from_post;
      break;
   case ACCUMULATE_STOCHASTIC:
      accumulateFunctionPointer = &pvpatch_accumulate_stochastic;
      accumulateFunctionFromPostPointer = &pvpatch_accumulate_stochastic_from_post;
      break;
   case ACCUMULATE_MAXPOOLING:
      exitFailure("ACCUMULATE_MAXPOOLING not allowed in HyPerConn, use PoolingConn instead");
      //accumulateFunctionPointer = &pvpatch_max_pooling;
      //accumulateFunctionFromPostPointer = &pvpatch_max_pooling_from_post;
      break;
   case ACCUMULATE_SUMPOOLING:
      exitFailure("ACCUMULATE_SUMPOOLING not allowed in HyPerConn, use PoolingConn instead");
      //accumulateFunctionPointer = &pvpatch_sum_pooling;
      //accumulateFunctionFromPostPointer = &pvpatch_accumulate_from_post;
      break;
   case ACCUMULATE_AVGPOOLING:
      exitFailure("ACCUMULATE_AVGPOOLING not allowed in HyPerConn, use PoolingConn instead");
      //accumulateFunctionPointer = &pvpatch_sum_pooling;
      //accumulateFunctionFromPostPointer = &pvpatch_accumulate_from_post;
      break;
   default:
      pvAssert(0);
      break;
   }

   ioAppend = parent->getCheckpointReadFlag();
   io_timer     = new Timer(getName(), "conn", "io     ");
   update_timer = new Timer(getName(), "conn", "update ");

   _sparseWeightsAllocated.resize(numAxonalArborLists);
   std::fill(_sparseWeightsAllocated.begin(), _sparseWeightsAllocated.end(), false);

   return status;
}

int HyPerConn::setWeightInitializer() {
   weightInitializer = createInitWeightsObject(weightInitTypeString);
   if( weightInitializer == NULL) {
      weightInitializer = getDefaultInitWeightsMethod(getKeyword());
   }
   return weightInitializer==NULL ? PV_FAILURE : PV_SUCCESS;
}

/*
 * This method parses the weightInitType parameter and creates an
 * appropriate InitWeight object for the chosen weight initialization.
 * The preferred method is now (April 19, 2016) to have the create-function
 * (createHyPerConn, etc.) call getWeightInitializer() using the connection's
 * name and parent HyPerCol, and then pass the weight initializer in the
 * constructor.
 */
InitWeights * HyPerConn::createInitWeightsObject(const char * weightInitTypeStr) {
   pvAssert(weightInitializer == NULL);
   PV_Init * initObj = parent->getPV_InitObj();
   BaseObject * baseObject = initObj->create(weightInitTypeStr, name, parent);
   weightInitializer = dynamic_cast<InitWeights*>(baseObject);
   return weightInitializer;
}

int HyPerConn::setPreLayerName(const char * pre_name) {
   pvAssert(parent != NULL);
   pvAssert(preLayerName == NULL);

   if (pre_name != NULL) {
      preLayerName = strdup(pre_name);
      pvAssertMessage(preLayerName != NULL, "Connection \"%s\" error in rank %d process: unable to allocate memory for name of presynaptic layer \"%s\": %s", name, parent->columnId(), pre_name);
   }
   return PV_SUCCESS;
}

int HyPerConn::setPostLayerName(const char * post_name) {
   pvAssert(postLayerName==NULL);
   if (post_name != NULL) {
      postLayerName = strdup(post_name);
      pvAssertMessage(postLayerName != NULL, "Connection \"%s\" error in rank %d process: unable to allocate memory for name of postsynaptic layer \"%s\": %s", name, parent->columnId(), post_name, strerror(errno));
   }
   return PV_SUCCESS;
}

int HyPerConn::initNumWeightPatches() {
   numWeightPatches = pre->getNumExtended();
   return PV_SUCCESS;
}

int HyPerConn::initNumDataPatches() {
   if (sharedWeights) {
      int nxKernel = (pre->getXScale() < post->getXScale()) ? (int)pow(2,
            post->getXScale() - pre->getXScale()) : 1;
      int nyKernel = (pre->getYScale() < post->getYScale()) ? (int)pow(2,
            post->getYScale() - pre->getYScale()) : 1;
      numDataPatches = pre->getLayerLoc()->nf * nxKernel * nyKernel;
      return PV_SUCCESS;
   }
   else {
      numDataPatches = getNumWeightPatches();
   }
   return PV_SUCCESS;
}

int HyPerConn::initPlasticityPatches()
{
   if (!plasticityFlag) return PV_SUCCESS;
   int sx = nfp;
   int sy = sx * nxp;
   int sp = sy * nyp;
   int nPatches = getNumDataPatches();

   const int numAxons = numberOfAxonalArborLists();

   if (combine_dW_with_W_flag){
      dwDataStart = wDataStart;
      return PV_SUCCESS;
   }
   dwDataStart[0] = allocWeights(nPatches, nxp, nyp, nfp);
   pvAssert(get_dwDataStart(0) != NULL);
   for (int arborId = 0; arborId < numAxons; arborId++) {
      dwDataStart[arborId] = (dwDataStart[0] + sp * nPatches * arborId);
      pvAssert(get_dwDataStart(arborId) != NULL);
   } // loop over arbors

   if(sharedWeights && normalizeDwFlag){
      numKernelActivations[0] = (long*) pvCalloc(nxp*nyp*nfp*nPatches, sizeof(long));
      for (int arborId = 0; arborId < numAxons; arborId++) {
         numKernelActivations[arborId] = (numKernelActivations[0] + sp * nPatches * arborId);
         pvAssert(get_dwDataStart(arborId) != NULL);
      } // loop over arbors
   }

   return PV_SUCCESS;
}

// set member variables specified by user
int HyPerConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag)
{
   BaseConnection::ioParamsFillGroup(ioFlag);
   ioParam_sharedWeights(ioFlag);
   ioParam_weightInitType(ioFlag);
   if (weightInitializer != NULL) {
      weightInitializer->ioParamsFillGroup(ioFlag);
   }
   ioParam_initializeFromCheckpointFlag(ioFlag);
   ioParam_triggerLayerName(ioFlag);
   ioParam_triggerFlag(ioFlag);
   ioParam_triggerOffset(ioFlag);
   ioParam_weightUpdatePeriod(ioFlag);
   ioParam_initialWeightUpdateTime(ioFlag);
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
   ioParam_nxpShrunken(ioFlag);
   ioParam_nypShrunken(ioFlag);
   ioParam_nfp(ioFlag);
   ioParam_shrinkPatches(ioFlag);
   ioParam_normalizeMethod(ioFlag);
   if (normalizer != NULL && !strcmp(normalizer->getName(), getName())) {
      normalizer->ioParamsFillGroup(ioFlag);
   }
   ioParam_normalizeGroupName(ioFlag);
   ioParam_dWMax(ioFlag);
   ioParam_keepKernelsSynchronized(ioFlag);

   ioParam_normalizeDw(ioFlag);
   ioParam_useMask(ioFlag);
   ioParam_maskLayerName(ioFlag);
   ioParam_maskFeatureIdx(ioFlag);

   // GPU-specific parameters.  If not using GPUs, we read them anyway, with warnIfAbsent set to false, to prevent unnecessary warnings from unread or missing parameters.
   ioParam_gpuGroupIdx(ioFlag);
   // ioParam_receiveGpu(ioFlag); // read by parent class BaseConnection
   ioParam_preDataLocal(ioFlag);
   //Only read numX, Y, and F local if not using CUDNN
   ioParam_numXLocal(ioFlag);
   ioParam_numYLocal(ioFlag);
   ioParam_numFLocal(ioFlag);
   // Weight sparsity
   ioParam_weightSparsity(ioFlag);
   return PV_SUCCESS;
}

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)

void HyPerConn::ioParam_gpuGroupIdx(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "receiveGpu"));
   if(receiveGpu){
      parent->ioParamValue(ioFlag, name, "gpuGroupIdx", &gpuGroupIdx, gpuGroupIdx/*default*/, false/*warn if absent*/);
   }
}

void HyPerConn::ioParam_preDataLocal(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "receiveGpu"));
#ifndef PV_USE_CUDNN
   if(receiveGpu){
      parent->ioParamValue(ioFlag, name, "preDataLocal", &preDataLocal, true/*default*/, false/*warn if absent*/);
   }
#endif
}

void HyPerConn::ioParam_numXLocal(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "receiveGpu"));
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "updateGSynFromPostPerspective"));
   if(receiveGpu){
      //If we're using cudnn and updating from post, we don't need numX, Y, and F local
#ifdef PV_USE_CUDNN
      if(!updateGSynFromPostPerspective){
         parent->ioParamValue(ioFlag, name, "numXLocal", &numXLocal, 1, true);
      }
#else
      parent->ioParamValue(ioFlag, name, "numXLocal", &numXLocal, 1, true);
#endif
   }
}

void HyPerConn::ioParam_numYLocal(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "receiveGpu"));
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "updateGSynFromPostPerspective"));
   if(receiveGpu){
#ifdef PV_USE_CUDNN
      if(!updateGSynFromPostPerspective){
         parent->ioParamValue(ioFlag, name, "numYLocal", &numYLocal, 1, true);
      }
#else
      parent->ioParamValue(ioFlag, name, "numYLocal", &numYLocal, 1, true);
#endif
   }
}

void HyPerConn::ioParam_numFLocal(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "receiveGpu"));
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "updateGSynFromPostPerspective"));
   if(receiveGpu){
#ifdef PV_USE_CUDNN
      if(!updateGSynFromPostPerspective){
         parent->ioParamValue(ioFlag, name, "numFLocal", &numFLocal, 1, true);
      }
#else
      parent->ioParamValue(ioFlag, name, "numFLocal", &numFLocal, 1, true);
#endif
   }
}
#else // defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
// Some dummy ioParam_ functions to read GPU-specific params, without generating a warning if
// they are present, or generating a warning if they are absent.
void HyPerConn::ioParam_gpuGroupIdx(enum ParamsIOFlag ioFlag) {
   int dummyVar = 0;
   parent->ioParamValue(ioFlag, name, "gpuGroupIdx", &dummyVar, dummyVar/*default*/, false/*warnIfAbsent*/);
}
void HyPerConn::ioParam_preDataLocal(enum ParamsIOFlag ioFlag) {
   bool dummyFlag = false;
   parent->ioParamValue(ioFlag, name, "preDataLocal", &dummyFlag, dummyFlag/*default*/, false/*warnIfAbsent*/);
}
void HyPerConn::ioParam_numXLocal(enum ParamsIOFlag ioFlag) {
   int dummyVar = 0;
   parent->ioParamValue(ioFlag, name, "numXLocal", &dummyVar, dummyVar/*default*/, false/*warnIfAbsent*/);
}
void HyPerConn::ioParam_numYLocal(enum ParamsIOFlag ioFlag) {
   int dummyVar = 0;
   parent->ioParamValue(ioFlag, name, "numYLocal", &dummyVar, dummyVar/*default*/, false/*warnIfAbsent*/);
}
void HyPerConn::ioParam_numFLocal(enum ParamsIOFlag ioFlag) {
   int dummyVar = 0;
   parent->ioParamValue(ioFlag, name, "numFLocal", &dummyVar, dummyVar/*default*/, false/*warnIfAbsent*/);
}
#endif // defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)

void HyPerConn::ioParam_channelCode(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) {
      int ch = 0;
      parent->ioParamValueRequired(ioFlag, name, "channelCode", &ch);
      int status = decodeChannel(ch, &channel);
      if (status != PV_SUCCESS) {
         if (parent->columnId()==0) {
            logError("%s \"%s\": channelCode %d is not a valid channel.\n", getKeyword(), name,  ch);
         }
         MPI_Barrier(parent->icCommunicator()->communicator());
         pvAssert(0);
      }
   }
   else if (ioFlag==PARAMS_IO_WRITE) {
      int ch = (int) channel;
      parent->ioParamValueRequired(ioFlag, name, "channelCode", &ch);
   }
   else {
      exitFailure("All possibilities of ioFlag are covered above.");
   }
}

void HyPerConn::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "sharedWeights", &sharedWeights, true/*default*/, true/*warn if absent*/);
}

void HyPerConn::ioParam_weightInitType(enum ParamsIOFlag ioFlag) {
   parent->ioParamString(ioFlag, name, "weightInitType", &weightInitTypeString, NULL, true/*warnIfAbsent*/);
   if (ioFlag==PARAMS_IO_READ && weightInitializer==NULL) {
      int status = setWeightInitializer();
      pvAssertMessage(status == PV_SUCCESS, "%s \"%s\": Rank %d process unable to construct weightInitializer", getKeyword(), name, parent->columnId());
   }
}

void HyPerConn::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag) {
      parent->ioParamString(ioFlag, name, "triggerLayerName", &triggerLayerName, NULL, false/*warnIfAbsent*/);
      if (ioFlag==PARAMS_IO_READ) {
         triggerFlag = (triggerLayerName!=NULL && triggerLayerName[0]!='\0');
      }
   }
}

// triggerFlag was deprecated Aug 17, 2015.
// Setting triggerLayerName to a nonempty string has the effect of triggerFlag=true, and
// setting triggerLayerName to NULL or "" has the effect of triggerFlag=false.
// While triggerFlag is being deprecated, it is an error for triggerFlag to be false
// and triggerLayerName to be a nonempty string.
void HyPerConn::ioParam_triggerFlag(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag) {
      pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
      if (ioFlag == PARAMS_IO_READ && parent->parameters()->present(name, "triggerFlag")) {
         if (parent->columnId()==0) {
            logError("Connection \"%s\" warning: triggerFlag has been deprecated.", name);
         }
         bool flagFromParams = false;
         parent->ioParamValue(ioFlag, name, "triggerFlag", &flagFromParams, flagFromParams);
         if (flagFromParams != triggerFlag) {
            if (parent->columnId()==0) {
               logError("Connection \"%s\" Error: triggerLayerName=", name);
               if (triggerLayerName) { logError("\"%s\"", triggerLayerName); }
               else { logError("NULL"); }
               logError(" implies triggerFlag=%s but triggerFlag was set in params to %s",
                     triggerFlag ? "true" : "false", flagFromParams ? "true" : "false");
            }
            MPI_Barrier(parent->icCommunicator()->communicator());
            pvAssert(0);
         }
         else {
            if (parent->columnId()==0) {
               logError("   If triggerLayerName is a nonempty string, triggering will be on;");
               logError("   if triggerLayerName is empty or null, triggering will be off.");
            }
         }
      }
   }
}

void HyPerConn::ioParam_triggerOffset(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag) {
      pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
      if (triggerFlag) {
         parent->ioParamValue(ioFlag, name, "triggerOffset", &triggerOffset, triggerOffset);
         if(triggerOffset < 0){
            logError("%s \"%s\" error in rank %d process: TriggerOffset (%f) must be positive", getKeyword(), name, parent->columnId(), triggerOffset);
            pvAssert(0);
         }
      }
   }
}

void HyPerConn::ioParam_weightUpdatePeriod(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag) {
	   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
	   if (!triggerLayerName) {
	      parent->ioParamValue(ioFlag, name, "weightUpdatePeriod", &weightUpdatePeriod, parent->getDeltaTime());
	   }
   }
}

void HyPerConn::ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag) {
      pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
      initialWeightUpdateTime = parent->getStartTime();
      if (!triggerLayerName) {
         parent->ioParamValue(ioFlag, name, "initialWeightUpdateTime", &initialWeightUpdateTime, initialWeightUpdateTime, true/*warnIfAbsent*/);
      }
   }
   if (ioFlag==PARAMS_IO_READ) {
      weightUpdateTime=initialWeightUpdateTime;
   }
}

void HyPerConn::ioParam_pvpatchAccumulateType(enum ParamsIOFlag ioFlag) {
   PVParams * params = parent->parameters();
   // stochasticReleaseFlag deprecated on Aug 22, 2013, and declared obsolete Apr 10, 2015.
   if (ioFlag==PARAMS_IO_READ && params->present(name, "stochasticReleaseFlag")) {
      if (parent->columnId()==0) {
         logError("%s \"%s\" error: parameter stochasticReleaseFlag is obsolete.  Instead, set pvpatchAccumulateType to either \"convolve\" (the default) or \"stochastic\".", getKeyword(), name);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exitFailure("");
   }

   parent->ioParamString(ioFlag, name, "pvpatchAccumulateType", &pvpatchAccumulateTypeString, "convolve");
   if (ioFlag==PARAMS_IO_READ) {
      if (pvpatchAccumulateTypeString==NULL) {
         unsetAccumulateType();
         return;
      }
      // Convert string to lowercase so that capitalization doesn't matter.
      for (char * c = pvpatchAccumulateTypeString; *c!='\0'; c++) {
         *c = (char) tolower((int) *c);
      }

      if (strcmp(pvpatchAccumulateTypeString,"convolve")==0) {
         pvpatchAccumulateType = ACCUMULATE_CONVOLVE;
      }
      else if (strcmp(pvpatchAccumulateTypeString,"stochastic")==0) {
         pvpatchAccumulateType = ACCUMULATE_STOCHASTIC;
      }
      //IOParam for different pooling types is still handled here, TODO, move this to PoolingConn
      else if ((strcmp(pvpatchAccumulateTypeString,"maxpooling")==0) ||
	       (strcmp(pvpatchAccumulateTypeString,"max_pooling")==0) ||
	       (strcmp(pvpatchAccumulateTypeString,"max pooling")==0)) {
         pvpatchAccumulateType = ACCUMULATE_MAXPOOLING;
      }
      else if ((strcmp(pvpatchAccumulateTypeString,"sumpooling")==0) ||
	       (strcmp(pvpatchAccumulateTypeString,"sum_pooling")==0)  ||
	       (strcmp(pvpatchAccumulateTypeString,"sum pooling")==0)) {
         pvpatchAccumulateType = ACCUMULATE_SUMPOOLING;
      }
      else if ((strcmp(pvpatchAccumulateTypeString,"avgpooling")==0) ||
	       (strcmp(pvpatchAccumulateTypeString,"avg_pooling")==0)  ||
	       (strcmp(pvpatchAccumulateTypeString,"avg pooling")==0)) {
         pvpatchAccumulateType = ACCUMULATE_AVGPOOLING;
      }
      else {
         unsetAccumulateType();
      }
   }
}

void HyPerConn::unsetAccumulateType() {
   if (parent->columnId()==0) {
      if (pvpatchAccumulateTypeString) {
         logError("%s \"%s\" error: pvpatchAccumulateType \"%s\" is unrecognized.",
               getKeyword(), name, pvpatchAccumulateTypeString);
      }
      else {
         logError("%s \"%s\" error: pvpatchAccumulateType NULL is unrecognized.",
               getKeyword(), name);
      }
      logError("  Allowed values are \"convolve\", \"stochastic\", or \"maxpooling\".");
   }
   MPI_Barrier(parent->icCommunicator()->communicator());
   exitFailure("");
}


void HyPerConn::ioParam_writeStep(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "writeStep", &writeStep, parent->getDeltaTime());
}

void HyPerConn::ioParam_initialWriteTime(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "writeStep"));
   if (writeStep>=0) {
      double start_time = parent->getStartTime();
      parent->ioParamValue(ioFlag, name, "initialWriteTime", &initialWriteTime, start_time);
      if (ioFlag == PARAMS_IO_READ) {
         if (writeStep>0 && initialWriteTime < start_time) {
            if (parent->columnId()==0) {
               logInfo("%s \"%s\": initialWriteTime %f earlier than starting time %f.  Adjusting initialWriteTime:\n",
                     getKeyword(), name, initialWriteTime, start_time);
               fflush(stdout);
            }
            while (initialWriteTime < start_time) {
               initialWriteTime += writeStep;
            }
            if (parent->columnId()==0) {
               logInfo("%s \"%s\": initialWriteTime adjusted to %f\n",
                     getKeyword(), name, initialWriteTime);
            }
         }
         writeTime = initialWriteTime;
      }
   }
}

void HyPerConn::ioParam_writeCompressedWeights(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "writeStep"));
   if (writeStep>=0) {
      parent->ioParamValue(ioFlag, name, "writeCompressedWeights", &writeCompressedWeights, writeCompressedWeights, /*warnifabsent*/true);
   }
}

void HyPerConn::ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag) {
   if (parent->getCheckpointWriteFlag() || !parent->getSuppressLastOutputFlag()) {
      parent->ioParamValue(ioFlag, name, "writeCompressedCheckpoints", &writeCompressedCheckpoints, writeCompressedCheckpoints, /*warnifabsent*/true);
   }
}

void HyPerConn::ioParam_selfFlag(enum ParamsIOFlag ioFlag) {
   // selfFlag indicates whether pre and post layers refer to the same neurons.
   // The default value for selfFlag should be pre==post, but at the time ioParams(PARAMS_IO_READ) is called,
   // pre and post have not been set.  So we read the value with no warning if it's present;
   // if it's absent, set the value to pre==post in the communicateInitInfo stage and issue
   // the using-default-value warning then.
   parent->ioParamValue(ioFlag, name, "selfFlag", &selfFlag, selfFlag, false/*warnIfAbsent*/);
}

void HyPerConn::ioParam_combine_dW_with_W_flag(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag){
      parent->ioParamValue(ioFlag, name, "combine_dW_with_W_flag", &combine_dW_with_W_flag, combine_dW_with_W_flag, true/*warnIfAbsent*/);
   }

}

void HyPerConn::ioParam_nxp(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "nxp", &nxp, 1);
}

void HyPerConn::ioParam_nyp(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "nyp", &nyp, 1);
}

// nxpShrunken and nypShrunken were deprecated Feb 2, 2015
void HyPerConn::ioParam_nxpShrunken(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "nxp"));
   if (ioFlag==PARAMS_IO_READ) {
      if (parent->parameters()->present(name, "nxpShrunken")) {
         int nxpShrunken;
         parent->ioParamValue(ioFlag, name, "nxpShrunken", &nxpShrunken, nxp);
         if (nxpShrunken <= nxp) {
            nxp = nxpShrunken;
            if (parent->columnId()==0) {
               logError("%s \"%s\" warning: nxpShrunken is deprecated, as nxp can now take any of the values nxpShrunken could take before.  nxp will be set to %d and nxpShrunken will not be used.",
                     getKeyword(), name, nxp);
            }
         }
         else {
            if (parent->columnId()==0) {
               logError("%s \"%s\" warning: nxpShrunken is deprecated.  Instead, nxp can take any of the values nxpShrunken could take before.",
                     getKeyword(), name);
               logError("However, setting nxp to %d and nxpShrunken to the larger value %d is probably not what you meant.  Exiting.", nxp, nxpShrunken);
            }
            MPI_Barrier(parent->icCommunicator()->communicator());
         }
      }
   }
}

void HyPerConn::ioParam_nypShrunken(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "nyp"));
   if (ioFlag==PARAMS_IO_READ) {
      if (parent->parameters()->present(name, "nypShrunken")) {
         int nypShrunken;
         parent->ioParamValue(ioFlag, name, "nypShrunken", &nypShrunken, nyp);
         if (nypShrunken <= nyp) {
            nyp = nypShrunken;
            if (parent->columnId()==0) {
               logError("%s \"%s\" warning: nypShrunken is deprecated, as nyp can now take any of the values nypShrunken could take before.  nyp will be set to %d and nypShrunken will not be used.",
                     getKeyword(), name, nyp);
            }
         }
         else {
            if (parent->columnId()==0) {
               logError("%s \"%s\" warning: nypShrunken is deprecated.  Instead, nyp can take any of the values nypShrunken could take before.",
                     getKeyword(), name);
               logError("However, setting nyp to %d and nypShrunken to the larger value %d is probably not what you meant.  Exiting.", nyp, nypShrunken);
            }
            MPI_Barrier(parent->icCommunicator()->communicator());
         }
      }
   }
}

void HyPerConn::ioParam_nfp(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "nfp", &nfp, -1, false);
   if (ioFlag==PARAMS_IO_READ && nfp==-1 && !parent->parameters()->present(name, "nfp") && parent->columnId()==0) {
      logInfo("%s \"%s\": nfp will be set in the communicateInitInfo() stage.\n",
            getKeyword(), name);
   }
}

void HyPerConn::ioParam_shrinkPatches(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "shrinkPatches", &shrinkPatches_flag, shrinkPatches_flag);
}

void HyPerConn::ioParam_shrinkPatchesThresh(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "shrinkPatches"));
   if (shrinkPatches_flag) {
      parent->ioParamValue(ioFlag, name, "shrinkPatchesThresh", &shrinkPatchesThresh, shrinkPatchesThresh);
   }
}

void HyPerConn::ioParam_updateGSynFromPostPerspective(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "updateGSynFromPostPerspective", &updateGSynFromPostPerspective, updateGSynFromPostPerspective);
}

void HyPerConn::ioParam_dWMax(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag) {
      parent->ioParamValueRequired(ioFlag, name, "dWMax", &dWMax);
   }
}

void HyPerConn::ioParam_normalizeMethod(enum ParamsIOFlag ioFlag) {
   parent->ioParamString(ioFlag, name, "normalizeMethod", &normalizeMethod, NULL, true/*warnIfAbsent*/);
   if (ioFlag==PARAMS_IO_READ) {
      if (normalizeMethod==NULL) {
         if (parent->columnId()==0) {
            exitFailure("Error in %s \"%s\": specifying a normalizeMethod string is required.\n", getKeyword(), name);
         }
      }
      if (!strcmp(normalizeMethod, "")) {
         free(normalizeMethod);
         normalizeMethod = strdup("none");
      }
      if (normalizer==NULL) {
         int status = setWeightNormalizer();
         if (status != PV_SUCCESS) {
            exitFailure("%s \"%s\": Rank %d process unable to construct weight normalizer\n", getKeyword(), name, parent->columnId());
         }
      }
      if (normalizer!=NULL) {
         normalizer->addConnToList(this);
      }
   }
}

void HyPerConn::ioParam_weightSparsity(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "weightSparsity", &_weightSparsity, 0.0f, false);
}

int HyPerConn::setWeightNormalizer() {
   pvAssert(normalizer==NULL);
   pvAssert(normalizeMethod != NULL);
   CoreParamGroupHandler * weightNormalizerHandler = new CoreParamGroupHandler();
   normalizer = weightNormalizerHandler->createWeightNormalizer(normalizeMethod, name, parent);
   delete weightNormalizerHandler;
   int status = PV_SUCCESS;
   return status;
}

void HyPerConn::ioParam_normalizeGroupName(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "normalizeMethod"));
   // Note: subclasses may override ioParam_normalizeMethod so that it is possible for normalizeMethod to be NULL
   // even though HyPerConn::ioParam_normalizeMethod itself always sets normalizeMethod
   if (normalizeMethod && !strcmp(normalizeMethod, "normalizeGroup")) {
      parent->ioParamStringRequired(ioFlag, name, "normalizeGroupName", &normalizeGroupName);
   }
}

void HyPerConn::ioParam_keepKernelsSynchronized(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "sharedWeights"));
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (sharedWeights && plasticityFlag) {
      parent->ioParamValue(ioFlag, name, "keepKernelsSynchronized", &keepKernelsSynchronized_flag, true/*default*/, true/*warnIfAbsent*/);
   }
}

void HyPerConn::ioParam_normalizeDw(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if(plasticityFlag){
      getParent()->ioParamValue(ioFlag, getName(), "normalizeDw", &normalizeDwFlag, true, false/*warnIfAbsent*/);
   }
}

void HyPerConn::ioParam_useMask(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if(plasticityFlag){
      getParent()->ioParamValue(ioFlag, getName(), "useMask", &useMask, false, false/*warnIfAbsent*/);
   }
}

void HyPerConn::ioParam_maskLayerName(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if(plasticityFlag){
      pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "useMask"));
      if(useMask){
         parent->ioParamStringRequired(ioFlag, name, "maskLayerName", &maskLayerName);
      }
   }
}

void HyPerConn::ioParam_maskFeatureIdx(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if(plasticityFlag){
      pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "useMask"));
      if(useMask){
         parent->ioParamValue(ioFlag, name, "maskFeatureIdx", &maskFeatureIdx, maskFeatureIdx);
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
      int nxp_orig = xPatchSize();
      int nyp_orig = yPatchSize();
      nxpPost = nxp_orig;
      if(xscaleDiff > 0 ) {
         nxpPost *= (int) pow( 2, xscaleDiff );
      }
      else if(xscaleDiff < 0) {
         nxpPost /= (int) pow(2,-xscaleDiff);
         pvAssert(nxp_orig==nxpPost*pow( 2, (float) (-xscaleDiff) ));
      }

      int yscaleDiff = post->getYScale() - pre->getYScale();
      nypPost = nyp_orig;
      if(yscaleDiff > 0 ) {
         nypPost *= (int) pow( 2, yscaleDiff );
      }
      else if(yscaleDiff < 0) {
         nypPost /= (int) pow(2,-yscaleDiff);
         pvAssert(nyp_orig==nypPost*pow( 2, (float) (-yscaleDiff) ));
      }

      nfpPost = pre->getLayerLoc()->nf;
   }

   return PV_SUCCESS;
}

int HyPerConn::communicateInitInfo() {
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
   // Subclasses (e.g. CloneKernelConn) may also need
   // to send messages to related layers and connections before the allocation
   // phase.  These subclasses should override communicateInitInfo(), and the
   // subclass's communicateInitInfo() should call the parent class's communicateInitInfo().

   int status = BaseConnection::communicateInitInfo();
   if (status != PV_SUCCESS) {
      if (parent->columnId()==0) {
         logError("%s \"%s\": communicateInitInfo failed.\n", getKeyword(), name);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   pvAssert(preSynapticLayer()!=NULL && postSynapticLayer()!=NULL);
   handleDefaultSelfFlag();

   if(useMask) {
      mask = getParent()->getLayerFromName(maskLayerName);
      if (mask == NULL) {
         if (getParent()->columnId()==0) {
            logError("Connection \"%s\": maskLayerName \"%s\" does not correspond to a layer in the column.\n", getName(), maskLayerName);
         }
         status = PV_FAILURE;
         exit(-1);
      }
      //Check mask with restricted post layer
      const PVLayerLoc * maskLoc = mask->getLayerLoc();
      const PVLayerLoc * postLoc = post->getLayerLoc();
      if(postLoc->nx != maskLoc->nx || postLoc->ny != maskLoc->ny){
         if (getParent()->columnId()==0) {
            logError("Connection \"%s\": Mask \"%s\" (%d, %d, %d) must have the same x and y size as post layer \"%s\" (%d, %d, %d).\n", getName(), maskLayerName, maskLoc->nx, maskLoc->ny, maskLoc->nf, post->getName(), postLoc->nx, postLoc->ny, postLoc->nf);
         }
         status = PV_FAILURE;
         exit(-1);
      }
      //Make sure maskFeatureIdx is within bounds
      if(maskFeatureIdx >= maskLoc->nf || maskFeatureIdx < -1){
         logError("Connection \"%s\": maskFeatureIdx must be between -1 (inclusive) and mask layer \"%s\" (%d, %d, %d) nf dimension (exclusive)\n", getName(), maskLayerName, maskLoc->nx, maskLoc->ny, maskLoc->nf);
         status = PV_FAILURE;
         exit(-1);
      }

      //This check is only required if a maskFeatureIdx is not specified, aka, pointwise masking
      if(maskFeatureIdx == -1){
         if(postLoc->nf != maskLoc->nf && maskLoc->nf != 1){
            if (getParent()->columnId()==0) {
               logError("Connection \"%s\": Mask \"%s\" (%d, %d, %d) nf dimension must be either the same as post layer \"%s\" (%d, %d, %d) or 1\n", getName(), maskLayerName, maskLoc->nx, maskLoc->ny, maskLoc->nf, post->getName(), postLoc->nx, postLoc->ny, postLoc->nf);
            }
            status = PV_FAILURE;
            exit(-1);
         }
      }
   }


   if (getPvpatchAccumulateType()==ACCUMULATE_STOCHASTIC && (getConvertRateToSpikeCount() || pre->activityIsSpiking())) {
      if (parent->columnId()==0) {
         logError("Connection \"%s\": stochastic accumulation function is not consistent with ", getName());
         if (getConvertRateToSpikeCount()) {
            logError("setting convertRateToSpikeCount to true.\n");
         }
         else {
            pvAssertMessage(pre->activityIsSpiking(), "a spiking presynaptic layer \"%s\".\n", pre->getName());
         }
      }
      MPI_Barrier(getParent()->icCommunicator()->communicator());
      status = PV_FAILURE;
      exit(-1);
   }

   status = setPatchSize();
   status = checkPatchDimensions();

   if (nfp == -1) {
      nfp = post->getCLayer()->loc.nf;
      if (warnDefaultNfp && parent->columnId()==0) {
         logInfo("Connection \"%s\" setting nfp to number of postsynaptic features = %d.\n", name, nfp);
      }
   }
   if (nfp != post->getCLayer()->loc.nf) {
      if (parent->columnId()==0) {
         logError("Params file specifies %d features for connection \"%s\",\n", nfp, name );
         logError("but %d features for post-synaptic layer %s\n", post->getCLayer()->loc.nf, post->getName() );
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(PV_FAILURE);
   }
   // Currently, the only acceptable number for nfp is the number of post-synaptic features.
   // However, we may add flexibility on this score in the future, e.g. MPI in feature space
   // with each feature connecting to only a few nearby features.
   // Accordingly, we still keep readNfp.

   int xmargin = computeMargin(pre->getXScale(), post->getXScale(), nxp);
   int ymargin = computeMargin(pre->getYScale(), post->getYScale(), nyp);
   int receivedxmargin = 0;
   int statusx = pre->requireMarginWidth(xmargin, &receivedxmargin, 'x');
   if (statusx != PV_SUCCESS) {
      logError("Margin Failure for layer %s.  Received x-margin is %d, but connection \"%s\" requires margin of at least %d\n", pre->getName(),receivedxmargin, name, xmargin);
      status = PV_MARGINWIDTH_FAILURE;
   }
   int receivedymargin = 0;
   int statusy = pre->requireMarginWidth(ymargin, &receivedymargin, 'y');
   if (statusy != PV_SUCCESS) {
      logError("Margin Failure for layer %s.  Received y-margin is %d, but connection \"%s\" requires margin of at least %d\n", pre->getName(),receivedymargin, name, ymargin);
      status = PV_MARGINWIDTH_FAILURE;
   }

   status = setPostPatchSize();

   //Trigger stuff
   if(triggerLayerName){
      triggerLayer = parent->getLayerFromName(triggerLayerName);
      if (triggerLayer==NULL) {
         if (parent->columnId()==0) {
            logError("%s \"%s\" error: triggerLayer \"%s\" is not a layer in the HyPerCol.\n",
                    getKeyword(), name, triggerLayerName);
         }
         MPI_Barrier(parent->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }

      //Although weightUpdatePeriod and weightUpdateTime is being set here, if trigger flag is set, they are not being used
      //Only updating for backwards compatibility
      weightUpdatePeriod = triggerLayer->getDeltaUpdateTime();
      if(weightUpdatePeriod <= 0){
         if(plasticityFlag == true){
            std::cout << "Warning: Connection " << name << "triggered layer " << triggerLayerName << " never updates, turning placisity flag off\n";
            plasticityFlag = false;
         }
      }
      if(weightUpdatePeriod != -1 && triggerOffset >= weightUpdatePeriod){
         logError("%s \"%s\" error in rank %d process: TriggerOffset (%f) must be lower than the change in update time (%f) of the attached trigger layer\n", getKeyword(), name, parent->columnId(), triggerOffset, weightUpdatePeriod);
         exit(EXIT_FAILURE);
      }
      weightUpdateTime = parent->getDeltaTime();
   }

   if (weightInitializer) { weightInitializer->communicateParamsInfo(); }

   if (sharedWeights) {
      fileType = PVP_KERNEL_FILE_TYPE;
   }
   else {
      fileType = PVP_WGT_FILE_TYPE;
   }

   if (normalizeGroupName) {
      pvAssert(!strcmp(normalizeMethod, "normalizeGroup"));
      NormalizeBase * groupNormalizer = parent->getNormalizerFromName(normalizeGroupName);
      if (groupNormalizer==NULL) {
         if (parent->columnId()==0) {
            logError("%s \"%s\" error: normalizeGroupName \"%s\" is not a recognized normalizer.\n",
                  getKeyword(), name, normalizeGroupName);
         }
         MPI_Barrier(parent->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      groupNormalizer->addConnToList(this);
   }

   //Check if need transpose
   if(updateGSynFromPostPerspective) {
      needPost = true;
   }

//GPU stuff
#if defined(PV_USE_CUDA)
   //Here, the connection tells all participating recev layers to allocate memory on gpu
   //if receive from gpu is set. These buffers should be set in allocate
   if(receiveGpu){
      //we need pre datastore, this conn's weights, and post gsyn on the channel of this connection
      pre->setAllocDeviceDatastore();
      if(updateGSynFromPostPerspective){
         setAllocPostDeviceWeights();
         //Increment number of postKernels for workspace memory
         parent->getDevice()->incrementConvKernels();
      }
      else{
         setAllocDeviceWeights();
      }
      post->setAllocDeviceGSyn();

      //If recv from pre and pre layer is sparse, allocate activeIndices
      if(!updateGSynFromPostPerspective && pre->getSparseFlag()){
         pre->setAllocDeviceActiveIndices();
      }
   }
#endif

   //No batches with non-shared weights
   if(parent->getNBatch() > 1 && !sharedWeights){
      if (parent->columnId()==0) {
         logError("%s \"%s\" error: Non-shared weights with batches not implemented yet.\n",
               getKeyword(), name);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   return status;
}

int HyPerConn::allocatePostToPreBuffer(){
   if(postToPreActivity){return PV_SUCCESS;}
   //update conn to original connection
   const PVLayerLoc * sourceLoc = preSynapticLayer()->getLayerLoc();
   const PVLayerLoc * targetLoc = postSynapticLayer()->getLayerLoc();

   const int sourceNx = sourceLoc->nx;
   const int sourceNy = sourceLoc->ny;
   const int sourceNf = sourceLoc->nf;
   const int targetNx = targetLoc->nx;
   const int targetNy = targetLoc->ny;
   const int targetNf = targetLoc->nf;

   float sourceToTargetScaleX = (float)sourceNx/targetNx;
   float sourceToTargetScaleY = (float)sourceNy/targetNy;

   const PVHalo * sourceHalo = &sourceLoc->halo;

   const int numRestricted = postSynapticLayer()->getNumNeurons();

   postToPreActivity = (long*)pvMalloc(sizeof(long) * numRestricted);

   //origpre many, origpost one
   if(sourceToTargetScaleX >= 1 && sourceToTargetScaleY >= 1){
      for (int kTargetRes = 0; kTargetRes < numRestricted; kTargetRes++){
         int kTargetXRes = kxPos(kTargetRes, targetNx, targetNy, targetNf);
         int kTargetYRes = kyPos(kTargetRes, targetNx, targetNy, targetNf);

         int xVal = (sourceToTargetScaleX * kTargetXRes) - ((postConn->xPatchSize() - sourceToTargetScaleX)/2);
         int yVal = (sourceToTargetScaleY * kTargetYRes) - ((postConn->yPatchSize() - sourceToTargetScaleY)/2);

         postToPreActivity[kTargetRes] = kIndex(xVal + sourceHalo->lt, yVal + sourceHalo->up, 0,
               sourceNx + sourceHalo->lt + sourceHalo->rt, sourceNy + sourceHalo->up + sourceHalo->dn, sourceNf); 
      }
   }

   //origpost many, origpre one
   else if(sourceToTargetScaleX <= 1 && sourceToTargetScaleY <= 1){
      int targetToSourceScaleX = (float)1/sourceToTargetScaleX;
      int targetToSourceScaleY = (float)1/sourceToTargetScaleY;
      for (int kTargetRes = 0; kTargetRes < numRestricted; kTargetRes++){
         int kTargetXRes = kxPos(kTargetRes, targetNx, targetNy, targetNf);
         int kTargetYRes = kyPos(kTargetRes, targetNx, targetNy, targetNf);

         int centerX = floor((float)kTargetXRes/((float)targetToSourceScaleX/2));
         int centerY = floor((float)kTargetYRes/((float)targetToSourceScaleY/2));
         int offsetX = postConn->xPatchSize()-1;
         int offsetY = postConn->yPatchSize()-1;

         int xVal = floor(((float)centerX - offsetX)/2);
         int yVal = floor(((float)centerY - offsetY)/2);
         postToPreActivity[kTargetRes] = kIndex(xVal + sourceHalo->lt, yVal + sourceHalo->up, 0,
               sourceNx + sourceHalo->lt + sourceHalo->rt, sourceNy + sourceHalo->up + sourceHalo->dn, sourceNf); 
      }
   }
   else{
      exitFailure("sourceToTargetScaleX= %f, sourceToTargetScaleY= %f: the case of many-to-one in one dimension and one-to-many in the other has not yet been implemented.\n", sourceToTargetScaleX, sourceToTargetScaleY);
   }
   
   return PV_SUCCESS;
}


void HyPerConn::handleDefaultSelfFlag() {
   if (!parent->parameters()->present(name, "selfFlag")) {
      selfFlag = (pre == post);
   }
   else {
      // parameter was specified in params; use that value.
   }
}

int HyPerConn::setPatchSize() {
   int status = PV_SUCCESS;
   // Some subclasses determine some of {nxp, nyp, nfp} from other layers or connections (e.g. TransposeConn, CloneKernelConn)
   // instead of reading them from params.  They should override setPatchSize() to set those params.
   return status;
}

// returns handle to initialized weight patches
PVPatch *** HyPerConn::initializeWeights(PVPatch *** patches, pvwdata_t ** dataStart)
{
   PVPatch *** patches_arg = sharedWeights ? NULL : patches;
   weightInitializer->initializeWeights(patches_arg, dataStart);
   return patches;
}

int HyPerConn::allocatePostConn(){
   int status = PV_SUCCESS;
   //Allocate private transpose conn
   if(needPost) {
      char privateConnName [PV_PATH_MAX];
      sprintf(privateConnName, "&%s_privatePostConn", name);
      postConn = new privateTransposeConn(privateConnName, parent, this);
      pvAssert(postConn);
      status = postConn->allocateDataStructures();
   }
   //Can't do this with shrink patches flag
   if(needPost && !shrinkPatches_flag) {
      status = allocatePostToPreBuffer();
      postConn->allocatePostToPreBuffer();
   }
   return status;
}


int HyPerConn::allocateDataStructures() {
#if defined(PV_USE_CUDA)
   if (!pre->getDataStructuresAllocatedFlag()) {
      if (parent->columnId()==0) {
         const char * connectiontype = this->getKeyword();
         logInfo("%s \"%s\" must wait until pre-synaptic layer \"%s\" has finished its allocateDataStructures stage.\n", connectiontype, name, pre->getName());
      }
      return PV_POSTPONE;
   }
#endif // defined(PV_USE_CUDA)
   int status = BaseConnection::allocateDataStructures();
   pvAssert(status == PV_SUCCESS);
   initNumWeightPatches();
   initNumDataPatches();
   initPatchToDataLUT();

   if (pvpatchAccumulateType == ACCUMULATE_STOCHASTIC) {
      bool from_post = getUpdateGSynFromPostPerspective();
      if (from_post) {
         randState = new Random(parent, postSynapticLayer()->getLayerLoc(), false/*isExtended*/);
      }
      else {
         randState = new Random(parent, preSynapticLayer()->getLayerLoc(), true/*isExtended*/);
      }
   }

   if (plasticityFlag) {
      if (parent->getCheckpointReadFlag()==false && weightUpdateTime < parent->simulationTime()) {
         while(weightUpdateTime <= parent->simulationTime()) {weightUpdateTime += weightUpdatePeriod;}
         if (parent->columnId()==0) {
            logError("Warning: initialWeightUpdateTime of %s \"%s\" less than simulation start time.  Adjusting weightUpdateTime to %f\n",
                  getKeyword(), name, weightUpdateTime);
         }
      }
      lastUpdateTime = weightUpdateTime - parent->getDeltaTime();
   }

   status = constructWeights();

   allocatePostConn();

#if defined(PV_USE_CUDA)
   status = allocateDeviceBuffers();
   if(receiveGpu){
      if(updateGSynFromPostPerspective){
         status |= allocateReceivePostKernel();
      }
      else{
         status |= allocateReceivePreKernel();
      }
   }
   if(status == 0) {
      status = PV_SUCCESS;
   }
   else{
      logError("Connection \"%s\" unable to allocate device memory in rank %d process: %s\n", getName(), getParent()->columnId(), strerror(errno));
      exit(PV_FAILURE);
   }
#endif

   //Allocate temp buffers if needed, 1 for each thread
   //Only allocate for recv from pre, and not threading over batches
   if(!getUpdateGSynFromPostPerspective() && parent->getNumThreads() > 1) {
      //thread_gSyn is only a buffer for one batch, as if we're not threading over batches, batches will be sequential
      thread_gSyn = (pvdata_t**) pvMalloc(sizeof(pvdata_t*) * parent->getNumThreads());

      //Assign thread_gSyn to different points of tempMem
      for(int i = 0; i < parent->getNumThreads(); i++){
         thread_gSyn[i] = (pvdata_t*) pvMallocError(sizeof(pvdata_t) * post->getNumNeurons(), "HyPerLayer \"%s\" error: rank %d unable to allocate %zu memory for thread_gSyn", name, parent->columnId(), sizeof(pvdata_t) * post->getNumNeurons());
      }
   }

   //Allocate batchSkip buffer
   batchSkip = (bool*) pvMallocError(parent->getNBatch() * sizeof(bool), "HyPerLayer \"%s\" error: rank %d unable to allocate %zu memory for batchSkip", name, parent->columnId(), sizeof(bool) * parent->getNBatch());

   for(int i = 0; i < parent->getNBatch(); i++){
      batchSkip[i] = false;
   }

   return status;
}

void HyPerConn::initPatchToDataLUT() {
   pvAssert(patch2datalookuptable==NULL);
   if (sharedWeights) {
      int numWeightPatches=getNumWeightPatches();

      patch2datalookuptable = (int *)pvCalloc(numWeightPatches, sizeof(int));

      for(int i = 0; i < numWeightPatches; i++) {
         int kernelindex = patchIndexToDataIndex(i);
         patch2datalookuptable[i] = kernelindex;
      }
   } else {
      // lookuptable just returns the patchindex
   }
}

taus_uint4 * HyPerConn::getRandState(int index) {
   taus_uint4 * state = NULL;
   if (pvpatchAccumulateType==ACCUMULATE_STOCHASTIC) {
      state = randState->getRNG(index);
   }
   return state;
}

InitWeights * HyPerConn::getDefaultInitWeightsMethod(const char * keyword) {
   if (parent->columnId()==0) {
      logError("Connection \"%s\": weightInitType \"%s\" not recognized.  Exiting\n", name, weightInitTypeString);
   }
   MPI_Barrier(parent->icCommunicator()->communicator());
   exit(EXIT_FAILURE);
}

#if defined(PV_USE_CUDA)

int HyPerConn::allocatePostDeviceWeights(){
   pvAssert(postConn);
   postConn->allocateDeviceWeights();
   return PV_SUCCESS;
}

int HyPerConn::allocateDeviceWeights(){
#ifdef PV_USE_CUDA
   PVCuda::CudaDevice * device = parent->getDevice();
#endif
   const size_t size = numberOfAxonalArborLists() * getNumDataPatches() * xPatchSize() * yPatchSize() * fPatchSize() * sizeof(pvwdata_t);
#ifdef PV_USE_CUDA
   d_WData = device->createBuffer(size);
   pvAssert(d_WData);
#ifdef PV_USE_CUDNN
   cudnn_WData = device->createBuffer(size);
#endif
#endif

   return PV_SUCCESS;

}

int HyPerConn::allocateDeviceBuffers()
{
   int status = 0;

#ifdef PV_USE_CUDA
   PVCuda::CudaDevice * device = parent->getDevice();
#endif

   bool needAlloc = true;
   if(allocDeviceWeights || allocPostDeviceWeights){
      //Check group here
#ifdef PV_USE_CUDA
      if(gpuGroupIdx >= 0){
         //Add to group if set
         parent->addGpuGroup(this, gpuGroupIdx);
         BaseConnection * b_conn = parent->getGpuGroupConn(gpuGroupIdx);
         //This connection must exist if gpuGroupIdx >= 0
         pvAssert(b_conn);
         HyPerConn * group_conn = dynamic_cast<HyPerConn *>(b_conn);
         if(!group_conn){
            std::cout << "FATAL ERROR: GPU group connection " << b_conn->getName() << " is not of type HyPerConn.\n";
            exit(-1);
         }
         //If this connection is NOT the "base" group conn that allocates
         //check dims and don't allocate
         if(group_conn != this){
            //Different num arbors is okay, since GPU mem holds only one arbor at a time
            //nxp, nyp, nfp, numKernels all have to be the same
            if(group_conn->xPatchSize() != xPatchSize() ||
               group_conn->yPatchSize() != yPatchSize() ||
               group_conn->fPatchSize() != fPatchSize() ||
               group_conn->getNumDataPatches() != getNumDataPatches() ||
               group_conn->numberOfAxonalArborLists() != numberOfAxonalArborLists()){
                  std::cout << "Connection " << getName() << " of size (" <<
                  numberOfAxonalArborLists() << ", " <<
                  getNumDataPatches() << ", " << 
                  xPatchSize() << ", " <<
                  yPatchSize() << ", " <<
                  fPatchSize() << 
                  ") does not match the gpuGroupConnection " << 
                  group_conn->getName() << " of size (" <<
                  group_conn->numberOfAxonalArborLists() << ", " <<
                  group_conn->getNumDataPatches() << ", " << 
                  group_conn->xPatchSize() << ", " <<
                  group_conn->yPatchSize() << ", " <<
                  group_conn->fPatchSize() << ").\n";
                  exit(-1);
            }
            //set d_WData to the group's d_WData
            d_WData = group_conn->getDeviceWData();
            pvAssert(d_WData);
#ifdef PV_USE_CUDNN
            cudnn_WData = group_conn->getCudnnWData();
            pvAssert(cudnn_WData);
#endif
            needAlloc = false;
         }
      }
#endif // PV_USE_CUDA

      if(needAlloc){
         if(allocPostDeviceWeights){
            allocatePostDeviceWeights();
         }
         if(allocDeviceWeights){
            allocateDeviceWeights();
         }
      }
   }

   if(receiveGpu){
      if(updateGSynFromPostPerspective){
         int numPostRes = post->getNumNeurons();
#ifdef PV_USE_CUDA
         d_PostToPreActivity = device->createBuffer(numPostRes*sizeof(long)); 
#endif

         if(sharedWeights){
            int numWeightPatches = postConn->getNumWeightPatches();
#ifdef PV_USE_CUDA
            d_Patch2DataLookupTable = device->createBuffer(numWeightPatches * sizeof(int));  
#endif
         }
      }
      else{
         //Calculate local pre size here
         const PVLayerLoc * preLoc = pre->getLayerLoc();
         const PVLayerLoc * postLoc = post->getLayerLoc();
         PVHalo haloPre;
         PVHalo haloPost;
         
         //Set local sizes here
         float preToPostScaleX = (float)preLoc->nx/((float)postLoc->nx);
         float preToPostScaleY = (float)preLoc->ny/((float)postLoc->ny);

         int preNf = preLoc->nf;
         int postNf = postLoc->nf;

         //This should be the case with petavision restrictions
         pvAssert(postNf == nfp);

         int numWeightPatches = pre->getNumExtended() ;
         int patchSize = numWeightPatches * sizeof(PVPatch);

#ifdef PV_USE_CUDA
         d_Patches = device->createBuffer(patchSize); 
#endif

         //Need a buffer for gsynpatch start for one arbor
         int gsynPatchStartIndexSize = numWeightPatches * sizeof(size_t);
#ifdef PV_USE_CUDA
         d_GSynPatchStart = device->createBuffer(gsynPatchStartIndexSize); 
#endif

         if(numberOfAxonalArborLists() == 1){
            PVPatch* h_patches = weights(0)[0]; //0 beacuse it's one block of memory
#ifdef PV_USE_CUDA
            PVCuda::CudaBuffer * d_patches = getDevicePatches();
#endif
            pvAssert(d_patches);
            d_patches->copyToDevice(h_patches);

            size_t* h_GSynPatchStart = getGSynPatchStart()[0];
#ifdef PV_USE_CUDA
            PVCuda::CudaBuffer * d_GSynPatchStart = getDeviceGSynPatchStart();
#endif
            pvAssert(d_GSynPatchStart);
            d_GSynPatchStart->copyToDevice(h_GSynPatchStart);
         }

         if(sharedWeights){
            int numWeightPatches = getNumWeightPatches();
#ifdef PV_USE_CUDA
            d_Patch2DataLookupTable = device->createBuffer(numWeightPatches * sizeof(int));  
#endif
         }
      }
   }
   return status;
}

int HyPerConn::allocateReceivePreKernel()
{
#ifdef PV_USE_CUDA
   int status = 0;
   PVCuda::CudaDevice * device = parent->getDevice();
   krRecvPre = new PVCuda::CudaRecvPre(device);
#endif

   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const PVLayerLoc * postLoc = post->getLayerLoc();
   const PVHalo * preHalo = &pre->getLayerLoc()->halo;
   const PVHalo * postHalo = &post->getLayerLoc()->halo;

#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer* d_PreData = pre->getDeviceDatastore();
   PVCuda::CudaBuffer* d_PostGSyn = post->getDeviceGSyn();
#endif

   pvAssert(d_PreData);
   pvAssert(d_PostGSyn);

   pvAssert(getDeviceWData());
   pvAssert(d_Patches);
   pvAssert(d_GSynPatchStart);

   int nxp = xPatchSize();
   int nyp = yPatchSize();
   int nfp = fPatchSize();
   float dt_factor = getConvertToRateDeltaTimeFactor();
   int i_sharedWeights = sharedWeights;

   int sy = getPostNonextStrides()->sy;
   int syw = yPatchStride();

   bool isSparse = pre->getSparseFlag();

   int numPreExt = pre->getNumExtended();
   int numPostRes = post->getNumNeurons();

   int nbatch = postLoc->nbatch;


#ifdef PV_USE_CUDA
      PVCuda::CudaBuffer* d_activeIndices = NULL;
      PVCuda::CudaBuffer* d_numActive = NULL;
#endif
   if(isSparse){
      d_numActive = pre->getDeviceNumActive();
      pvAssert(d_numActive);
      d_activeIndices = pre->getDeviceActiveIndices();
      pvAssert(d_activeIndices);
   }

   //Since it never changes, set this buffer here
   d_Patch2DataLookupTable->copyToDevice(getPatchToDataLUT());


   //Need to calculate new patches for weights
   //= conn->weights(arborID)[0]; //0 because it's one block of memory
   //CLBuffer * d_patches = getClPatches();
   //

#ifdef PV_USE_CUDA
   krRecvPre->setArgs(
      nbatch,
      numPreExt,
      numPostRes,
      nxp,
      nyp,
      nfp,

      sy,
      syw,
      dt_factor,
      i_sharedWeights,
      d_Patches,
      d_GSynPatchStart,

      d_PreData,
      getDeviceWData(),
      d_PostGSyn,
      d_Patch2DataLookupTable,

      isSparse,
      d_numActive,
      d_activeIndices
   );
#endif
   return status;
}

int HyPerConn::allocateReceivePostKernel()
{
   std::cout << name << " setting up post kernel\n";
#ifdef PV_USE_CUDA
   int status = 0;
   PVCuda::CudaDevice * device = parent->getDevice();
   krRecvPost = new PVCuda::CudaRecvPost(device);
#endif // PV_USE_CUDA

   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const PVLayerLoc * postLoc = post->getLayerLoc();
   const PVHalo* preHalo = &pre->getLayerLoc()->halo;
   const PVHalo* postHalo = &post->getLayerLoc()->halo;

#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer* d_PreData = pre->getDeviceDatastore();
   PVCuda::CudaBuffer* d_PostGSyn = post->getDeviceGSyn();
   PVCuda::CudaBuffer* d_origWData = postConn->getDeviceWData();

#ifdef PV_USE_CUDNN
   PVCuda::CudaBuffer * cudnn_preData = pre->getCudnnDatastore();
   PVCuda::CudaBuffer * cudnn_gSyn = post->getCudnnGSyn();
   PVCuda::CudaBuffer * cudnn_origWData = postConn->getCudnnWData();
   pvAssert(cudnn_preData);
   pvAssert(cudnn_gSyn);
   pvAssert(cudnn_origWData);
#endif

#endif // PV_USE_CUDA

   pvAssert(d_PreData);
   pvAssert(d_PostGSyn);
   pvAssert(d_origWData);


   int sy  = (preLoc->nx+preHalo->rt+preHalo->lt)*preLoc->nf;
   int syp = postConn->yPatchStride();
   int numPerStride = postConn->xPatchSize() * postConn->fPatchSize();
   float dt_factor = getConvertToRateDeltaTimeFactor();
   int i_sharedWeights = sharedWeights;

   const PVHalo* oHalo = &postConn->preSynapticLayer()->getLayerLoc()->halo;
   int oNblt = oHalo->lt;
   int oNbrt = oHalo->rt;
   int oNbup = oHalo->up;
   int oNbdn = oHalo->dn;

   //nxp, nyp, and nfp should be orig conn's
   int oNxp = postConn->xPatchSize();
   int oNyp = postConn->yPatchSize();
   int oNfp = postConn->fPatchSize();
   int postNx = postLoc->nx;
   int postNy = postLoc->ny;
   int postNf = postLoc->nf;

   int preNx = preLoc->nx;
   int preNy = preLoc->ny;
   int preNf = preLoc->nf;
   int preNblt = preHalo->lt;
   int preNbrt = preHalo->rt;
   int preNbup = preHalo->up;
   int preNbdn = preHalo->dn;

   int nbatch = preLoc->nbatch;
   

   //Set local sizes here
   float preToPostScaleX = (float)preLoc->nx/((float)postLoc->nx);
   float preToPostScaleY = (float)preLoc->ny/((float)postLoc->ny);

   //Since it never changes, set this buffer here
   //Need to set orig connection's patch2datalookuptable
   d_PostToPreActivity->copyToDevice(getPostToPreActivity());

   d_Patch2DataLookupTable->copyToDevice(postConn->getPatchToDataLUT());

   //In receive from post, we need to make sure x, y, and f local size is divisible by the actual number of post neurons
   if(postLoc->nx % numXLocal != 0){
      std::cout << "X local size of " << numXLocal << " is not divisible by post nx of " << postLoc->nx << "\n";
      exit(EXIT_FAILURE);
   }

   if(postLoc->ny % numYLocal != 0){
      std::cout << "Y local size of " << numYLocal << " is not divisible by post ny of " << postLoc->ny << "\n";
      exit(EXIT_FAILURE);
   }

   if(postLoc->nf % numFLocal != 0){
      std::cout << "F local size of " << numFLocal << " is not divisible by post nf of " << postLoc->nf << "\n";
      exit(EXIT_FAILURE);
   }

   int localBufSizeX;
   int localBufSizeY;
   //See the size of buffer needed based on x and y
   //oNxp is the patch size from the post point of view
   if(preToPostScaleX >= 1){
      localBufSizeX = oNxp + (preToPostScaleX * (numXLocal - 1));
   }
   else{
      localBufSizeX = oNxp + ((preToPostScaleX * numXLocal) - 1);
   }
   if(preToPostScaleY >= 1){
      localBufSizeY = oNyp + (preToPostScaleY * (numYLocal - 1));
   }
   else{
      localBufSizeY = oNyp + ((preToPostScaleY * numYLocal) - 1);
   }

   if (parent->columnId()==0) {
      std::cout << "preToPostScale: (" << preToPostScaleX << "," << preToPostScaleY << ")\n";
      std::cout << "patch size: (" << oNxp << "," << oNyp << ") numLocal: (" << numXLocal << "," << numYLocal << ")\n";
      std::cout << "local sizes: (" << localBufSizeX << "," << localBufSizeY << ")\n";
   }
   
#ifdef PV_USE_CUDA
   krRecvPost->setArgs(
      nbatch,
      postNx, //num post neurons
      postNy,
      postNf,

      oNblt, //Border of orig
      oNbrt, //Border of orig
      oNbdn, //Border of orig
      oNbup, //Border of orig

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

      localBufSizeX,
      localBufSizeY,
      preToPostScaleX,
      preToPostScaleY,

      sy,
      syp,
      numPerStride,
      dt_factor,
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
      d_Patch2DataLookupTable,

      preDataLocal
   );
#endif
   return status;
}

#endif

int HyPerConn::writeWeights(double timed, bool last)
{
   PVPatch *** patches_arg = sharedWeights ? NULL : wPatches;
   return writeWeights(patches_arg, get_wDataStart(), getNumDataPatches(), NULL, timed, writeCompressedWeights, last);
}

int HyPerConn::writeWeights(const char * filename) {
   PVPatch *** patches_arg = sharedWeights ? NULL : wPatches;
   return writeWeights(patches_arg, get_wDataStart(), getNumDataPatches(), filename, parent->simulationTime(), writeCompressedWeights, true);
}

int HyPerConn::writeWeights(PVPatch *** patches, pvwdata_t ** dataStart, int numPatches,
      const char * filename, double timed, bool compressWeights, bool last) {
   int status = PV_SUCCESS;
   char path[PV_PATH_MAX];

   float minVal = FLT_MAX;
   float maxVal = -FLT_MAX;
   for(int arbor=0; arbor<numberOfAxonalArborLists(); arbor++) {
      float minVal1 = minWeight(arbor);
      if( minVal1 < minVal ) minVal = minVal1;
      float maxVal1 = maxWeight(arbor);
      if( maxVal1 > maxVal ) maxVal = maxVal1;
   }

   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const PVLayerLoc * postLoc = post->getLayerLoc();

   int chars_needed = 0;
   if (filename == NULL) {
      pvAssert(parent->includeConnectionName()<=2 && parent->includeConnectionName()>=0);
      switch(parent->includeConnectionName()) {
      case 0:
         chars_needed = snprintf(path, PV_PATH_MAX, "%s/w%d.pvp", parent->getOutputPath(), getConnectionId());
         break;
      case 1:
         chars_needed = snprintf(path, PV_PATH_MAX, "%s/w%d_%s.pvp", parent->getOutputPath(), getConnectionId(), name);
         break;
      case 2:
         chars_needed = snprintf(path, PV_PATH_MAX, "%s/%s.pvp", parent->getOutputPath(), name);
         break;
      default:
         pvAssert(0);
         break;
      }
   }
   else {
      chars_needed = snprintf(path, PV_PATH_MAX, "%s", filename);
   }
   if (chars_needed >= PV_PATH_MAX) {
      logError("HyPerConn::writeWeights in connection \"%s\": path is too long (it would be cut off as \"%s\")\n", name, path);
      abort();
   }

   Communicator * comm = parent->icCommunicator();

   bool append = last ? false : ioAppend;

   status = PV::writeWeights(path, comm, (double) timed, append, preLoc, postLoc, nxp, nyp,
         nfp, minVal, maxVal, patches, dataStart, numPatches,
         numberOfAxonalArborLists(), compressWeights, fileType);
   if(status != PV_SUCCESS) {
      logError("%s \"%s\" error in writing weights.\n", getKeyword(), name);
      exit(EXIT_FAILURE);
   }

   return status;
}

int HyPerConn::writeTextWeights(const char * filename, int k)
{
   if (parent->icCommunicator()->commSize()>1) {
      logError("writeTextWeights error for connection \"%s\": writeTextWeights is not compatible with MPI", name);
      abort();
      // NOTE : if run under MPI when more than one process sees the same file system, the contending processes will clobber each other.
   }
   PV_Stream * pvstream = NULL;

   if (filename != NULL) {
      char outfile[PV_PATH_MAX];
      snprintf(outfile, PV_PATH_MAX-1, "%s/%s", parent->getOutputPath(), filename);
      pvstream = PV_fopen(outfile, "w", parent->getVerifyWrites());
   }
   else {
      pvstream = PV_stdout();
   }
   if (pvstream == NULL) {
     logError("writeWeights: ERROR opening file \"%s\"\n", filename);
     return PV_FAILURE;
   }

   FILE * fd = pvstream->fp;
   fprintf(fd, "Weights for connection \"%s\", neuron %d\n", name, k);
   fprintf(fd, "   (kxPre,kyPre,kfPre)   = (%i,%i,%i)\n",
           kxPos(k,pre->getLayerLoc()->nx + pre->getLayerLoc()->halo.lt + pre->getLayerLoc()->halo.rt,
                 pre->getLayerLoc()->ny + pre->getLayerLoc()->halo.dn + pre->getLayerLoc()->halo.up, pre->getLayerLoc()->nf),
           kyPos(k,pre->getLayerLoc()->nx + pre->getLayerLoc()->halo.lt + pre->getLayerLoc()->halo.rt,
                 pre->getLayerLoc()->ny + pre->getLayerLoc()->halo.dn + pre->getLayerLoc()->halo.up, pre->getLayerLoc()->nf),
           featureIndex(k,pre->getLayerLoc()->nx + pre->getLayerLoc()->halo.lt + pre->getLayerLoc()->halo.rt,
                 pre->getLayerLoc()->ny + pre->getLayerLoc()->halo.dn + pre->getLayerLoc()->halo.up, pre->getLayerLoc()->nf) );
   fprintf(fd, "   (nxp,nyp,nfp)   = (%i,%i,%i)\n", (int) nxp, (int) nyp, (int) nfp);
   fprintf(fd, "   pre  (nx,ny,nf) = (%i,%i,%i)\n",
           pre->getLayerLoc()->nx, pre->getLayerLoc()->ny, pre->getLayerLoc()->nf);
   fprintf(fd, "   post (nx,ny,nf) = (%i,%i,%i)\n",
           post->getLayerLoc()->nx, post->getLayerLoc()->ny, post->getLayerLoc()->nf);
   fprintf(fd, "\n");

   for(int arbor = 0; arbor<numberOfAxonalArborLists(); arbor++) {
      fprintf(fd, "displaying arbor %1.1d\n", arbor);
      // give a chance for derived classes to add extra information
      //
      writeTextWeightsExtra(pvstream, k, arbor);
      pv_text_write_patch(pvstream, wPatches[arbor][k], get_wData(arbor,k), nfp, sxp, syp, sfp);
      fprintf(fd, "----------------------------\n");
   }

   PV_fclose(pvstream);

   return 0;
}

int HyPerConn::readStateFromCheckpoint(const char * cpDir, double * timeptr) {
   // If timeptr is NULL, the timestamps in the pvp files are ignored.  If non-null, they are compared to the value of *timeptr and
   // a warning is issued if there is a discrepancy.
   int status = PV_SUCCESS;
   status = readWeightsFromCheckpoint(cpDir, timeptr);
   return status;
}

int HyPerConn::readWeightsFromCheckpoint(const char * cpDir, double * timeptr) {
   clearWeights(get_wDataStart(), getNumDataPatches(), nxp, nyp, nfp);
   char * path = parent->pathInCheckpoint(cpDir, getName(), "_W.pvp");
   PVPatch *** patches_arg = sharedWeights ? NULL : wPatches;
   double filetime=0.0;
   int status = PV::readWeights(patches_arg, get_wDataStart(), numberOfAxonalArborLists(), getNumDataPatches(), nxp, nyp, nfp, path, parent->icCommunicator(), &filetime, pre->getLayerLoc());
   if (parent->columnId()==0 && timeptr && *timeptr != filetime) {
      logError("Warning: \"%s\" checkpoint has timestamp %g instead of the expected value %g.\n", path, filetime, *timeptr);
   }
   free(path);
   return status;
}

int HyPerConn::checkpointRead(const char * cpDir, double * timeptr) {
  //if((getPvpatchAccumulateType() == ACCUMULATE_MAXPOOLING) || (getPvpatchAccumulateType() == ACCUMULATE_SUMPOOLING)){
  //  return PV_SUCCESS;
  //}
   int status = readStateFromCheckpoint(cpDir, timeptr);

   status = parent->readScalarFromFile(cpDir, getName(), "lastUpdateTime", &lastUpdateTime, lastUpdateTime);
   pvAssert(status == PV_SUCCESS);
   if (plasticityFlag && !triggerLayerName && weightUpdateTime<parent->simulationTime()) {
      status = parent->readScalarFromFile(cpDir, getName(), "weightUpdateTime", &weightUpdateTime, weightUpdateTime);
      pvAssert(status == PV_SUCCESS);
      // simulationTime() may have been changed by HyPerCol::checkpoint, so this repeats the sanity check on weightUpdateTime in allocateDataStructures
      while(weightUpdateTime <= parent->simulationTime()) {weightUpdateTime += weightUpdatePeriod;}
      if (parent->columnId()==0) {
         logError("Warning: initialWeightUpdateTime of %s \"%s\" less than simulation start time.  Adjusting weightUpdateTime to %f\n",
               getKeyword(), name, weightUpdateTime);
      }
   }

   status = parent->readScalarFromFile(cpDir, getName(), "nextWrite", &writeTime, writeTime);
   pvAssert(status == PV_SUCCESS);

   return status;
}

int HyPerConn::checkpointWrite(const char * cpDir) {
  //if((getPvpatchAccumulateType() == ACCUMULATE_MAXPOOLING) || (getPvpatchAccumulateType() == ACCUMULATE_SUMPOOLING)){
  //  return PV_SUCCESS;
  //}
   char filename[PV_PATH_MAX];
   int status = checkpointFilename(filename, PV_PATH_MAX, cpDir);
   pvAssert(status==PV_SUCCESS);
   PVPatch *** patches_arg = sharedWeights ? NULL : wPatches;
   status = writeWeights(patches_arg, wDataStart, getNumDataPatches(), filename, parent->simulationTime(), writeCompressedCheckpoints, /*last*/true);
   pvAssert(status==PV_SUCCESS);

   // TODO: split the writeScalarToFile calls up into virtual methods so that subclasses that don't use these member variables don't have to save them.
   status = parent->writeScalarToFile(cpDir, getName(), "nextWrite", writeTime);
   pvAssert(status==PV_SUCCESS);
   status = parent->writeScalarToFile(cpDir, getName(), "lastUpdateTime", lastUpdateTime);
   pvAssert(status==PV_SUCCESS);
   if (plasticityFlag && !triggerLayerName) {
      status = parent->writeScalarToFile(cpDir, getName(), "weightUpdateTime", weightUpdateTime);
   }
   pvAssert(status==PV_SUCCESS);
   return status;
}

int HyPerConn::checkpointFilename(char * cpFilename, int size, const char * cpDir) {
   int chars_needed = snprintf(cpFilename, size, "%s/%s_W.pvp", cpDir, name);
   if(chars_needed >= PV_PATH_MAX) {
      if ( parent->icCommunicator()->commRank()==0 ) {
         logError("HyPerConn::checkpointFilename error: path \"%s/%s_W.pvp\" is too long.\n", cpDir, name);
      }
      abort();
   }
   return PV_SUCCESS;
}

int HyPerConn::writeTimers(FILE* stream){
   if (parent->icCommunicator()->commRank()==0) {
      io_timer->fprint_time(stream);
      update_timer->fprint_time(stream);
      for (int p=0; p<numProbes; p++){
         probes[p]->writeTimer(stream);
      }
      if(needPost){
         postConn->writeTimers(stream);
      }
   }
   return PV_SUCCESS;
}

float HyPerConn::minWeight(int arborId)
{
   const int num_data_patches = getNumDataPatches();
   float min_weight = FLT_MAX;
   if (sharedWeights) {
      const int numWeights = nxp * nyp * nfp;
      for (int iKernel = 0; iKernel < num_data_patches; iKernel++) {
         pvwdata_t * kernelWeights = get_wDataHead(arborId, iKernel);
         for (int iWeight = 0; iWeight < numWeights; iWeight++) {
            min_weight = (min_weight < kernelWeights[iWeight]) ? min_weight
                  : kernelWeights[iWeight];
         }
      }
   }
   else {
      for (int i_patch = 0; i_patch < num_data_patches; i_patch++) {
         pvwdata_t * w_data = get_wData(arborId, i_patch);
         PVPatch * w_patch = getWeights(i_patch, arborId);
         int num_weights = fPatchSize() * w_patch->nx * w_patch->ny;
         for (int iWeight = 0; iWeight < num_weights; iWeight++) {
            min_weight = (min_weight < w_data[iWeight]) ? min_weight
                  : w_data[iWeight];
         }
      }
   }
   return min_weight;
}

float HyPerConn::maxWeight(int arborId)
{
   const int num_data_patches = getNumDataPatches();
   float max_weight = -FLT_MAX;
   if (sharedWeights) {
      const int numWeights = nxp * nyp * nfp;
      for (int iKernel = 0; iKernel < num_data_patches; iKernel++) {
         pvwdata_t * kernelWeights = get_wDataHead(arborId, iKernel);
         for (int iWeight = 0; iWeight < numWeights; iWeight++) {
            max_weight = (max_weight > kernelWeights[iWeight]) ? max_weight
                  : kernelWeights[iWeight];
         }
      }
   }
   else {
      for (int i_weight = 0; i_weight < num_data_patches; i_weight++) {
         pvwdata_t * w_data = get_wData(arborId, i_weight);
         PVPatch * w_patch = getWeights(i_weight, arborId);
         int num_weights = fPatchSize() * w_patch->nx * w_patch->ny;
         for (int iWeight = 0; iWeight < num_weights; iWeight++) {
            max_weight = (max_weight > w_data[iWeight]) ? max_weight
                  : w_data[iWeight];
         }
      }
   }
   return max_weight;
}

int HyPerConn::insertProbe(BaseConnectionProbe * p)
{
   if(p->getTargetConn() != this) {
      logError("HyPerConn \"%s\": insertProbe called with probe %p, whose targetConn is not this connection.  Probe was not inserted.\n", name, p);
      return numProbes;
   }
   for( int i=0; i<numProbes; i++ ) {
      if( p == probes[i] ) {
         logError("HyPerConn \"%s\": insertProbe called with probe %p, which has already been inserted as probe %d.\n", name, p, i);
         return numProbes;
      }
   }

   BaseConnectionProbe ** tmp;
   // malloc'ing a new buffer, copying data over, and freeing the old buffer could be replaced by malloc
   tmp = (BaseConnectionProbe **) pvMalloc((numProbes + 1) * sizeof(BaseConnectionProbe *));

   for (int i = 0; i < numProbes; i++) {
      tmp[i] = probes[i];
   }
   delete probes;

   probes = tmp;
   probes[numProbes] = p;

   return ++numProbes;
}

int HyPerConn::setInitialValues() {
   initializeWeights(wPatches, wDataStart);
   return PV_SUCCESS;
}

int HyPerConn::outputProbeParams() {
   int status = PV_SUCCESS;
   for (int p=0; p<numProbes; p++) {
      int status1 = probes[p]->ioParams(PARAMS_IO_WRITE);
      if (status1 != PV_SUCCESS) { status = PV_FAILURE; }
   }
   return status;
}

int HyPerConn::outputState(double timef, bool last)
{
   int status = 0;
   io_timer->start();

   if( !last ) {
      for (int i = 0; i < numProbes; i++) {
         probes[i]->outputStateWrapper(timef, parent->getDeltaTime());
      }
   }

   if (last) {
      status = writeWeights(timef, last);
      pvAssert(status == 0);
   }
   else if ( (writeStep >= 0) && (timef >= writeTime) ) {
      writeTime += writeStep;

      status = writeWeights(timef, last);
      pvAssert(status == 0);

      // append to output file after original open
      ioAppend = true;
   }
   else if (writeStep < 0) { // If writeStep is negative, we never call writeWeights, but someone might restart from a checkpoint with a different writeStep, so we should still maintain writeTime
      writeTime = timef;
   }

   io_timer->stop();
   return status;
}

bool HyPerConn::needUpdate(double time, double dt){
   if( !plasticityFlag ) {
      return false;
   }
   if(triggerLayer){
      double nextUpdateTime = triggerLayer->getNextUpdateTime();
      //never update flag
      if(nextUpdateTime == -1){
         return false;
      }
      //Check for equality
      if(fabs(time - (nextUpdateTime - triggerOffset)) < (dt/2)){
         return true;
      }
      //If it gets to this point, don't update
      return false;
   }
   //If no trigger, use weightUpdateTime
   else{
      if( time >= weightUpdateTime) {
         return true;
      }
      return false;
   }
}

int HyPerConn::updateState(double time, double dt){
   int status = PV_SUCCESS;
   if( !plasticityFlag ){return status;}

   update_timer->start();
   if(needUpdate(time, dt)){
      //Need to finish command queue of pre and post activity
      //Doing both in case of multiple gpus running

      //TODO: commented out to compile, but we'll want to average across only batches where timeScale >= timeScaleMin.
      for(int b = 0; b < parent->getNBatch(); b++){
         double preTimeScale = pre->getTimeScale(b); 
         double postTimeScale = post->getTimeScale(b);
         double colTimeScale = parent->getTimeScale(b);
         double timeScaleMin = parent->getTimeScaleMin();
         double skip = false;
         //If timeScale is less than the value for dtScaleMin specified in the params but not -1, don't updateState.
         //This is implemented as an optimization so weights don't change dramatically as ANNNormalizedErrorLayer values get large.
         if (preTimeScale > 0 && preTimeScale < timeScaleMin) { 
            if (parent->icCommunicator()->commRank()==0) {
               logInfo("TimeScale = %f for layer %s batch %d, which is less than your specified dtScaleMin, %f. updateState won't be called for connection \"%s\" this timestep.\n", preTimeScale, pre->getName(), b, timeScaleMin, getName());
            }
            skip = true;
         }
         else if (postTimeScale > 0 && postTimeScale < timeScaleMin) { 
            if (parent->icCommunicator()->commRank()==0) {
               logInfo("TimeScale = %f for layer %s batch %d, which is less than your specified dtScaleMin, %f. updateState won't be called for connection \"%s\" this timestep.\n", postTimeScale, post->getName(), b, timeScaleMin, getName());
            }
            skip = true;
         }
         else if (colTimeScale > 0 && colTimeScale < timeScaleMin) { 
            if (parent->icCommunicator()->commRank()==0) {
               logInfo("TimeScale = %f for column %s batch %d, which is less than your specified dtScaleMin, %f. updateState won't be called for connection \"%s\" this timestep.\n", colTimeScale, parent->getName(), b, timeScaleMin, getName());
            }
            skip = true;
         }
         batchSkip[b] = skip;
      }

      status = calc_dW();        // Calculate changes in weights

      for(int arborId=0;arborId<numberOfAxonalArborLists();arborId++){
         status = updateWeights(arborId);  // Apply changes in weights
         if (status==PV_BREAK) { break; }
         pvAssert(status==PV_SUCCESS);
      }

      lastUpdateTime = time;
      computeNewWeightUpdateTime(time, weightUpdateTime);
      needFinalize = true;
   }
   update_timer->stop();
   return status;
}

int HyPerConn::clear_numActivations(int arborId){
   // zero out all dW.
   // This also zeroes out the unused parts of shrunken patches
   for(int kArbor = 0; kArbor < numberOfAxonalArborLists(); kArbor++){
      for(int kKernel = 0; kKernel < getNumDataPatches(); kKernel++){
         int syPatch = syp;
         int nkPatch = nfp * nxp;
         long* activations = get_activationsHead(kArbor, kKernel);
         for(int kyPatch = 0; kyPatch < nyp; kyPatch++){
            for(int kPatch = 0; kPatch < nkPatch; kPatch++){
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
   for(int kArbor = 0; kArbor < numberOfAxonalArborLists(); kArbor++){
      for(int kKernel = 0; kKernel < getNumDataPatches(); kKernel++){
         int syPatch = syp;
         int nkPatch = nfp * nxp;
         pvwdata_t * dWeights = get_dwDataHead(kArbor,kKernel);
         for(int kyPatch = 0; kyPatch < nyp; kyPatch++){
            for(int kPatch = 0; kPatch < nkPatch; kPatch++){
               dWeights[kPatch] = 0.0f;
            }
            dWeights += syPatch;
         }
      }
   }
   return PV_BREAK;
}

int HyPerConn::initialize_dW(int arborId){
   if (!combine_dW_with_W_flag) { clear_dW(arborId); }
   if (numKernelActivations) { clear_numActivations(arborId); }
   //default initialize_dW returns PV_BREAK
   return PV_BREAK;
}

int HyPerConn::finalizeUpdate(double timed, double dt){
   if (!needFinalize) { return PV_SUCCESS; }

#if defined(PV_USE_CUDA)
   if(allocDeviceWeights){
      updateDeviceWeights();
   }
#endif

   //Update postConn if needed
   if(needPost){
      int status = postConn->finalizeUpdate(timed, dt);
      pvAssert(status == PV_SUCCESS);
   }

   needFinalize = false;
   return PV_SUCCESS;
}

int HyPerConn::reduce_dW(int arborId){
   int kernel_status = PV_BREAK;
   if(sharedWeights){
      kernel_status = reduceKernels(arborId); // combine partial changes in each column
      if(normalizeDwFlag){
         int activation_status = reduceActivations(arborId);
         pvAssert(kernel_status == activation_status);
      }
   }
   return kernel_status;
}

int HyPerConn::reduceActivations(int arborID){
   pvAssert(sharedWeights && plasticityFlag);
   Communicator * comm = parent->icCommunicator();
   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();
   const int nbProcs = comm->numCommBatches();
   const int nProcs = nxProcs * nyProcs * nbProcs;
   if(numKernelActivations && nProcs != 1){
      const MPI_Comm mpi_comm = comm->globalCommunicator();
      int ierr;
      const int numPatches = getNumDataPatches();
      const size_t patchSize = (size_t)nxp * (size_t)nyp * (size_t)nfp;
      const size_t localSize = numPatches * patchSize;
      const size_t arborSize = localSize * numberOfAxonalArborLists();
      ierr = MPI_Allreduce(MPI_IN_PLACE, get_activations(arborID), arborSize, MPI_LONG, MPI_SUM, mpi_comm);
   }

   return PV_BREAK;
}

int HyPerConn::reduceKernels(int arborID) {
   pvAssert(sharedWeights && plasticityFlag);
   Communicator * comm = parent->icCommunicator();
   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();
   const int nbProcs = comm->numCommBatches();
   const int nProcs = nxProcs * nyProcs * nbProcs;
   if (nProcs != 1){
      const MPI_Comm mpi_comm = comm->globalCommunicator();
      int ierr;
      const int numPatches = getNumDataPatches();
      const size_t patchSize = (size_t)nxp * (size_t)nyp * (size_t)nfp;
      const size_t localSize = numPatches * patchSize;
      const size_t arborSize = localSize * numberOfAxonalArborLists();
      ierr = MPI_Allreduce(MPI_IN_PLACE, get_dwDataStart(arborID), arborSize, MPI_FLOAT, MPI_SUM, mpi_comm);
   }

   return PV_BREAK;
}

int HyPerConn::calc_dW() {
   pvAssert(plasticityFlag);
   int status;
   for(int arborId=0;arborId<numberOfAxonalArborLists();arborId++) {
      status = initialize_dW(arborId);
      if (status==PV_BREAK) { break; }
      pvAssert(status == PV_SUCCESS);
   }
   for(int arborId=0;arborId<numberOfAxonalArborLists();arborId++) {
      status = update_dW(arborId);
      if (status==PV_BREAK) { break; }
      pvAssert(status == PV_SUCCESS);
   }
   for(int arborId=0;arborId<numberOfAxonalArborLists();arborId++) {
      status = reduce_dW(arborId);
      if (status==PV_BREAK) { break; }
      pvAssert(status == PV_SUCCESS);
   }
   if(normalizeDwFlag){
      for(int arborId=0;arborId<numberOfAxonalArborLists();arborId++) {
         status = normalize_dW(arborId);
         if (status==PV_BREAK) { break; }
         pvAssert(status == PV_SUCCESS);
      }
   }
   return status;
}

int HyPerConn::update_dW(int arbor_ID) {
   // compute dW but don't add them to the weights yet.
   // That takes place in reduceKernels, so that the output is
   // independent of the number of processors.
   int nExt = preSynapticLayer()->getNumExtended();
   const PVLayerLoc * loc = preSynapticLayer()->getLayerLoc();
   int nbatch = loc->nbatch;

   if(sharedWeights){
      //Calculate x and y cell size
      int xCellSize = zUnitCellSize(pre->getXScale(), post->getXScale());
      int yCellSize = zUnitCellSize(pre->getYScale(), post->getYScale());
      int nxExt = loc->nx + loc->halo.lt + loc->halo.rt;
      int nyExt = loc->ny + loc->halo.up + loc->halo.dn;
      int nf = loc->nf;
      int numKernels = getNumDataPatches();

      for(int b = 0; b < nbatch; b++){
         if(batchSkip[b]) continue;
         //Shared weights done in parallel, parallel in numkernels
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for(int kernelIdx = 0; kernelIdx < numKernels; kernelIdx++){

            //Calculate xCellIdx, yCellIdx, and fCellIdx from kernelIndex
            int kxCellIdx = kxPos(kernelIdx, xCellSize, yCellSize, nf);
            int kyCellIdx = kyPos(kernelIdx, xCellSize, yCellSize, nf);
            int kfIdx = featureIndex(kernelIdx, xCellSize, yCellSize, nf);
            //Loop over all cells in pre ext
            int kyIdx = kyCellIdx;
            int yCellIdx = 0;
            while(kyIdx < nyExt){
               int kxIdx = kxCellIdx;
               int xCellIdx = 0;
               while(kxIdx < nxExt){
                  //Calculate kExt from ky, kx, and kf
                  int kExt = kIndex(kxIdx, kyIdx, kfIdx, nxExt, nyExt, nf);
                  updateInd_dW(arbor_ID, b, kExt);
                  xCellIdx++;
                  kxIdx = kxCellIdx + xCellIdx * xCellSize;
               }
               yCellIdx++;
               kyIdx = kyCellIdx + yCellIdx * yCellSize;
            }
         }
      }
   }
   else{
      for(int b = 0; b < nbatch; b++){
         //Shared weights done in parallel, parallel in numkernels
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for(int kExt=0; kExt<nExt;kExt++) {
            updateInd_dW(arbor_ID, b, kExt);
         }
      }
   }

   //If update from clones, update dw here as well
   //Updates on all PlasticClones
   for(int clonei = 0; clonei < clones.size(); clonei++){
      pvAssert(clones[clonei]->preSynapticLayer()->getNumExtended() == nExt);
      for(int b = 0; b < parent->getNBatch(); b++){
         for(int kExt=0; kExt<nExt;kExt++) {
            clones[clonei]->updateInd_dW(arbor_ID, b, kExt);
         }
      }
   }

   return PV_SUCCESS;
}

int HyPerConn::updateInd_dW(int arbor_ID, int batch_ID, int kExt){
   const PVLayerLoc * postLoc = post->getLayerLoc();

   const pvdata_t * preactbufHead = preSynapticLayer()->getLayerData(getDelay(arbor_ID));
   const pvdata_t * postactbufHead = postSynapticLayer()->getLayerData();

   const pvdata_t * maskactbuf = NULL;
   if(useMask){
      const pvdata_t * maskactbufHead = mask->getLayerData();
      maskactbuf = maskactbufHead + batch_ID * mask->getNumExtended(); 
   }
   const pvdata_t * preactbuf = preactbufHead + batch_ID * preSynapticLayer()->getNumExtended();
   const pvdata_t * postactbuf = postactbufHead + batch_ID * postSynapticLayer()->getNumExtended();

   int sya = (post->getLayerLoc()->nf * (post->getLayerLoc()->nx + post->getLayerLoc()->halo.lt + post->getLayerLoc()->halo.rt));

   pvdata_t preact = preactbuf[kExt];
   if (skipPre(preact)) return PV_CONTINUE;

   PVPatch * weights = getWeights(kExt,arbor_ID);
   int ny = weights->ny;
   int nk = weights->nx * nfp;
   if (ny==0 || nk==0) { return PV_SUCCESS; }

   size_t offset = getAPostOffset(kExt, arbor_ID);
   const pvdata_t * postactRef = &postactbuf[offset];

   int sym = 0;
   const pvdata_t * maskactRef = NULL;
   if(useMask){
      const PVLayerLoc * maskLoc = mask->getLayerLoc();
      //Calculate mask offset, must account for different size margins and the num features
      //offsetX and Y are restricted indices into post
      size_t offsetX, offsetY;
      offsetX = kxPos(offset, postLoc->nx+postLoc->halo.lt+postLoc->halo.rt,
        postLoc->ny+postLoc->halo.up+postLoc->halo.dn,
        postLoc->nf) - postLoc->halo.lt;
      offsetY = kyPos(offset, postLoc->nx+postLoc->halo.lt+postLoc->halo.rt,
        postLoc->ny+postLoc->halo.up+postLoc->halo.dn,
        postLoc->nf) - postLoc->halo.up;
      //Sanity check, offset should be in restricted
      pvAssert(offsetX < postLoc->nx+postLoc->halo.lt);
      pvAssert(offsetY < postLoc->ny+postLoc->halo.up);
      //Convert to maskOffsetX and Y, extended (in mask)
      size_t maskOffsetX, maskOffsetY;
      maskOffsetX = offsetX + maskLoc->halo.lt;
      maskOffsetY = offsetY + maskLoc->halo.up;
      //Convert to extIndex into mask

      size_t maskOffset = kIndex(maskOffsetX, maskOffsetY, 0,
            maskLoc->nx+maskLoc->halo.lt+maskLoc->halo.rt,
            maskLoc->ny+maskLoc->halo.up+maskLoc->halo.dn,
            maskLoc->nf); //This should take into account if maskLoc's nf is either 1 or the size of post

      maskactRef = &maskactbuf[maskOffset];
      sym = (maskLoc->nf * (maskLoc->nx + maskLoc->halo.lt + maskLoc->halo.rt));
   }

   pvwdata_t * dwdata = get_dwData(arbor_ID, kExt);
   long * activations = NULL;
   if(sharedWeights && normalizeDwFlag){
      activations = get_activations(arbor_ID, kExt);
   }

   int lineoffsetw = 0;
   int lineoffseta = 0;
   int lineoffsetm = 0;
   for( int y=0; y<ny; y++ ) {
      for( int k=0; k<nk; k++ ) {
         pvdata_t aPost = postactRef[lineoffseta+k];
         //calculate contribution to dw unless masked out
         pvAssert(!useMask || maskactRef!=NULL); // if useMask is true, maskactRef must not be null
         float maskVal = 1;
         if(useMask){
            if(mask->getLayerLoc()->nf == 1){
               maskVal = maskactRef[lineoffsetm+((int)k/postLoc->nf)];
            }
            else{
               //If a maskFeatureIdx was specified
               if(maskFeatureIdx >= 0){
                  //k is an index into x/f space. Convert back to x space, and find the 0 feature index
                  int startingMaskK = ((int)k/postLoc->nf) * postLoc->nf;
                  //Offset into maskFeatureIdx
                  maskVal = maskactRef[lineoffsetm + startingMaskK + maskFeatureIdx];
               }
               else{
                  maskVal = maskactRef[lineoffsetm+k];
               }
            }
         }
         if (maskVal != 0) {
            //Note: this is a hack, as batching calls this function, but overwrites to allocate numKernelActivations with non-shared weights
            if(activations){
               //Offset in the case of a shrunken patch, where dwdata is applying when calling get_dwData
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

void HyPerConn::addClone(PlasticCloneConn* conn){
   //Make sure that the origional conn is indeed this
   pvAssert(conn->getOriginalConn() == this);
   clones.push_back(conn);
}

int HyPerConn::normalize_dW(int arbor_ID){
   //This is here in case other classes overwrite the outer class calling this function
   if(!normalizeDwFlag){
      return PV_SUCCESS;
   }
   if (sharedWeights) {
      pvAssert(numKernelActivations);
      int numKernelIndices = getNumDataPatches();
      for(int loop_arbor = 0; loop_arbor < numberOfAxonalArborLists(); loop_arbor++){
         // Divide by numKernelActivations in this timestep
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for( int kernelindex=0; kernelindex<numKernelIndices; kernelindex++ ) {
            //Calculate pre feature index from patch index
            int numpatchitems = nxp*nyp*nfp;
            pvwdata_t * dwpatchdata = get_dwDataHead(loop_arbor,kernelindex);
            long * activations = get_activationsHead(loop_arbor, kernelindex);
            for( int n=0; n<numpatchitems; n++ ) {
               long divisor = activations[n];

               //for(int i = 0; i < clones.size(); i++){
               //   long * cloneActivation = clones[i]->get_activationsHead(loop_arbor, kernelindex);
               //   divisor += cloneActivation[n];
               //}
               if(divisor != 0){
                  dwpatchdata[n] /= divisor;
               }
               else{
                  dwpatchdata[n] = 0;
               }
            }
         }
      }
   }
   //TODO: non-shared weights should divide by batch period if applicable
   return PV_BREAK;
}

pvdata_t HyPerConn::updateRule_dW(pvdata_t pre, pvdata_t post) {
   return dWMax * pre * post;
}


int HyPerConn::updateWeights(int arborId)
{
   // add dw to w
   for(int kArbor = 0; kArbor < numberOfAxonalArborLists(); kArbor++){
      pvwdata_t * w_data_start = get_wDataStart(kArbor);
      for( long int k=0; k<patchStartIndex(getNumDataPatches()); k++ ) {
         w_data_start[k] += get_dwDataStart(kArbor)[k];
      }
   }
   return PV_BREAK;
}

double HyPerConn::computeNewWeightUpdateTime(double time, double currentUpdateTime) {
   //Only called if plasticity flag is set
   if (!triggerLayer) {
      while(time >= weightUpdateTime){
         weightUpdateTime += weightUpdatePeriod;
      }
   }
   return weightUpdateTime;
}

PVPatch * HyPerConn::getWeights(int k, int arbor)
{
   // a separate arbor/patch of weights for every neuron
   return wPatches[arbor][k];
}

int HyPerConn::deliver() {
   int status = PV_SUCCESS;

   //Check if updating from post perspective
   HyPerLayer * pre = preSynapticLayer();
   PVLayerCube cube;
   memcpy(&cube.loc, pre->getLayerLoc(), sizeof(PVLayerLoc));
   cube.numItems = pre->getNumExtended();
   cube.size = sizeof(PVLayerCube);

   DataStore * store = parent->icCommunicator()->publisherStore(pre->getLayerId());
   int numArbors = numberOfAxonalArborLists();

   for (int arbor=0; arbor<numArbors; arbor++) {
      int delay = getDelay(arbor);
      cube.data = (pvdata_t *) store->buffer(0, delay); //First element is batch, but since it's a continuous buffer, 0 here is alright
      if(!getUpdateGSynFromPostPerspective()){
         cube.isSparse = store->isSparse();
         if(cube.isSparse){
            cube.numActive = store->numActiveBuffer(0, delay);
            cube.activeIndices = store->activeIndicesBuffer(0, delay);
         }
#if defined(PV_USE_CUDA)
         if(getReceiveGpu()){
            status = deliverPresynapticPerspectiveGPU(&cube, arbor);
            //No need to update GSyn since it's already living on gpu
            post->setUpdatedDeviceGSynFlag(false);
         }
         else
#endif
         {
            status = deliverPresynapticPerspective(&cube, arbor);
#if defined(PV_USE_CUDA)
            //CPU updated gsyn, need to update gsyn
            post->setUpdatedDeviceGSynFlag(true);
#endif
         }
      }
      else{
#if defined(PV_USE_CUDA)
         if(getReceiveGpu()){
            status = deliverPostsynapticPerspectiveGPU(&cube, arbor);
            //GSyn already living on GPU
            post->setUpdatedDeviceGSynFlag(false);
         }
         else
#endif
         {
            status = deliverPostsynapticPerspective(&cube, arbor);
#if defined(PV_USE_CUDA)
            //CPU updated gsyn, need to update on GPU
            post->setUpdatedDeviceGSynFlag(true);
#endif
         }
      }
      pvAssert(status == PV_SUCCESS || status == PV_BREAK);
      if (status == PV_BREAK){
         break; // Breaks out of arbor loop
      }
   }
   return PV_SUCCESS;
}

int HyPerConn::deliverPresynapticPerspective(PVLayerCube const * activity, int arborID) {
   //Check if we need to update based on connection's channel
   if(getChannel() == CHANNEL_NOUPDATE){
      return PV_SUCCESS;
   }
   pvAssert(post->getChannel(getChannel()));

   float dt_factor;
   if (getPvpatchAccumulateType()==ACCUMULATE_STOCHASTIC) {
      dt_factor = getParent()->getDeltaTime();
   }
   else {
      dt_factor = getConvertToRateDeltaTimeFactor();
   }

   if (_weightSparsity > 0.0f && !_sparseWeightsAllocated[arborID]) {
      allocateSparseWeightsPre(activity, arborID);
   }

   const PVLayerLoc * preLoc = preSynapticLayer()->getLayerLoc();
   const PVLayerLoc * postLoc = postSynapticLayer()->getLayerLoc();


   pvAssert(arborID >= 0);
   const int numExtended = activity->numItems;

   int nbatch = parent->getNBatch();

   for(int b = 0; b < nbatch; b++){
      pvdata_t * activityBatch = activity->data + b * (preLoc->nx + preLoc->halo.rt + preLoc->halo.lt) * (preLoc->ny + preLoc->halo.up + preLoc->halo.dn) * preLoc->nf;
      pvdata_t * gSynPatchHeadBatch = post->getChannel(getChannel()) + b * postLoc->nx * postLoc->ny * postLoc->nf;
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

         float a = activityBatch[kPreExt] * dt_factor;
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
         if (_weightSparsity <= 0.0f) {
            deliverOnePreNeuronActivity(kPreExt, arborID, a, gSynPatchHead, getRandState(kPreExt));
         } else {
            deliverOnePreNeuronActivitySparseWeights(kPreExt, arborID, a, gSynPatchHead, getRandState(kPreExt));
         }
      }
#ifdef PV_USE_OPENMP_THREADS
      //Accumulate back into gSyn // Should this be done in HyPerLayer where it can be done once, as opposed to once per connection?
      if(thread_gSyn){
         pvdata_t * gSynPatchHead = gSynPatchHeadBatch;
         int numNeurons = post->getNumNeurons();
         //Looping over neurons first to be thread safe
#pragma omp parallel for
         for(int ni = 0; ni < numNeurons; ni++){
            for(int ti = 0; ti < parent->getNumThreads(); ti++){
               gSynPatchHead[ni] += thread_gSyn[ti][ni];
            }
         }
      }
#endif
   }
   return PV_SUCCESS;
}


int HyPerConn::deliverPostsynapticPerspective(PVLayerCube const * activity, int arborID, int* numActive, int** activeList) {
   //Make sure numActive and activeList are either both null or both valid pointers
   if(numActive){
      assert(activeList);
   }
   else{
      assert(!activeList);
   }

   //Check channel number for noupdate
   if(getChannel() == CHANNEL_NOUPDATE){
      return PV_SUCCESS;
   }
   pvAssert(post->getChannel(getChannel()));

   pvAssert(arborID >= 0);
   //Get number of neurons restricted target
   const int numPostRestricted = post->getNumNeurons();

   float dt_factor;
   if (getPvpatchAccumulateType()==ACCUMULATE_STOCHASTIC) {
      dt_factor = getParent()->getDeltaTime();
   }
   else {
      dt_factor = getConvertToRateDeltaTimeFactor();
   }

   if (_weightSparsity > 0.0f && !_sparseWeightsAllocated[arborID]) {
      allocateSparseWeightsPost(activity, arborID);
   }

   const PVLayerLoc * sourceLoc = preSynapticLayer()->getLayerLoc();
   const PVLayerLoc * targetLoc = post->getLayerLoc();

   const int sourceNx = sourceLoc->nx;
   const int sourceNy = sourceLoc->ny;
   const int sourceNf = sourceLoc->nf;
   const int targetNx = targetLoc->nx;
   const int targetNy = targetLoc->ny;
   const int targetNf = targetLoc->nf;
   const int nbatch = targetLoc->nbatch;

   const PVHalo * sourceHalo = &sourceLoc->halo;
   const PVHalo * targetHalo = &targetLoc->halo;

   //get source layer's extended y stride
   int sy  = (sourceNx+sourceHalo->lt+sourceHalo->rt)*sourceNf;

   //The start of the gsyn buffer
   pvdata_t * gSynPatchHead = post->getChannel(getChannel());

   long * startSourceExtBuf = getPostToPreActivity();
   if(!startSourceExtBuf){
      std::cout << "HyPerLayer::recvFromPost error getting preToPostActivity from connection. Is shrink_patches on?\n";
      exit(EXIT_FAILURE);
   }

   //If numActive is a valid pointer, we're recv from post sparse
   bool recvPostSparse = numActive;

   for(int b = 0; b < nbatch; b++){
      int numLoop;
      if(recvPostSparse){
         numLoop = numActive[b];
      }
      else{
         numLoop = numPostRestricted;
      }
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
      for (int iLoop = 0; iLoop < numLoop; iLoop++){
         int kTargetRes;
         if(recvPostSparse){
            kTargetRes = activeList[b][iLoop];
         }
         else{
            kTargetRes = iLoop;
         }
         

         pvdata_t * activityBatch = activity->data + b * (sourceNx + sourceHalo->rt + sourceHalo->lt) * (sourceNy + sourceHalo->up + sourceHalo->dn) * sourceNf;
         pvdata_t * gSynPatchHeadBatch = gSynPatchHead + b * targetNx * targetNy * targetNf;
         //Change restricted to extended post neuron
         int kTargetExt = kIndexExtended(kTargetRes, targetNx, targetNy, targetNf, targetHalo->lt, targetHalo->rt, targetHalo->dn, targetHalo->up);

         //Read from buffer
         long startSourceExt = startSourceExtBuf[kTargetRes];

         //Calculate target's start of gsyn
         pvdata_t * gSynPatchPos = gSynPatchHeadBatch + kTargetRes;

         taus_uint4 * rngPtr = getRandState(kTargetRes);
         float* activityStartBuf = &(activityBatch[startSourceExt]); 

         if (_weightSparsity <= 0.0f) {
            deliverOnePostNeuronActivity(arborID, kTargetExt, sy, activityStartBuf, gSynPatchPos, dt_factor, rngPtr);
         } else {
            deliverOnePostNeuronActivitySparseWeights(arborID, kTargetExt, sy, activityStartBuf, gSynPatchPos, dt_factor, rngPtr);
         }
      }
   }
   return PV_SUCCESS;
}



#if defined(PV_USE_CUDA)
void HyPerConn::updateDeviceWeights(){
   //wDataStart is one big buffer, so this should grab everything
   float * h_weights = get_wDataStart(0);
#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer * d_weights = getDeviceWData();
#endif
   pvAssert(d_weights);
   d_weights->copyToDevice(h_weights);

   //Need barrier here?
   parent->getDevice()->syncDevice();

#if defined(PV_USE_CUDNN)
   //Set local sizes here
   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const PVLayerLoc * postLoc = post->getLayerLoc();
   //float preToPostScaleX = (float)postLoc->nx/((float)preLoc->nx);
   //float preToPostScaleY = (float)postLoc->ny/((float)preLoc->ny);
   //float preToPostScaleX = (float)preLoc->nx/((float)postLoc->nx);
   //float preToPostScaleY = (float)preLoc->ny/((float)postLoc->ny);

   pvAssert(cudnn_WData);
   cudnn_WData->permuteWeightsPVToCudnn(d_weights->getPointer(), numberOfAxonalArborLists(), getNumDataPatches(), nxp, nyp, nfp);
#endif
}

int HyPerConn::deliverPresynapticPerspectiveGPU(PVLayerCube const * activity, int arborID) {
   pvAssert(krRecvPre);
   //Check if we need to update based on connection's channel
   if(getChannel() == CHANNEL_NOUPDATE){
      return PV_SUCCESS;
   }
   pvAssert(post->getChannel(getChannel())); // pvAssert(GSyn && GSyn[conn->getChannel()]);

   float dt_factor;
   if (getPvpatchAccumulateType()==ACCUMULATE_STOCHASTIC) {
      dt_factor = getParent()->getDeltaTime();
   }
   else if (getPvpatchAccumulateType()==ACCUMULATE_CONVOLVE) {
      dt_factor = getConvertToRateDeltaTimeFactor();
   }
   else{
      std::cout << "Pooling accumulate not implemented for GPUs";
      exit(-1);
   }

#ifdef PV_USE_CUDA
   krRecvPre->set_dt_factor(dt_factor);
#endif // PV_USE_CUDA

   //Post layer receives synaptic input
   //Only with respect to post layer
   const PVLayerLoc * preLoc = preSynapticLayer()->getLayerLoc();
   const PVLayerLoc * postLoc = postSynapticLayer()->getLayerLoc();
   //If the connection uses gpu to receive, update all buffers

   //TODO see if you can avoid this step of transferring patches to gpu
   //Based on arborId
   //Other way would be to just allocate all arbors to gpu

   //If more than 1 arbor, need to update patches and GSynPatchStart.
   //If one arbor, done in allocatePreKernel in HyPerConn
   if(numberOfAxonalArborLists() > 1){
      PVPatch* h_patches = weights(arborID)[0]; //0 because it's one block of memory
#ifdef PV_USE_CUDA
      PVCuda::CudaBuffer * d_patches = getDevicePatches();
#endif
      pvAssert(d_patches);

      d_patches->copyToDevice(h_patches);

      size_t* h_GSynPatchStart = getGSynPatchStart()[arborID];
#ifdef PV_USE_CUDA
      PVCuda::CudaBuffer * d_GSynPatchStart = getDeviceGSynPatchStart();
#endif
      pvAssert(d_GSynPatchStart);
      d_GSynPatchStart->copyToDevice(h_GSynPatchStart);
   }

   //Update pre datastore, post gsyn, and conn weights
   //Only if their updated
   if(preSynapticLayer()->getUpdatedDeviceDatastoreFlag()){
      float * h_preDatastore= activity->data;
#ifdef PV_USE_CUDA
      PVCuda::CudaBuffer * d_preDatastore = preSynapticLayer()->getDeviceDatastore();
#endif
      pvAssert(d_preDatastore);
      d_preDatastore->copyToDevice(h_preDatastore);

      //Copy active indices and num active if needed
      if(activity->isSparse){
#ifdef PV_USE_CUDA
         PVCuda::CudaBuffer * d_ActiveIndices;
         PVCuda::CudaBuffer * d_numActive;
#endif
         d_ActiveIndices = preSynapticLayer()->getDeviceActiveIndices();
         d_numActive = preSynapticLayer()->getDeviceNumActive();
         pvAssert(d_ActiveIndices);
         unsigned int * h_ActiveIndices = activity->activeIndices;
         long * h_numActive = activity->numActive;
         pvAssert(h_ActiveIndices);
         d_numActive->copyToDevice(h_numActive);
         d_ActiveIndices->copyToDevice(h_ActiveIndices);
      }
      //Device now has updated
      preSynapticLayer()->setUpdatedDeviceDatastoreFlag(false);
   }

   //X direction is active neuron
   //Y direction is post patch size
   long totActiveNeuron[parent->getNBatch()];
   long maxTotalActiveNeuron = 0;
   for(int b = 0; b < parent->getNBatch(); b++){
      if(activity->isSparse){
         totActiveNeuron[b] = activity->numActive[b];
      }
      else{
         totActiveNeuron[b] = preSynapticLayer()->getNumExtended();
      }
      if(totActiveNeuron[b] > maxTotalActiveNeuron){
         maxTotalActiveNeuron = totActiveNeuron[b];
      }
   }

   long totPatchSize = xPatchSize() * yPatchSize() * fPatchSize();

   long totThreads = maxTotalActiveNeuron * totPatchSize;

#ifdef PV_USE_CUDA
   //krRecvPre->set_numActive(totActiveNeuron);

   int maxThreads = parent->getDevice()->get_max_threads();
   int numLocalThreads = totPatchSize < maxThreads ? totPatchSize : maxThreads;

   krRecvPre->run_nocheck(totThreads, numLocalThreads);
#endif

   return PV_SUCCESS;
}

int HyPerConn::deliverPostsynapticPerspectiveGPU(PVLayerCube const * activity, int arborID) {

   //Check channel number for noupdate
   if(getChannel() == CHANNEL_NOUPDATE){
      return PV_SUCCESS;
   }
   pvAssert(post->getChannel(getChannel()));

   pvAssert(arborID >= 0);
   //Get number of neurons restricted target
   const int numRestricted = post->getNumNeurons();

   float dt_factor;
   if (getPvpatchAccumulateType()==ACCUMULATE_STOCHASTIC) {
      dt_factor = getParent()->getDeltaTime();
   }
   else if (getPvpatchAccumulateType()==ACCUMULATE_CONVOLVE) {
      dt_factor = getConvertToRateDeltaTimeFactor();
   }
   else{
      std::cout << "Pooling accumulate not implemented for GPUs";
      exit(-1);
   }

   pvAssert(krRecvPost);
#ifdef PV_USE_CUDA
   krRecvPost->set_dt_factor(dt_factor);
#endif // PV_USE_CUDA

   const PVLayerLoc * sourceLoc = pre->getLayerLoc();
   const PVLayerLoc * targetLoc = post->getLayerLoc();
   const PVHalo * sourceHalo = &sourceLoc->halo;

   const int sourceNx = sourceLoc->nx;
   const int sourceNy = sourceLoc->ny;
   const int sourceNf = sourceLoc->nf;
   const int targetNx = targetLoc->nx;
   const int targetNy = targetLoc->ny;
   const int targetNf = targetLoc->nf;

   //get source layer's extended y stride
   int sy  = (sourceNx+sourceHalo->rt+sourceHalo->lt)*sourceNf;
   //get source layer's patch y stride
   int syp = postConn->yPatchStride();
   //Iterate through y patch
   int numPerStride = postConn->xPatchSize() * postConn->fPatchSize();

   long * startSourceExtBuf = getPostToPreActivity();
   if(!startSourceExtBuf){
      std::cout << "HyPerLayer::recvFromPost error getting preToPostActivity from connection. Is shrink_patches on?\n";
      exit(EXIT_FAILURE);
   }

   bool updatePreAct = false;
   //Update pre activity, post gsyn, and conn weights
   //Only if they're updated
   if(pre->getUpdatedDeviceDatastoreFlag()){
      float * h_preDatastore = activity->data;
#ifdef PV_USE_CUDA
      PVCuda::CudaBuffer* d_preDatastore = pre->getDeviceDatastore();
#endif
      pvAssert(d_preDatastore);
      d_preDatastore->copyToDevice(h_preDatastore);
      //Device now has updated
      pre->setUpdatedDeviceDatastoreFlag(false);
      updatePreAct = true;
   }

#if defined(PV_USE_CUDA) && defined(PV_USE_CUDNN)
   //Permutation buffer is local to the kernel, NOT the layer
   //Therefore, we must permute Datastore every time
   krRecvPost->permuteDatastorePVToCudnn();
   //}
   
   //Permute GSyn
   krRecvPost->permuteGSynPVToCudnn(getChannel());
#endif // defined(PV_USE_CUDA) && defined(PV_USE_CUDNN)
//#endif // PV_USE_CUDA

   int totF = targetNf;
   int totX = targetNx;
   int totY = targetNy;
   //Make sure local sizes are divisible by f, x, and y
#ifdef PV_USE_CUDA
   krRecvPost->run(totX, totY, totF, getNumXLocal(), getNumYLocal(), getNumFLocal());
#endif

#if defined(PV_USE_CUDA) && defined(PV_USE_CUDNN)
   krRecvPost->permuteGSynCudnnToPV(getChannel());
#endif

   return PV_SUCCESS;
}
#endif // defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)


void HyPerConn::deliverOnePostNeuronActivity(int arborID, int kTargetExt, int inSy, float* activityStartBuf, pvdata_t* gSynPatchPos, float dt_factor, taus_uint4 * rngPtr){
   //get source layer's patch y stride
   int syp = postConn->yPatchStride();
   int yPatchSize = postConn->yPatchSize();
   //Iterate through y patch
   int numPerStride = postConn->xPatchSize() * postConn->fPatchSize();
   int kernelIndex = postConn->patchToDataLUT(kTargetExt);

   pvwdata_t* weightStartBuf = postConn->get_wDataHead(arborID, kernelIndex);
   int sf = 1;
   int offset = 0;
   for (int ky = 0; ky < yPatchSize; ky++){
      float * activityY = &(activityStartBuf[ky*inSy+offset]);
      pvwdata_t * weightY = weightStartBuf + ky*syp;
      //TODO add sf here
      (accumulateFunctionFromPostPointer)(0, numPerStride, gSynPatchPos, activityY, weightY, dt_factor, rngPtr, sf);
   }
}

void HyPerConn::deliverOnePreNeuronActivity(int kPreExt, int arbor, pvadata_t a, pvgsyndata_t * postBufferStart, void * auxPtr) {
   PVPatch * weights = getWeights(kPreExt, arbor);
   const int nk = weights->nx * fPatchSize();
   const int ny = weights->ny;
   const int sy  = getPostNonextStrides()->sy;       // stride in layer
   const int syw = yPatchStride();                   // stride in patch
   pvwdata_t * weightDataStart = NULL; 
   pvgsyndata_t * postPatchStart = postBufferStart + getGSynPatchStart(kPreExt, arbor);
   int offset = 0;
   int sf = 1;
     weightDataStart = get_wData(arbor,kPreExt); // make this a pvwdata_t const *?
     for (int y = 0; y < ny; y++) {
       (accumulateFunctionPointer)(0, nk, postPatchStart + y*sy + offset, a, weightDataStart + y*syw + offset, auxPtr, sf);
     }
}

int HyPerConn::createWeights(PVPatch *** patches, int nWeightPatches, int nDataPatches, int nxPatch,
      int nyPatch, int nfPatch, int arborId)
{
   // could create only a single patch with following call
   //   return createPatches(numAxonalArborLists, nxp, nyp, nfp);

   pvAssert(patches[arborId] == NULL);

   if (shrinkPatches_flag || arborId == 0){
      patches[arborId] = createPatches(nWeightPatches, nxPatch, nyPatch);
      pvAssert(patches[arborId] != NULL);
   }
   else{
      patches[arborId] = patches[0];
   }

   // allocate space for all weights at once (inplace), return pointer to beginning of weight array
   //pvdata_t * data_patches = allocWeights(patches, nDataPatches, nxPatch, nyPatch, nfPatch, arborId);
   return PV_SUCCESS;
}

float HyPerConn::getConvertToRateDeltaTimeFactor()
{
   float dt_factor = 1.0f;
   // if (preActivityIsNotRate) { // preActivityIsNotRate was replaced with convertRateToSpikeCount on Dec 31, 2014
   if (convertRateToSpikeCount && !pre->activityIsSpiking()) {
      enum ChannelType channel_type = getChannel();
      float dt = getParent()->getDeltaTime();
      float tau = post->getChannelTimeConst(channel_type);
      if (tau > 0) {
         double exp_dt_tau = exp(-dt / tau);
         dt_factor = (1 - exp_dt_tau) / exp_dt_tau;
         // the above factor was chosen so that for a constant input of G_SYN to an excitatory conductance G_EXC,
         // then G_EXC -> G_SYN as t -> inf
      }
      else {
         dt_factor = dt;
      }
   }
   return dt_factor;
}

/**
 * Create a separate patch of weights for every neuron
 */
int HyPerConn::createWeights(PVPatch *** patches, int arborId)
{
   int nWeightPatches = getNumWeightPatches();
   int nDataPatches = getNumDataPatches();
   int nxPatch = nxp;
   int nyPatch = nyp;
   int nfPatch = nfp;

   int status = createWeights(patches, nWeightPatches, nDataPatches, nxPatch, nyPatch, nfPatch, arborId);
   return status;
}

int HyPerConn::clearWeights(pvwdata_t ** dataStart, int numPatches, int nxp, int nyp, int nfp) {
   int status = PV_SUCCESS;
   for( int arborID = 0; arborID<numAxonalArborLists; arborID++ ) {
      if( clearWeights(dataStart[arborID], numPatches, nxp, nyp, nfp)!=PV_SUCCESS ) status = PV_FAILURE;
   }
   return status;
}

int HyPerConn::clearWeights(pvwdata_t * arborDataStart, int numPatches, int nxp, int nyp, int nfp) {
   for( long int w=0; w<patchStartIndex(numPatches); w++ ) {
      arborDataStart[w] = 0.0f;
   }
   return PV_SUCCESS;
}

int HyPerConn::deleteWeights() {
   // to be used if createPatches is used above
   // HyPerConn::deletePatches(numAxonalArborLists, wPatches);
   if (wPatches != NULL) {
      for (int arbor = 0; arbor < numAxonalArborLists; arbor++) {
         if (wPatches[arbor] != NULL) {
            if (shrinkPatches_flag || arbor == 0) {
               deletePatches(wPatches[arbor]);
            }
            wPatches[arbor] = NULL;
         }
      }  // arbor
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
      }  // arbor
      free(wDataStart);
      wDataStart = NULL;
      if (!combine_dW_with_W_flag) {
         free(dwDataStart);
      }
      dwDataStart = NULL;
   } // wDataStart != NULL

   if(numKernelActivations != NULL){
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
   }  // wPostPatches != NULL

   if (gSynPatchStart != NULL) {
      free(gSynPatchStart[0]); // All gSynPatchStart[k]'s were allocated together in a single malloc call.
      free(gSynPatchStart);
   }
   if (aPostOffset != NULL) {
      free(aPostOffset[0]); // All aPostOffset[k]'s were allocated together in a single malloc call.
      free(aPostOffset);
   }
   free(patch2datalookuptable); patch2datalookuptable = NULL;

   return PV_SUCCESS;
}


//This function is doing what adjust axonal arbors was doing before, but generalized to not use the pre/post layer's pre/post for use with gpu post groups
int HyPerConn::adjustAllPatches(
      int nxPre, int nyPre, int nfPre, const PVHalo * haloPre,
      int nxPost, int nyPost, int nfPost, const PVHalo * haloPost,
      PVPatch*** inWPatches,
      size_t** inGSynPatchStart,
      size_t** inAPostOffset,
      int arborId){

   const int xScaleDiff = pre->getXScale() - post->getXScale();
   const int xPostNeuronsPerPreNeuron = xScaleDiff > 0 ? (int) pow(2,xScaleDiff) : 1;
   const int xPreNeuronsPerPostNeuron = xScaleDiff < 0 ? (int) pow(2,-xScaleDiff) : 1;

   const int yScaleDiff = pre->getYScale() - post->getYScale();
   const int yPostNeuronsPerPreNeuron = yScaleDiff > 0 ? (int) pow(2,yScaleDiff) : 1;
   const int yPreNeuronsPerPostNeuron = yScaleDiff < 0 ? (int) pow(2,-yScaleDiff) : 1;


   // can't use member variable numWeightPatches because GPUs might call this routine with a smaller block.  Calculate from input arguments
   int numWeightPatches = (nxPre + haloPre->lt + haloPre->rt) * (nyPre+haloPre->up + haloPre->dn) * nfPre; 
   for (int kex=0; kex<numWeightPatches; kex++) {
      // calculate start of patch in postsynaptic restricted coordinates, and width of patch in postsynaptic restricted coordinates
      int xPre = kxPos(kex, nxPre+haloPre->lt+haloPre->rt, nyPre+haloPre->dn+haloPre->up, nfPre)-haloPre->lt; // x-coordinate of presynaptic neuron tied to patch kex, in presynaptic restricted coordinates.
      int xPostStart, xPatchStart, nxPatch;
      int status = adjustedPatchDimension(xPre, xPreNeuronsPerPostNeuron, xPostNeuronsPerPreNeuron, nxPost, nxp, &xPostStart, &xPatchStart, &nxPatch);
      int yPre = kyPos(kex, nxPre+haloPre->lt+haloPre->rt, nyPre+haloPre->dn+haloPre->up, nfPre)-haloPre->up; // y-coordinate of presynaptic neuron tied to patch kex, in presynaptic restricted coordinates.
      int yPostStart, yPatchStart, nyPatch;
      status = adjustedPatchDimension(yPre, yPreNeuronsPerPostNeuron, yPostNeuronsPerPreNeuron, nyPost, nyp, &yPostStart, &yPatchStart, &nyPatch);

      if(inAPostOffset){
         inAPostOffset[arborId][kex] = (size_t) kIndex(xPostStart+haloPost->lt,yPostStart+haloPost->up,0,nxPost+haloPost->lt+haloPost->rt,nyPost+haloPost->dn+haloPost->up,nfPost);
      }

      inGSynPatchStart[arborId][kex] = (size_t) kIndex(xPostStart,yPostStart,0,nxPost,nyPost,nfPost);

      PVPatch * w = inWPatches[arborId][kex];
      pvAssert(w->offset==0);
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
 *      - Each axonal arbor (PVAxonalArbor) connects to a patch of neurons in the post-synaptic layer.
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
int HyPerConn::adjustAxonalArbors(int arborId)
{

   const int nxPre = pre->getLayerLoc()->nx;
   const int nyPre = pre->getLayerLoc()->ny;
   const int nfPre = pre->getLayerLoc()->nf;
   const PVHalo * haloPre = &pre->getLayerLoc()->halo;
   const int nxPost = post->getLayerLoc()->nx;
   const int nyPost = post->getLayerLoc()->ny;
   const int nfPost = post->getLayerLoc()->nf;
   const PVHalo * haloPost = &post->getLayerLoc()->halo;

   return adjustAllPatches(nxPre, nyPre, nfPre, haloPre, nxPost, nyPost, nfPost, haloPost, wPatches, gSynPatchStart, aPostOffset, arborId);
}

PVPatch *** HyPerConn::convertPreSynapticWeights(double time)
{
   if (time <= wPostTime) {
      return wPostPatches;
   }
   wPostTime = time;

   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const PVLayerLoc * postLoc = post->getLayerLoc();

   const int xScale = post->getXScale() - pre->getXScale();
   const int yScale = post->getYScale() - pre->getYScale();
   const double powXScale = pow(2.0f, (double) xScale);
   const double powYScale = pow(2.0f, (double) yScale);

   // pre-synaptic weights are in extended layer reference frame
   const int nxPre = preLoc->nx + preLoc->halo.lt + preLoc->halo.rt;
   const int nyPre = preLoc->ny + preLoc->halo.dn + preLoc->halo.up;
   const int nfPre = preLoc->nf;

   const int nxPost  = postLoc->nx;
   const int nyPost  = postLoc->ny;
   const int nfPost  = postLoc->nf;
   const int numPost = post->getNumNeurons();

   nxpPost = (int) (nxp * powXScale);
   nypPost = (int) (nyp * powYScale);
   nfpPost = preLoc->nf;

   int sxPost = nfpPost;
   int syPost = sxPost * nxpPost;
   int spPost = syPost * nypPost;

   // the number of features is the end-point value (normally post-synaptic)
   const int numPostPatch = nxpPost * nypPost * nfpPost; // Post-synaptic weights are never shrunken

   if (wPostPatches == NULL) {
      wPostPatches = (PVPatch***) pvCalloc(numAxonalArborLists, sizeof(PVPatch**));
      pvAssert(wPostDataStart == NULL);
      //TODO-CER-2014.4.3 - is the sizeof part correct??????????????????
      //PFS-2014.6.4 - This looks correct; it's of the form "foo * x = (foo *) calloc(numfoos, sizeof(foo))"
      wPostDataStart = (pvwdata_t **) pvCalloc(numAxonalArborLists, sizeof(pvwdata_t *));
      wPostDataStart[0] = allocWeights(numPost, nxpPost, nypPost, nfpPost);
      pvAssert(wPostDataStart[0] != NULL);
      for(int arborID=0;arborID<numberOfAxonalArborLists();arborID++) {
         int status = createWeights(wPostPatches, numPost, numPost, nxpPost, nypPost, nfpPost, arborID);
         pvAssert(status == PV_SUCCESS);
         if (arborID > 0){  // wDataStart already allocated
            wPostDataStart[arborID] = wPostDataStart[0] + spPost * numPost * arborID;
            pvAssert(wPostDataStart[arborID] != NULL);
         }
      }
   }

   //loop through all axons:
   for(int arborID=0;arborID<numberOfAxonalArborLists();arborID++) {

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

         pvwdata_t * postData = wPostDataStart[arborID] + nxpPost*nypPost*nfpPost*kPost + wPostPatches[arborID][kPost]->offset;
         for (int kp = 0; kp < numPostPatch; kp++) {

            // calculate extended indices of presynaptic neuron {kPre, kzPre}
            int kxPostPatch = (int) kxPos(kp, nxpPost, nypPost, nfPre);
            int kyPostPatch = (int) kyPos(kp, nxpPost, nypPost, nfPre);
            int kfPostPatch = (int) featureIndex(kp, nxpPost, nypPost, nfPre);

            int kxPre = kxPreHead + kxPostPatch;
            int kyPre = kyPreHead + kyPostPatch;
            int kfPre = kfPostPatch;
            int kPre = kIndex(kxPre, kyPre, kfPre, nxPre, nyPre, nfPre);

            // if {kPre, kzPre} out of bounds, set post weight to zero
            if (kxPre < 0 || kyPre < 0 || kxPre >= nxPre || kyPre >= nyPre) {
               pvAssert(kxPre < 0 || kyPre < 0 || kxPre >= nxPre || kyPre >= nyPre);
               postData[kp] = 0.0; // wPostPatches[arborID][kPost]->data[kp] = 0.0;
            }
            else {
               // {kzPostHead} store the restricted indices of the postsynaptic patch head
               int kxPostHead, kyPostHead, kfPostHead;
               int nxp_post, nyp_post;  // shrunken patch dimensions
               int dx_nxp, dy_nyp;  // shrinkage

               postSynapticPatchHead(kPre, &kxPostHead, &kyPostHead, &kfPostHead, &dx_nxp,
                                        &dy_nyp,  &nxp_post,   &nyp_post);

               int kxPrePatch, kyPrePatch; // relative index in shrunken patch
               kxPrePatch = kxPost - kxPostHead;
               kyPrePatch = kyPost - kyPostHead;
               int kPrePatch = kfPost * sfp + kxPrePatch * sxp + kyPrePatch * syp;
               pvwdata_t * preData = get_wDataStart(arborID) + patchStartIndex(kPre) + getWeights(kPre,arborID)->offset;
               postData[kp] = preData[kPrePatch];
            }
         }
      }
   }
   return wPostPatches;
}

PVPatch **** HyPerConn::point2PreSynapticWeights()
{

   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const PVLayerLoc * postLoc = post->getLayerLoc();

   const int xScale = post->getXScale() - pre->getXScale();
   const int yScale = post->getYScale() - pre->getYScale();
   const double powXScale = pow(2.0f, (double) xScale);
   const double powYScale = pow(2.0f, (double) yScale);

   // pre-synaptic weights are in extended layer reference frame
   const int nxPre = preLoc->nx + preLoc->halo.lt + preLoc->halo.rt;
   const int nyPre = preLoc->ny + preLoc->halo.dn + preLoc->halo.up;
   const int nfPre = preLoc->nf;

   const int nxPost  = postLoc->nx;
   const int nyPost  = postLoc->ny;
   const int nfPost  = postLoc->nf;
   const int numPost = post->getNumNeurons();

   nxpPost = (int) (nxp * powXScale);
   nypPost = (int) (nyp * powYScale);
   nfpPost = preLoc->nf;
   pvwdata_t z = 0;

   // the number of features is the end-point value (normally post-synaptic)
   const int numPostPatch = nxpPost * nypPost * nfpPost; // Post-synaptic weights are never shrunken

   if (wPostPatchesp == NULL) {

      //Return data structure
      wPostPatchesp = (PVPatch****) pvCalloc(numAxonalArborLists, sizeof(PVPatch***));
      pvAssert(wPostDataStartp == NULL);
      wPostDataStartp = (pvwdata_t ***) pvCalloc(numAxonalArborLists, sizeof(pvwdata_t **));

      for(int arborID=0;arborID<numberOfAxonalArborLists();arborID++) {

         wPostPatchesp[arborID] = (PVPatch***) pvCalloc(numPost, sizeof(PVPatch**));

         int sx = nfpPost;
         int sy = sx * nxpPost;
         int sp = sy * nypPost;

         size_t patchSize = sp * sizeof(pvwdata_t);
         size_t dataSize = numPost * patchSize;

         wPostDataStartp[arborID] = (pvwdata_t **) pvCalloc(dataSize, sizeof(char*));


         PVPatch** patcharray = (PVPatch**) pvCalloc(numPost, sizeof(PVPatch*));
         PVPatch ** curpatch = patcharray;
         for (int i = 0; i < numPost; i++) {
            wPostPatchesp[arborID][i] = curpatch;
            curpatch++;
         }
      }
   }

   //loop through all arbors:
   for(int arborID=0;arborID<numberOfAxonalArborLists();arborID++) {

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

         //Accessing by patch offset through wPostDataStart by x,y,and feature of a patch
         pvwdata_t ** postData = wPostDataStartp[arborID] + nxpPost*nypPost*nfpPost*kPost + 0;
         for (int kp = 0; kp < numPostPatch; kp++) {

            // calculate extended indices of presynaptic neuron {kPre, kzPre}
            int kxPostPatch = (int) kxPos(kp, nxpPost, nypPost, nfPre);
            int kyPostPatch = (int) kyPos(kp, nxpPost, nypPost, nfPre);
            int kfPostPatch = (int) featureIndex(kp, nxpPost, nypPost, nfPre);

            int kxPre = kxPreHead + kxPostPatch;
            int kyPre = kyPreHead + kyPostPatch;
            int kfPre = kfPostPatch;
            int kPre = kIndex(kxPre, kyPre, kfPre, nxPre, nyPre, nfPre);

            // if {kPre, kzPre} out of bounds, set post weight to zero
            if (kxPre < 0 || kyPre < 0 || kxPre >= nxPre || kyPre >= nyPre) {
               pvAssert(kxPre < 0 || kyPre < 0 || kxPre >= nxPre || kyPre >= nyPre);
               postData[kp] = &z; // wPostPatches[arborID][kPost]->data[kp] = 0.0;
            }
            else {
               // {kzPostHead} store the restricted indices of the postsynaptic patch head
               int kxPostHead, kyPostHead, kfPostHead;
               int nxp_post, nyp_post;  // shrunken patch dimensions
               int dx_nxp, dy_nyp;  // shrinkage

               postSynapticPatchHead(kPre, &kxPostHead, &kyPostHead, &kfPostHead, &dx_nxp,
                                        &dy_nyp,  &nxp_post,   &nyp_post);

               //pvAssert(nxp_post == p->nx);
               //pvAssert(nyp_post == p->ny);
               //pvAssert(nfp == lPost->loc.nf);

               int kxPrePatch, kyPrePatch; // relative index in shrunken patch
               kxPrePatch = kxPost - kxPostHead;
               kyPrePatch = kyPost - kyPostHead;
               int kPrePatch = kfPost * sfp + kxPrePatch * sxp + kyPrePatch * syp;
               pvwdata_t * preData = get_wDataStart(arborID) + patchStartIndex(kPre) + getWeights(kPre,arborID)->offset;
               postData[kp] = &(preData[kPrePatch]);
               // wPostPatches[arborID][kPost]->data[kp] = preData[kPrePatch];

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
int HyPerConn::preSynapticPatchHead(int kxPost, int kyPost, int kfPost, int * kxPre, int * kyPre)
{
   int status = 0;

   const int xScale = post->getXScale() - pre->getXScale();
   const int yScale = post->getYScale() - pre->getYScale();
   const double powXScale = pow(2, (double) xScale);
   const double powYScale = pow(2, (double) yScale);

   const int nxPostPatch = (int) (nxp * powXScale);
   const int nyPostPatch = (int) (nyp * powYScale);

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
int HyPerConn::postSynapticPatchHead(int kPreEx,
                                     int * kxPostOut, int * kyPostOut, int * kfPostOut,
                                     int * dxOut, int * dyOut, int * nxpOut, int * nypOut)
{
   int status = 0;

   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const PVLayerLoc * postLoc = post->getLayerLoc();

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
   int dx = 0;
   int dy = 0;
   int nxPatch = nxp;
   int nyPatch = nyp;

   if (kxPost < 0) {
      nxPatch -= -kxPost;
      kxPost = 0;
      if (nxPatch < 0) nxPatch = 0;
      dx = nxp - nxPatch;
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
      if (nyPatch < 0) nyPatch = 0;
      dy = nyp - nyPatch;
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
      dx = 0;
      dy = 0;
      nxPatch = 0;
      nyPatch = 0;
      logError("HyPerConn::postSynapticPatchHead: WARNING patch size is zero\n");
   }

   *kxPostOut = kxPost;
   *kyPostOut = kyPost;
   *kfPostOut = kfPost;

   *dxOut = dx;
   *dyOut = dy;
   *nxpOut = nxPatch;
   *nypOut = nyPatch;

   return status;
}

int HyPerConn::writePostSynapticWeights(double timef, bool last) {
   int status = PV_SUCCESS;
   char path[PV_PATH_MAX];

   const PVLayerLoc * preLoc = pre->getLayerLoc();

   float minVal = FLT_MAX;
   float maxVal = -FLT_MAX;
   for(int arbor=0; arbor<numberOfAxonalArborLists(); arbor++) {
      float minVal1 = minWeight(arbor);
      if( minVal1 < minVal ) minVal = minVal1;
      float maxVal1 = maxWeight(arbor);
      if( maxVal1 > maxVal ) maxVal = maxVal1;
   }

   const int numPostPatches = post->getNumNeurons();

   const int xScale = post->getXScale() - pre->getXScale();
   const int yScale = post->getYScale() - pre->getYScale();
   const double powXScale = pow(2, (double) xScale);
   const double powYScale = pow(2, (double) yScale);

   const int nxPostPatch = (int) (nxp * powXScale);
   const int nyPostPatch = (int) (nyp * powYScale);
   const int nfPostPatch = preLoc->nf;

   const char * last_str = (last) ? "_last" : "";

   int chars_needed = 0;
   pvAssert(parent->includeConnectionName()<=2 && parent->includeConnectionName()>=0);
   switch(parent->includeConnectionName()) {
   case 0:
      chars_needed = snprintf(path, PV_PATH_MAX-1, "%s/w%d_post%s.pvp", parent->getOutputPath(), getConnectionId(), last_str);
      break;
   case 1:
      chars_needed = snprintf(path, PV_PATH_MAX-1, "%s/w%d_%s_post%s.pvp", parent->getOutputPath(), getConnectionId(), name, last_str);
      break;
   case 2:
      chars_needed = snprintf(path, PV_PATH_MAX-1, "%s/%s_post%s.pvp", parent->getOutputPath(), name, last_str);
      break;
   default:
      pvAssert(0);
      break;
   }

   const PVLayerLoc * postLoc = post->getLayerLoc();
   Communicator * comm = parent->icCommunicator();

   bool append = (last) ? false : ioAppend;

   status = PV::writeWeights(path, comm, (double) timef, append, postLoc, preLoc,
        nxPostPatch, nyPostPatch, nfPostPatch, minVal, maxVal, wPostPatches,
        wPostDataStart, numPostPatches, numberOfAxonalArborLists(),
        writeCompressedWeights, fileType);

   if(status != PV_SUCCESS) {
      fflush(stdout);
      logError("Connection \"%s\": writePostSynapticWeights failed at time %f.  Exiting.\n", name, timef);
      abort();
   }

   return PV_SUCCESS;
}

int HyPerConn::sumWeights(int nx, int ny, int offset, pvwdata_t * dataStart, double * sum, double * sum2, pvdata_t * maxVal)
{
   // TODO CER - should make volatile conditional on GPU usage (this could be slow otherwise)?
   volatile pvwdata_t * w = dataStart + offset;
   double sum_tmp = 0;
   double sum2_tmp = 0;
   pvdata_t max_tmp = -FLT_MAX;
   for (int ky = 0; ky < ny; ky++) {
      for(int iWeight = 0; iWeight < syp; iWeight++ ){
         sum_tmp += w[iWeight];
         sum2_tmp += w[iWeight] * w[iWeight];
         max_tmp = ( max_tmp > w[iWeight] ) ? max_tmp : w[iWeight];
      }
      w += syp;
   }
   *sum = sum_tmp;
   *sum2 = sum2_tmp;
   *maxVal = max_tmp;
   return PV_SUCCESS;
} // sumWeights


int HyPerConn::checkPatchDimensions() {
   int statusx = checkPatchSize(nxp, pre->getXScale(), post->getXScale(), 'x');
   int statusy = checkPatchSize(nyp, pre->getYScale(), post->getYScale(), 'y');
   int status = statusx==PV_SUCCESS && statusy==PV_SUCCESS ? PV_SUCCESS : PV_FAILURE;
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
      int scaleFactor = (int) pow(2, (double) scaleDiff);
      int multipleOfScaleFactor = patchSize/scaleFactor;
      goodsize = multipleOfScaleFactor>0 && patchSize == multipleOfScaleFactor*scaleFactor;
   }
   else {
      pvAssert(scaleDiff < 0);
      // any patch size is allowed
      goodsize = true;
   }
   if( !goodsize ) {
      logError("Error:  Connection: %s\n",name);
      logError("Presynaptic layer:  %s\n", pre->getName());
      logError("Postsynaptic layer: %s\n", post->getName());
      logError("Patch size n%cp=%d is not compatible with presynaptic n%cScale %f\n",
              dim,patchSize,dim,pow(2,-scalePre));
      logError("and postsynaptic n%cScale %f.\n",dim,pow(2,-scalePost));
      if (scaleDiff ==0) {
         logError("(presynaptic scale) == (postsynaptic scale);\n");
         logError("therefore patch size must be odd\n");
      }
      if (scaleDiff > 0) {
         int scaleFactor = (int) pow(2, (float) scaleDiff);
         logError("(postsynaptic scale) = %d * (presynaptic scale);\n", scaleFactor);
         logError("therefore compatible sizes are multiples of %d.\n", scaleFactor);
      }
      else {
         // If scaleDiff < 0 any patch size is acceptable
         pvAssert(0);
      }
      logError("Exiting.\n");
      exit(1);
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
   postExtStrides.sy = nfp * (post->getLayerLoc()->nx+post->getLayerLoc()->halo.lt+post->getLayerLoc()->halo.rt);

   return PV_SUCCESS;
}

pvwdata_t * HyPerConn::allocWeights(int nPatches, int nxPatch, int nyPatch, int nfPatch)
{
   bool overflow = false; // Do sanity checking on the size of the weight allocation.

   int sx = nfPatch;
   int sy = sx * nxPatch;
   if (sy / sx != nxPatch) { overflow = true; }
   int sp = sy * nyPatch;
   if (sp / sy != nyPatch) { overflow = true; }

   size_t patchSize = sp * sizeof(pvwdata_t);
   if (patchSize / sp != sizeof(pvwdata_t) ) { overflow = true; }
   size_t dataSize = nPatches * patchSize;
   if (dataSize / nPatches != patchSize) { overflow = true; }
   size_t arborSize = dataSize * numberOfAxonalArborLists();
   if (arborSize / dataSize != numberOfAxonalArborLists()) { overflow = true; }

   if (overflow) {
      logError("Connection \"%s\" is too big (%d patches of size nxPatch=%d by nyPatch=%d by nfPatch=%d; %d arbors, weight size=%zu bytes).  Exiting.\n",
            getName(), nPatches, nxPatch, nyPatch, nfPatch, numberOfAxonalArborLists(), sizeof(pvwdata_t));
      exit(EXIT_FAILURE);
   }

   return (pvwdata_t *)pvCallocError(arborSize, sizeof(char), "Error allocating weights for connection \"%s\"", getName());
}

int HyPerConn::patchToDataLUT(int patchIndex) {
   return sharedWeights ? patch2datalookuptable[patchIndex] : patchIndex;
}

int HyPerConn::patchIndexToDataIndex(int patchIndex, int * kx/*default=NULL*/, int * ky/*default=NULL*/, int * kf/*default=NULL*/) {
   int dataIndex;
   if (sharedWeights) {
      dataIndex = calcUnitCellIndex(patchIndex, kx, ky, kf);
   }
   else {
      const PVLayerLoc * preLoc = pre->getLayerLoc();
      if(kx) *kx = kxPos(patchIndex, preLoc->nx + preLoc->halo.lt + preLoc->halo.rt, preLoc->ny + preLoc->halo.dn + preLoc->halo.up, preLoc->nf);
      if(ky) *ky = kyPos(patchIndex, preLoc->nx + preLoc->halo.lt + preLoc->halo.rt, preLoc->ny + preLoc->halo.dn + preLoc->halo.up, preLoc->nf);
      if(kf) *kf = featureIndex(patchIndex, preLoc->nx + preLoc->halo.lt + preLoc->halo.rt, preLoc->ny + preLoc->halo.dn + preLoc->halo.up, preLoc->nf);
      dataIndex = patchIndex;
   }
   return dataIndex;
}

int HyPerConn::dataIndexToUnitCellIndex(int dataIndex, int * kx/*default=NULL*/, int * ky/*default=NULL*/, int * kf/*default=NULL*/) {
   int unitCellIndex;
   if (sharedWeights) {
      int nfUnitCell = pre->getLayerLoc()->nf;
      int nxUnitCell = zUnitCellSize(pre->getXScale(), post->getXScale());
      int nyUnitCell = zUnitCellSize(pre->getYScale(), post->getYScale());
      pvAssert( dataIndex >= 0 && dataIndex < nxUnitCell*nyUnitCell*nfUnitCell );
      if(kx) *kx = kxPos(dataIndex, nxUnitCell, nyUnitCell, nfUnitCell);
      if(ky) *ky = kyPos(dataIndex, nxUnitCell, nyUnitCell, nfUnitCell);
      if(kf) *kf = featureIndex(dataIndex, nxUnitCell, nyUnitCell, nfUnitCell);
      unitCellIndex = dataIndex;
   }
   else {
      unitCellIndex = calcUnitCellIndex(dataIndex, kx, ky, kf);
   }
   return unitCellIndex;
}

int HyPerConn::calcUnitCellIndex(int patchIndex, int * kxUnitCellIndex/*default=NULL*/, int * kyUnitCellIndex/*default=NULL*/, int * kfUnitCellIndex/*default=NULL*/) {
   const PVLayerLoc * preLoc = pre->getLayerLoc();
   int nxUnitCell = zUnitCellSize(pre->getXScale(), post->getXScale());
   int nyUnitCell = zUnitCellSize(pre->getYScale(), post->getYScale());
   int unitCellIndex = layerIndexToUnitCellIndex(patchIndex, preLoc, nxUnitCell, nyUnitCell,
         kxUnitCellIndex, kyUnitCellIndex, kfUnitCellIndex);
   return unitCellIndex;
}

void HyPerConn::connOutOfMemory(const char * funcname) {
   exitFailure("Out of memory error in %s for connection \"%s\"\n", funcname, name);
}


/**
 * Find the weight value that that is in the nth percentile
 */
SparseWeightInfo HyPerConn::findPercentileThreshold(float percentile, pvwdata_t **wDataStart, size_t numAxonalArborLists, size_t numPatches, size_t patchSize) const {
   pvAssert(percentile >= 0.0f);
   pvAssert(percentile <= 1.0f);

   size_t fullWeightSize = numAxonalArborLists * numPatches * patchSize;
   SparseWeightInfo info;
   info.percentile = percentile;

   if (percentile >= 1.0) {
      info.size = fullWeightSize;
      info.thresholdWeight = 0.0;
      return info;
   }

   std::vector<pvwdata_t> weights;
   weights.reserve(fullWeightSize);

   for (int ar = 0; ar < numAxonalArborLists; ar++) {
      for (int pt = 0; pt < numPatches; pt++) {
         pvwdata_t *weight = &wDataStart[ar][pt * patchSize];
         for (int k = 0; k < patchSize; k++) {
            weights.push_back(fabs(weight[k]));
         }
      }
   }

   std::sort(weights.begin(), weights.end());
   int index = weights.size() * info.percentile;

   info.thresholdWeight = weights[index];
   info.size = weights.size() - index;
   return info;
}

SparseWeightInfo HyPerConn::calculateSparseWeightInfo() const {
   size_t patchSize = nfp * nxp * nyp;
   return findPercentileThreshold(_weightSparsity, wDataStart, numAxonalArborLists, numDataPatches, patchSize);
}

void HyPerConn::allocateSparseWeightsPre(PVLayerCube const *activity, int arbor) {
   _sparseWeightInfo = calculateSparseWeightInfo();

   std::map<const WeightType * const, int> sizes;

   for (int kPreExt = 0; kPreExt < activity->numItems; kPreExt++) {
      PVPatch *patch = getWeights(kPreExt, arbor);
      const int nk = patch->nx * fPatchSize();
      const int nyp = patch->ny;
      const WeightType * const weightDataStart = get_wData(arbor, kPreExt);

      for (int y = 0; y < nyp; y++) {
         const WeightType * const weightPtr = weightDataStart + y * yPatchStride();

         // Don't re-sparsify something that's already been put thru the sparsfication grinder
         bool shouldSparsify = false;

         // Find the weight pointers for this nk sized patch
         WeightMapType::iterator sparseWeightValuesNk = _sparseWeightValues.find(nk);
         IndexMapType::iterator sparseWeightIndexesNk = _sparseWeightIndexes.find(nk);

         if (sparseWeightValuesNk == _sparseWeightValues.end()) {
            // Weight pointers don't exist for this sized nk. Allocate a map for this nk
            _sparseWeightValues.insert(make_pair(nk, WeightPtrMapType()));
            _sparseWeightIndexes.insert(make_pair(nk, WeightIndexMapType()));
            // Get references
            sparseWeightValuesNk = _sparseWeightValues.find(nk);
            sparseWeightIndexesNk = _sparseWeightIndexes.find(nk);
            shouldSparsify = true;
         } else if (sparseWeightValuesNk->second.find(weightPtr) == sparseWeightValuesNk->second.end()) {
            // This nk group exists, but no weight pointer.
            shouldSparsify = true;
         }

         if (shouldSparsify) {
            WeightListType sparseWeight;
            IndexListType idx;

            // Equivalent to inner loop accumulate
            for (int k = 0; k < nk; k++) {
               WeightType weight = weightPtr[k];
               if (std::abs(weight) >= _sparseWeightInfo.thresholdWeight) {
                  sparseWeight.push_back(weight);
                  idx.push_back(k);
               }
            }

            sparseWeightValuesNk->second.insert(make_pair(weightPtr, sparseWeight));
            sparseWeightIndexesNk->second.insert(make_pair(weightPtr, idx));
         }
      }

      _kPreExtWeightSparsified.insert(kPreExt);
   }

   _sparseWeightsAllocated[arbor] = true;
}

void HyPerConn::allocateSparseWeightsPost(PVLayerCube const *activity, int arbor) {
   _sparseWeightInfo = calculateSparseWeightInfo();
   const PVLayerLoc *targetLoc = post->getLayerLoc();
   const PVHalo *targetHalo = &targetLoc->halo;
   const int targetNx = targetLoc->nx;
   const int targetNy = targetLoc->ny;
   const int targetNf = targetLoc->nf;

   for (int kTargetRes = 0; kTargetRes < post->getNumNeurons(); kTargetRes++) {
      // Change restricted to extended post neuron
      int kTargetExt = kIndexExtended(kTargetRes, targetNx, targetNy, targetNf, targetHalo->lt, targetHalo->rt, targetHalo->dn, targetHalo->up);
      // get source layer's patch y stride
      int syp = postConn->yPatchStride();
      int yPatchSize = postConn->yPatchSize();
      // Iterate through y patch
      int nk = postConn->xPatchSize() * postConn->fPatchSize();
      int kernelIndex = postConn->patchToDataLUT(kTargetExt);

      const WeightType * const weightDataStart = postConn->get_wDataHead(arbor, kernelIndex);

      for (int ky = 0; ky < yPatchSize; ky++) {
         const WeightType * const weightPtr = weightDataStart + ky * syp;

         // Don't re-sparsify something that's already been put thru the sparsfication grinder
         bool shouldSparsify = false;

         // Find the weight pointers for this nk sized patch
         // Find the weight pointers for this nk sized patch
         WeightMapType::iterator sparseWeightValuesNk = _sparseWeightValues.find(nk);
         IndexMapType::iterator sparseWeightIndexesNk = _sparseWeightIndexes.find(nk);

         if (_sparseWeightValues.find(nk) == _sparseWeightValues.end()) {
            // Weight pointers don't exist for this sized nk. Allocate a map for this nk
            _sparseWeightValues.insert(make_pair(nk, WeightPtrMapType()));
            _sparseWeightIndexes.insert(make_pair(nk, WeightIndexMapType()));
            // Get references
            sparseWeightValuesNk = _sparseWeightValues.find(nk);
            sparseWeightIndexesNk = _sparseWeightIndexes.find(nk);
            shouldSparsify = true;
         } else if (sparseWeightValuesNk->second.find(weightPtr) == sparseWeightValuesNk->second.end()) {
            // This nk group exists, but no weight pointer.
            shouldSparsify = true;
         }

         if (shouldSparsify) {
            WeightListType sparseWeight;
            IndexListType idx;

            for (int k = 0; k < nk; k++) {
               WeightType weight = weightPtr[k];
               if (std::abs(weight) >= _sparseWeightInfo.thresholdWeight) {
                  sparseWeight.push_back(weight);
                  idx.push_back(k);
               }
            }

            sparseWeightValuesNk->second.insert(make_pair(weightPtr, sparseWeight));
            sparseWeightIndexesNk->second.insert(make_pair(weightPtr, idx));
         }
      }

      _kPreExtWeightSparsified.insert(kTargetRes);
   }

   _sparseWeightsAllocated[arbor] = true;
}

void HyPerConn::deliverOnePreNeuronActivitySparseWeights(int kPreExt, int arbor, pvadata_t a, pvgsyndata_t * postBufferStart, void * auxPtr) {
   pvAssert(_sparseWeightsAllocated[arbor] == true);
   pvAssert(_kPreExtWeightSparsified.find(kPreExt) != _kPreExtWeightSparsified.end());

   PVPatch *patch = getWeights(kPreExt, arbor);
   const int nk = patch->nx * fPatchSize();
   const int nyp = patch->ny;
   const int sy  = getPostNonextStrides()->sy;       // stride in layer
   auto weightDataStart = get_wData(arbor, kPreExt);
   pvgsyndata_t *postPatchStart = postBufferStart + getGSynPatchStart(kPreExt, arbor);
   int offset = 0;

   for (int y = 0; y < nyp; y++) {
      WeightType *weightPtr = weightDataStart + y * yPatchStride();
      pvadata_t *post = postPatchStart + y * sy + offset;

      pvAssert(_sparseWeightValues.find(nk) != _sparseWeightValues.end());
      pvAssert(_sparseWeightValues.find(nk)->second.find(weightPtr) != _sparseWeightValues.find(nk)->second.end());

      const WeightListType& sparseWeights = _sparseWeightValues.find(nk)->second.find(weightPtr)->second;
      const IndexListType& idx = _sparseWeightIndexes.find(nk)->second.find(weightPtr)->second;

      for (int k = 0; k < sparseWeights.size(); k++) {
         int outIdx = idx[k];
         post[outIdx] += a * sparseWeights[k];
      }
   }
}

void HyPerConn::deliverOnePostNeuronActivitySparseWeights(int arborID, int kTargetExt, int inSy, float* activityStartBuf, pvdata_t* gSynPatchPos, float dt_factor, taus_uint4 * rngPtr) {
   // get source layer's patch y stride
   int syp = postConn->yPatchStride();
   int yPatchSize = postConn->yPatchSize();
   // Iterate through y patch
   int nk = postConn->xPatchSize() * postConn->fPatchSize();
   int kernelIndex = postConn->patchToDataLUT(kTargetExt);

   pvwdata_t* weightStartBuf = postConn->get_wDataHead(arborID, kernelIndex);
   int offset = 0;
   for (int ky = 0; ky < yPatchSize; ky++) {
      float * activityY = &(activityStartBuf[ky*inSy+offset]);
      
      pvwdata_t *weightPtr = weightStartBuf + ky * syp;

      pvAssert(_sparseWeightValues.find(nk) != _sparseWeightValues.end());
      pvAssert(_sparseWeightValues.find(nk)->second.find(weightPtr) != _sparseWeightValues.find(nk)->second.end());

      const WeightListType& sparseWeight = _sparseWeightValues.find(nk)->second.find(weightPtr)->second;
      const IndexListType& idx = _sparseWeightIndexes.find(nk)->second.find(weightPtr)->second;
      
      float dv = 0.0;
      for (int k = 0; k < sparseWeight.size(); k++) {
         dv += activityY[idx[k]] * sparseWeight[k];
      }
      *gSynPatchPos += dt_factor * dv;
   }
}

BaseObject * createHyPerConn(char const * name, HyPerCol * hc) {
   if (hc==NULL) { return NULL; }
   InitWeights * weightInitializer = getWeightInitializer(name, hc);
   NormalizeBase * weightNormalizer = getWeightNormalizer(name, hc);
   return new HyPerConn(name, hc, weightInitializer, weightNormalizer);
}

InitWeights * getWeightInitializer(char const * name, HyPerCol * hc) {
   if (hc==NULL) { return NULL; }
   InitWeights * weightInitializer = NULL;
   char const * weightInitStr = hc->parameters()->stringValue(name, "weightInitType", false/*warnIfAbsent*/);
   if (weightInitStr!=NULL) {
      BaseObject * baseObj = hc->getPV_InitObj()->create(weightInitStr, name, hc);
      weightInitializer = dynamic_cast<InitWeights*>(baseObj);
      if (weightInitializer==NULL) {
         if (hc->columnId()==0) {
            fprintf(stderr, "HyPerConn \"%s\" error: weightInitType \"%s\" is not recognized.\n", name, weightInitStr);
            exit(EXIT_FAILURE);
         }
      }
   }
   return weightInitializer;
}

NormalizeBase * getWeightNormalizer(char const * name, HyPerCol * hc) {
   if (hc==NULL) { return NULL; }
   NormalizeBase * weightNormalizer = NULL;
   char const * weightNormalizerStr = hc->parameters()->stringValue(name, "normalizeMethod", false/*warnIfAbsent*/);
   if (weightNormalizerStr!=NULL) {
      BaseObject * baseObj = hc->getPV_InitObj()->create(weightNormalizerStr, name, hc);
      weightNormalizer = dynamic_cast<NormalizeBase*>(baseObj);
      if (weightNormalizerStr==NULL) {
         if (hc->columnId()==0) {
            fprintf(stderr, "HyPerConn \"%s\" error: normalizeMethod \"%s\" is not recognized.\n", name, weightNormalizerStr);
            exit(EXIT_FAILURE);
         }
      }
   }
   return weightNormalizer;
}

} // namespace PV

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#ifndef PV_USE_OPENCL
#  include "../kernels/HyPerLayer_recv_synaptic_input.cl"
#else
#  undef PV_USE_OPENCL
#  include "../kernels/HyPerLayer_recv_synaptic_input.cl"
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif // __cplusplus
