/*
 * HyPerConn.cpp
 *
 *  Created on: Oct 21, 2008
 *      Author: Craig Rasmussen
 */

#include "HyPerConn.hpp"
#include "../include/default_params.h"
#include "../io/io.h"
#include "../io/fileio.hpp"
#include "../utils/conversions.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#include "../weightinit/InitWeights.hpp"
#include "../weightinit/InitCocircWeights.hpp"
#include "../weightinit/InitSmartWeights.hpp"
#include "../weightinit/InitUniformRandomWeights.hpp"
#include "../weightinit/InitGaussianRandomWeights.hpp"
#include "../normalizers/NormalizeBase.hpp"
#include "../normalizers/NormalizeSum.hpp"
#include "../normalizers/NormalizeL2.hpp"
#include "../normalizers/NormalizeMax.hpp"
#include "../normalizers/NormalizeContrastZeroMean.hpp"
#ifdef USE_SHMGET
   #include <sys/shm.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void HyPerLayer_recv_synaptic_input (
      int kx, int ky, int lidx, int lidy, int nxl, int nyl,
          int nxPre,
          int nyPre,
          int nfPre,
          int nbPre,
          int nxp,
          int nyp,
          int nfp,
          float fScale,
          float xScale,
          float yScale,
          size_t offsetA,
          int * p2dLUT,
           float * A,
           float * W,
           int Gstart,
           float   * G);


#ifdef __cplusplus
}
#endif // __cplusplus

namespace PV {

HyPerConn::HyPerConn()
{
   initialize_base();
}

HyPerConn::HyPerConn(const char * name, HyPerCol * hc, const char * pre_layer_name,
      const char * post_layer_name)
{
   initialize_base();
   initialize(name, hc, pre_layer_name, post_layer_name, NULL, NULL);
#ifdef PV_USE_OPENCL
   gpuAccelerateFlag=post->getUseGPUFlag();
   if(gpuAccelerateFlag)
      initializeGPU();
#endif
}

HyPerConn::HyPerConn(const char * name, HyPerCol * hc, const char * pre_layer_name,
      const char * post_layer_name, InitWeights *weightInit)
{
   initialize_base();
   initialize(name, hc, pre_layer_name, post_layer_name, NULL, weightInit);
#ifdef PV_USE_OPENCL
   gpuAccelerateFlag=post->getUseGPUFlag();
   if(gpuAccelerateFlag)
      initializeGPU();
#endif
}

// provide filename or set to NULL
HyPerConn::HyPerConn(const char * name, HyPerCol * hc, const char * pre_layer_name,
      const char * post_layer_name, const char * filename)
{
   initialize_base();
   initialize(name, hc, pre_layer_name, post_layer_name, filename, NULL);
#ifdef PV_USE_OPENCL
   gpuAccelerateFlag=post->getUseGPUFlag();
   if(gpuAccelerateFlag)
      initializeGPU();
#endif
}

HyPerConn::HyPerConn(const char * name, HyPerCol * hc, const char * pre_layer_name,
      const char * post_layer_name, const char * filename, InitWeights *weightInit)
{
   initialize_base();
   initialize(name, hc, pre_layer_name, post_layer_name, filename, weightInit);
#ifdef PV_USE_OPENCL
   gpuAccelerateFlag=post->getUseGPUFlag();
   if(gpuAccelerateFlag)
      initializeGPU();
#endif
}

HyPerConn::~HyPerConn()
{
   delete normalizer;
   if (parent->columnId() == 0) {
      printf("%32s: total time in %6s %10s: ", name, "conn", "io     ");
      io_timer->elapsed_time();
      printf("%32s: total time in %6s %10s: ", name, "conn", "update ");
      update_timer->elapsed_time();
      fflush(stdout);
   }
   delete io_timer;      io_timer     = NULL;
   delete update_timer;  update_timer = NULL;

   free(name);

#ifdef PV_USE_OPENCL
   if((gpuAccelerateFlag)&&(!ignoreGPUflag)) {
      delete krRecvSyn;

      if (clWeights != NULL) {
         for (int arbor = 0; arbor < numAxonalArborLists; arbor++) {
            delete clWeights[arbor];
         }
         free(clWeights);
         clWeights = NULL;
      }

      free(evRecvSynWaitList);
      evRecvSynWaitList=NULL;
      //delete gSynSemaphors;
      //gSynSemaphors=NULL;
   }
#endif // PV_USE_OPENCL

   deleteWeights();

   // free the task information

   // Moved to deleteWeights()
   // free(*gSynPatchStart); // All gSynPatchStart[k]'s were allocated together in a single malloc call.
   // free(gSynPatchStart);
   // free(*aPostOffset); // All aPostOffset[k]'s were allocated together in a single malloc call.
   // free(aPostOffset);

   free(delays);
   for (int i_probe = 0; i_probe < this->numProbes; i_probe++){
      free(probes[i_probe]);
   }
   free(this->probes);
   free(this->preLayerName);
   free(this->postLayerName);
   free(this->filename);

   delete weightInitializer;
   delete randState;

   if(triggerLayerName){
      //free(triggerLayerName);
      triggerLayerName = NULL;
   }
   //if(triggerLayer){
   //   free(triggerLayer);
   //}
}

//!
/*!
 *
 *
 *
 */
int HyPerConn::initialize_base()
{
   this->name = strdup("Unknown");
   this->nxp = 1;
   this->nyp = 1;
   this->nxpShrunken = nxp;
   this->nypShrunken = nyp;
   this->offsetShrunken = 0;
   this->nfp = -1; // A negative value for nfp will be converted to postsynaptic layer's nf.
   this->warnDefaultNfp = true;  // Issue a warning if default value of nfp (post's nf) is used.  Derived layers can set to false if only one nfp is allowed (e.g. IdentConn)
   this->sxp = 1;
   this->syp = 1;
   this->sfp = 1;
   this->parent = NULL;
   this->connId = 0;
   this->preLayerName = NULL;
   this->postLayerName = NULL;
   this->pre = NULL;
   this->post = NULL;
   this->filename = NULL;
   this->numAxonalArborLists = 1;
   this->channel = CHANNEL_EXC;
   this->ioAppend = false;

   this->weightInitializer = NULL;

   this->probes = NULL;
   this->numProbes = 0;

   this->io_timer     = new Timer();
   this->update_timer = new Timer();

   this->wMin = 0.0;
   this->wMax = 1.0;
   this->wPostTime = -1.0;
   this->wPostPatches = NULL;
   this->wPostDataStart = NULL;
   this->wPostPatchesp = NULL;
   this->wPostDataStartp = NULL;
   this->nxpPost = 0;
   this->nypPost = 0;
   this->nfpPost = 0;
   this->writeCompressedWeights = false;
   this->writeCompressedCheckpoints = false;
   this->fileType = PVP_WGT_FILE_TYPE; // Subclass's initialize_base() gets called after HyPerConn's initialize_base(), so this can be changed in subclasses.

   wPatches=NULL;
   // axonalArborList=NULL;
   // dwPatches = NULL;
   aPostOffset = NULL;

   this->selfFlag = false;  // specifies whether connection is from a layer to itself (i.e. a self-connection)
   this->combine_dW_with_W_flag = false;
   this->normalizer = NULL;
   // this->normalize_flag = true; // replaced by testing whether normalizer!=NULL
   this->plasticityFlag = false;
   this->shrinkPatches_flag = false; // default value, overridden by params file parameter "shrinkPatches" in readShrinkPatches()
   this->shrinkPatchesThresh = 0;
   this->normalizeArborsIndividually = true;
   this->normalize_max = false;
   this->normalize_zero_offset = false;
   this->normalize_cutoff = 0.0f;
   this->normalize_RMS_amp = false;
   this->dWMax            = 1;

   //This flag is only set otherwise in kernelconn
   this->useWindowPost = false;

   this->updateGSynFromPostPerspective = false;

   this->pvpatchAccumulateType = ACCUMULATE_CONVOLVE;

   this->initInfoCommunicatedFlag = false;
   this->dataStructuresAllocatedFlag = false;

   this->randState = NULL;

   this->triggerFlag = false; //Default to update every timestamp
   this->triggerLayer = NULL;
   this->triggerLayerName = NULL;
   this->triggerOffset = 0;

#ifdef USE_SHMGET
   shmget_flag = false;
   shmget_owner = NULL;
   shmget_id = NULL;
#endif
   return PV_SUCCESS;
}

int HyPerConn::createArbors() {
   wPatches = (PVPatch***) calloc(numAxonalArborLists, sizeof(PVPatch**));
   if( wPatches == NULL ) {
      createArborsOutOfMemory();
      assert(false);
   }
   // GTK:  gSynPatchStart redefined as offset form beginning of gSyn buffer for the corresponding channel
   //gSynPatchStart = (pvdata_t ***) calloc( numAxonalArborLists, sizeof(pvdata_t **) );
   gSynPatchStart = (size_t **) calloc( numAxonalArborLists, sizeof(size_t *) );
   if( gSynPatchStart == NULL ) {
      createArborsOutOfMemory();
      assert(false);
   }
   //   gSynPatchStartBuffer = (pvdata_t **) calloc(
   //         (this->shrinkPatches_flag ? numAxonalArborLists : 1)
   //               * preSynapticLayer()->getNumExtended(), sizeof(pvdata_t *));
   gSynPatchStartBuffer = (size_t *) calloc(
		   (this->shrinkPatches_flag ? numAxonalArborLists : 1)
		   * preSynapticLayer()->getNumExtended(), sizeof(size_t));
   if (gSynPatchStartBuffer == NULL) {
      createArborsOutOfMemory();
      assert(false);
   }
   for (int k = 0; k < numAxonalArborLists; k++) {
      gSynPatchStart[k] = gSynPatchStartBuffer
            + this->shrinkPatches_flag * k * preSynapticLayer()->getNumExtended();
   }

   aPostOffset = (size_t **) calloc(numAxonalArborLists, sizeof(size_t *));
   if( aPostOffset == NULL ) {
      createArborsOutOfMemory();
      assert(false);
   }
   aPostOffsetBuffer = (size_t *) calloc(
         (this->shrinkPatches_flag ? numAxonalArborLists : 1)
               * preSynapticLayer()->getNumExtended(), sizeof(size_t));
   if( aPostOffsetBuffer == NULL ) {
      createArborsOutOfMemory();
      assert(false);
   }
   for( int k=0; k<numAxonalArborLists; k++ ) {
      aPostOffset[k] = aPostOffsetBuffer
            + this->shrinkPatches_flag * k * preSynapticLayer()->getNumExtended();
   }
   wDataStart = (pvdata_t **) calloc(numAxonalArborLists, sizeof(pvdata_t *));
   if( wDataStart == NULL ) {
      createArborsOutOfMemory();
      assert(false);
   }
   dwDataStart = (pvdata_t **) calloc(numAxonalArborLists, sizeof(pvdata_t *));
   if( dwDataStart == NULL ) {
      createArborsOutOfMemory();
      assert(false);
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
int HyPerConn::constructWeights(const char * filename)
{
   int sx = nfp;
   int sy = sx * nxp;
   int sp = sy * nyp;
   int nPatches = getNumDataPatches();
   int status = PV_SUCCESS;

   assert(!parent->parameters()->presentAndNotBeenRead(name, "shrinkPatches"));
   // createArbors() uses the value of shrinkPatches.  It should have already been read in setParams.
   //allocate the arbor arrays:
   createArbors();

   // setPatchSize(filename); // moved to readPatchSize() so that nxp, nyp are set as early as possible
   setPatchStrides();

   //allocate weight patches and axonal arbors for each arbor
   //Allocate all the weights
   wDataStart[0] = allocWeights(nPatches, nxp, nyp, nfp);
   assert(this->get_wDataStart(0) != NULL);
   for (int arborId=0;arborId<numAxonalArborLists;arborId++) {
      status = createWeights(wPatches, arborId);
      assert(wPatches[arborId] != NULL);

      //wDataStart[arborId] = createWeights(wPatches, arborId);
      if (arborId > 0){  // wDataStart already allocated
         wDataStart[arborId] = (this->get_wDataStart(0) + sp * nPatches * arborId);
         assert(this->wDataStart[arborId] != NULL);
      }
      if (shrinkPatches_flag || arborId == 0){
         status |= adjustAxonalArbors(arborId);
      }
   }  // arborId

   //initialize weights for patches:
   status |= initializeWeights(wPatches, wDataStart, getNumDataPatches(), filename) != NULL ? PV_SUCCESS : PV_FAILURE;
   assert(status == 0);
   status |= initPlasticityPatches();
   assert(status == 0);
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

int HyPerConn::shrinkPatch(int kExt, int arborId /* PVAxonalArbor * arbor */) {

   int kIndex = patchToDataLUT(kExt);

   PVPatch *weights = getWeights(kExt,arborId);

   pvdata_t * w = &get_wDataStart(arborId)[kIndex*nxp*nyp*nfp+weights->offset];
   // pvdata_t * w = weights->data;

   int nx = weights->nx;
   int ny = weights->ny;
   //int nfp = weights->nf;

   //int sxp = weights->sx;
   //int syp = weights->sy;
   //int sfp = weights->sf;

   int maxnx = INT_MIN;
   int minnx = INT_MAX;
   int maxny = INT_MIN;
   int minny = INT_MAX;

   bool nonZeroWeightFound = false;
   // loop over all post-synaptic cells in patch
   for (int y = 0; y < ny; y++) {
      for (int x = 0; x < nx; x++) {
         for (int f = 0; f < nfp; f++) {
            if(abs(w[x * sxp + y * syp + f * sfp]) <= shrinkPatchesThresh) {
               //w[x*sxp + y*syp + f*sfp] = 0;
               nonZeroWeightFound=true;
               //pvdata_t weight = w[x * sxp + y * syp + f * sfp];
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

      // adjust patch size (shrink) for the data to fit within interior of post-synaptic layer
      //
      // pvpatch_adjust(arbor->data, nxNew, nyNew, dxNew, dyNew);
      gSynPatchStart[arborId][kExt] += dxNew*getPostNonextStrides()->sx + dyNew*getPostNonextStrides()->sy;
      aPostOffset[arborId][kExt] += dxNew*getPostExtStrides()->sx + dyNew*getPostExtStrides()->sy; // Someone who uses these routines, please check that this is correct.
   }
   return 0;
}


void HyPerConn::readShrinkPatches(PVParams * params) {
   shrinkPatches_flag = params->value(name, "shrinkPatches", shrinkPatches_flag);
   if(shrinkPatches_flag){
      shrinkPatchesThresh = params->value(name, "shrinkPatchesThresh", shrinkPatchesThresh);
   }
}

void HyPerConn::readUpdateGSynFromPostPerspective(PVParams * params){
   updateGSynFromPostPerspective = (bool) params->value(name, "updateGSynFromPostPerspective", updateGSynFromPostPerspective);
}

int HyPerConn::initialize(const char * name, HyPerCol * hc, const char * pre_layer_name,
      const char * post_layer_name, const char * filename, InitWeights *weightInit) {
   int status = PV_SUCCESS;

   status = setParent(hc);
   status = setName(name);
   status = setPreLayerName(pre_layer_name);
   status = setPostLayerName(post_layer_name);
   status = setFilename(filename);
   status = setWeightInitializer(weightInit);

   assert(parent);
   PVParams * inputParams = parent->parameters();
   status = setParams(inputParams);

   //set accumulateFunctionPointer
   assert(!inputParams->presentAndNotBeenRead(name, "pvpatchAccumulateType"));
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
      accumulateFunctionPointer = &pvpatch_max_pooling;
      accumulateFunctionFromPostPointer = &pvpatch_max_pooling_from_post;
      break;
   default:
      assert(0);
      break;
   }

   ioAppend = parent->getCheckpointReadFlag();

//This has been commented out because layers will decide if GPU acceleration
//will happen and they will call the init methods as necessary
//#ifdef PV_USE_OPENCL
//   initializeThreadBuffers("HyPerLayer_recv_synaptic_input");
//   initializeThreadKernels("HyPerLayer_recv_synaptic_input");
//#endif // PV_USE_OPENCL
#ifdef PV_USE_OPENCL
   gpuAccelerateFlag=post->getUseGPUFlag();
#endif

   this->connId = parent->addConnection(this);

   return status;
}

int HyPerConn::setParent(HyPerCol * hc) {
   assert(parent==NULL);
   if(hc==NULL) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      fprintf(stderr, "HyPerConn error in rank %d process: constructor called with HyPerCol set to the null pointer.\n", rank);
      exit(EXIT_FAILURE);
   }
   parent = hc;
   return PV_SUCCESS;
}

int HyPerConn::setName(const char * name) {
   assert(parent!=NULL);
   if(name==NULL) {
      fprintf(stderr, "HyPerConn error in rank %d process: constructor called with name set to the null pointer.\n", parent->columnId());
      exit(EXIT_FAILURE);
   }
   free(this->name);  // name will already have been set in initialize_base()
   this->name = strdup(name);
   if (this->name==NULL) {
      fprintf(stderr, "Connection \"%s\" error in rank %d process: unable to allocate memory for name of connection: %s\n",
            name, parent->columnId(), strerror(errno));
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}

int HyPerConn::setPreLayerName(const char * pre_name) {
   assert(parent!=NULL);
   assert(this->preLayerName==NULL);
   if (pre_name != NULL) {
      this->preLayerName = strdup(pre_name);
      if (this->preLayerName==NULL) {
         fprintf(stderr, "Connection \"%s\" error in rank %d process: unable to allocate memory for name of presynaptic layer \"%s\": %s\n",
               name, parent->columnId(), pre_name, strerror(errno));
         exit(EXIT_FAILURE);
      }
   }
   return PV_SUCCESS;
}

int HyPerConn::setPostLayerName(const char * post_name) {
   assert(this->postLayerName==NULL);
   if (post_name != NULL) {
      this->postLayerName = strdup(post_name);
      if (this->postLayerName==NULL) {
         fprintf(stderr, "Connection \"%s\" error in rank %d process: unable to allocate memory for name of postsynaptic layer \"%s\": %s\n",
               name, parent->columnId(), post_name, strerror(errno));
         exit(EXIT_FAILURE);
      }
   }
   return PV_SUCCESS;
}

int HyPerConn::setFilename(const char * filename) {
   // filename can be (and often is) NULL, but for FileWeight initialization method, it is the name of the file used to load the weights.
   assert(parent!=NULL);
   assert(this->filename==NULL); // Should have been nulled in initialize_base, and nothing else should have changed it.
   if (filename!=NULL) {
      this->filename = strdup(filename);
      if (this->filename==NULL) {
         fprintf(stderr, "Connection \"%s\" error in rank %d process: unable to allocate memory for InitWeights filename \"%s\": %s\n",
               name, parent->columnId(), filename, strerror(errno));
         exit(EXIT_FAILURE);
      }
   }
   return PV_SUCCESS;
}

int HyPerConn::setWeightInitializer(InitWeights * weightInit) {
   //if a weight initializer hasn't been created already, use the default--> either 2D Gauss or read from file
   if(weightInit==NULL) {
      this->weightInitializer = handleMissingInitWeights(parent->parameters());
   }
   else {
      this->weightInitializer = weightInit;
   }
   return PV_SUCCESS;
}


int HyPerConn::getPreAndPostLayerNames(const char * name, PVParams * params, char ** preLayerNamePtr, char ** postLayerNamePtr) {
   // Retrieves preLayerName and postLayerName from parameter group whose name is given in the functions first argument.
   // This routine uses strdup to fill *{pre,post}LayerNamePtr, so the routine calling this one is responsible for freeing them.
   int status = PV_SUCCESS;
   *preLayerNamePtr = NULL;
   *postLayerNamePtr = NULL;
   const char * preLayerNameParam = params->stringValue(name, "preLayerName", false);
   const char * postLayerNameParam = params->stringValue(name, "postLayerName", false);
   if (preLayerNameParam != NULL && postLayerNameParam != NULL) {
      *preLayerNamePtr = strdup(preLayerNameParam);
      *postLayerNamePtr = strdup(postLayerNameParam);
   }
   else if (preLayerNameParam==NULL && postLayerNameParam!=NULL) {
      status = PV_FAILURE;
      if (params->getInterColComm()->commRank()==0) {
         fprintf(stderr, "Connection \"%s\" error: if postLayerName is specified, preLayerName must be specified as well.\n", name);
      }
   }
   else if (preLayerNameParam!=NULL && postLayerNameParam==NULL) {
      status = PV_FAILURE;
      if (params->getInterColComm()->commRank()==0) {
         fprintf(stderr, "Connection \"%s\" error: if preLayerName is specified, postLayerName must be specified as well.\n", name);
      }
   }
   else {
      assert(preLayerNameParam==NULL && postLayerNameParam==NULL);
      if (params->getInterColComm()->commRank()==0) {
         printf("Connection \"%s\": preLayerName and postLayerName will be inferred in the communicateInitInfo stage.\n", name);
      }
   }
   MPI_Barrier(params->getInterColComm()->communicator());
   if (status != PV_SUCCESS) {
      exit(EXIT_FAILURE);
   }
   return status;
}

int HyPerConn::handleMissingPreAndPostLayerNames() {
   return inferPreAndPostFromConnName(name, parent->parameters(), &preLayerName, &postLayerName);
}

int HyPerConn::inferPreAndPostFromConnName(const char * name, PVParams * params, char ** preLayerNamePtr, char ** postLayerNamePtr) {
   // If the connection name has the form "ABC to XYZ", then pre will be ABC and post will be XYZ.
   // If either of the intended pre- or post-layer names contains the string " to ", this method cannot be used to infer them.
   // This routine uses malloc to fill *{pre,post}LayerNamePtr, so the routine calling this one is responsible for freeing them.

   int status = PV_SUCCESS;
   // Check to see if the string " to " appears exactly once in name
   // If so, use part preceding " to " as pre-layer, and part after " to " as post.
   const char * separator = " to ";
   const char * locto = strstr(name, separator);
   if( locto != NULL ) {
      const char * nextto = strstr(locto+1, separator); // Make sure " to " doesn't appear again.
      if( nextto == NULL ) {
         int seplen = strlen(separator);

         int pre_len = locto - name;
         *preLayerNamePtr = (char *) malloc((size_t) (pre_len + 1));
         if( *preLayerNamePtr==NULL) {
            fprintf(stderr, "Error: unable to allocate memory for preLayerName in connection \"%s\": %s\n", name, strerror(errno));
            exit(EXIT_FAILURE);
         }
         const char * preInConnName = name;
         memcpy(*preLayerNamePtr, preInConnName, pre_len);
         (*preLayerNamePtr)[pre_len] = 0;

         int post_len = strlen(name)-pre_len-seplen;
         *postLayerNamePtr = (char *) malloc((size_t) (post_len + 1));
         if( *postLayerNamePtr==NULL) {
            fprintf(stderr, "Error: unable to allocate memory for postLayerName in connection \"%s\": %s\n", name, strerror(errno));
            exit(EXIT_FAILURE);
         }
         const char * postInConnName = &name[pre_len+seplen];
         memcpy(*postLayerNamePtr, postInConnName, post_len);
         (*postLayerNamePtr)[post_len] = 0;
      }
      else {
         status = PV_FAILURE;
         if (params->getInterColComm()->commRank()==0) {
            fprintf(stderr, "Unable to infer pre and post from connection name \"%s\":\n", name);
            fprintf(stderr, "The string \" to \" cannot appear in the name more than once.\n");
         }
      }
   }
   else {
      status = PV_FAILURE;
      if (params->getInterColComm()->commRank()==0) {
         fprintf(stderr, "Unable to infer pre and post from connection name \"%s\".\n", name);
         fprintf(stderr, "The connection name must have the form \"ABC to XYZ\", to infer the names,\n");
         fprintf(stderr, "but the string \" to \" does not appear.\n");
      }
   }
   return status;
}

int HyPerConn::initNumWeightPatches() {
   numWeightPatches = pre->getNumExtended();
   return PV_SUCCESS;
}

int HyPerConn::initNumDataPatches() {
   numDataPatches = getNumWeightPatches();
   return PV_SUCCESS;
}

int HyPerConn::initPlasticityPatches()
{
   int sx = nfp;
   int sy = sx * nxp;
   int sp = sy * nyp;
   int nPatches = getNumDataPatches();
   if (!plasticityFlag) return PV_SUCCESS;

   const int numAxons = numberOfAxonalArborLists();

   if (this->combine_dW_with_W_flag){
      dwDataStart = wDataStart;
      return PV_SUCCESS;
   }
   dwDataStart[0] = allocWeights(nPatches, nxp, nyp, nfp);
   assert(this->get_dwDataStart(0) != NULL);
   for (int arborId = 0; arborId < numAxons; arborId++) {
      dwDataStart[arborId] = (dwDataStart[0] + sp * nPatches * arborId);
      //set_dwDataStart(arborId, allocWeights(getNumDataPatches(), nxp, nyp, nfp, arborId));
      assert(get_dwDataStart(arborId) != NULL);
   } // loop over arbors

   return PV_SUCCESS;
}

// set member variables specified by user
int HyPerConn::setParams(PVParams * inputParams)
{
   readChannelCode(inputParams);
   readNumAxonalArbors(inputParams);
   readPlasticityFlag(inputParams);
   readWeightUpdatePeriod(inputParams);
   readInitialWeightUpdateTime(inputParams);
   readPvpatchAccumulateType(inputParams);
   readPreActivityIsNotRate(inputParams);
   readWriteStep(inputParams);
   readInitialWriteTime(inputParams);
   readWriteCompressedWeights(inputParams);
   readWriteCompressedCheckpoints(inputParams);
   readSelfFlag(inputParams);
   readCombine_dW_with_W_flag(inputParams);
   readDelay(inputParams);
   readPatchSize(inputParams);
   readNfp(inputParams);
   readShrinkPatches(inputParams); // Sets shrinkPatches_flag; derived-class methods that override readShrinkPatches must also set shrinkPatches_flag
   readUpdateGSynFromPostPerspective(inputParams);
   readTriggerFlag(inputParams);
   return PV_SUCCESS;

   return 0;
}

void HyPerConn::readTriggerFlag(PVParams * params) {
   triggerFlag = (bool) params->value(name, "triggerFlag", triggerFlag);
   if(triggerFlag){
      const char * trigger_layer_name = params->stringValue(name, "triggerLayerName");
      if(!trigger_layer_name){
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: triggerLayerName must be defined if triggerFlag is set\n", parent->parameters()->groupKeywordFromName(name), name);;
            exit(EXIT_FAILURE);
         }
      }
      triggerLayerName = strdup(trigger_layer_name);
      if (triggerLayerName ==NULL) {
         fprintf(stderr, "%s \"%s\" error: rank %d process unable to copy triggerLayerName \"%s\": %s\n",
                 parent->parameters()->groupKeywordFromName(name), name, parent->columnId(), trigger_layer_name, strerror(errno));
         exit(EXIT_FAILURE);
      }
      triggerOffset = params->value(name, "triggerOffset", triggerOffset);
   }
}

void HyPerConn::readChannelCode(PVParams * params) {
   int is_present = params->present(name, "channelCode");
   if (!is_present) {
      fprintf(stderr, "Group \"%s\" must set parameter channelCode.\n", name);
      abort();
   }
   int ch = (int)params->value(name, "channelCode");
   int status = decodeChannel(ch, &channel);
   if (status != PV_SUCCESS) {
      fprintf(stderr, "HyPerConn::readChannelCode: channelCode %d for connection \"%s\" is not a valid channel.\n",  ch, name);
      abort();
   }
}

int HyPerConn::decodeChannel(int channel_code, ChannelType * channel_type) {
   int status = PV_SUCCESS;
   switch( channel_code ) {
   case CHANNEL_EXC:
      *channel_type = CHANNEL_EXC;
      break;
   case CHANNEL_INH:
      *channel_type = CHANNEL_INH;
      break;
   case CHANNEL_INHB:
      *channel_type = CHANNEL_INHB;
      break;
   case CHANNEL_GAP:
      *channel_type = CHANNEL_GAP;
      break;
   case CHANNEL_NORM:
      *channel_type = CHANNEL_NORM;
      break;
   default:
      *channel_type = CHANNEL_INVALID;
      status = PV_FAILURE;
      break;
   }
   return status;
}

void HyPerConn::readNumAxonalArbors(PVParams * params) {
   numAxonalArborLists=(int) params->value(name, "numAxonalArbors", 1, true);
   if (numAxonalArborLists==0 && parent->columnId()==0) {
         fprintf(stdout, "HyPerConn:: Warning: Connection %s: Variable numAxonalArbors is set to 0. No connections will be made.\n",name);
   }
}

void HyPerConn::readPlasticityFlag(PVParams * params) {
   plasticityFlag = params->value(name, "plasticityFlag", plasticityFlag, true) != 0;
}

void HyPerConn::readWeightUpdatePeriod(PVParams * params) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   weightUpdatePeriod = 1.0f;
   if (plasticityFlag) {
      weightUpdatePeriod = params->value(name, "weightUpdatePeriod", weightUpdatePeriod);
   }
}

void HyPerConn::readInitialWeightUpdateTime(PVParams * params) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   weightUpdateTime = 0.0f;
   if (plasticityFlag) {
      weightUpdateTime = params->value(name, "initialWeightUpdateTime", weightUpdateTime);
   }
}

void HyPerConn::readPvpatchAccumulateType(PVParams * params) {
   // stochasticReleaseFlag deprecated on Aug 22, 2013.
   if (params->present(name, "stochasticReleaseFlag")) {
      bool stochasticReleaseFlag = params->value(name, "stochasticReleaseFlag");
      pvpatchAccumulateType = stochasticReleaseFlag ? ACCUMULATE_STOCHASTIC : ACCUMULATE_CONVOLVE;
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" warning: parameter stochasticReleaseFlag is deprecated.  Instead, set pvpatchAccumulateType to one of \"convolve\" (the default), \"stochastic\", or \"maxpooling\".\n", parent->parameters()->groupKeywordFromName(name), name);
         fprintf(stderr, "    pvpatcchAccumulateType set to \"%s\" \n", stochasticReleaseFlag ? "stochastic" : "convolve");
      }
      return;
   }

   const char * pvpatch_accumulate_type = params->stringValue(name, "pvpatchAccumulateType", true);
   if (pvpatch_accumulate_type==NULL) {
      if (parent->columnId()==0) {
         printf("%s \"%s\": pvpatchAccumulateType set to \"convolve\"\n", parent->parameters()->groupKeywordFromName(name), name);
      }
      pvpatchAccumulateType = ACCUMULATE_CONVOLVE;
   }
   else {
      // Convert string to lowercase so that capitalization doesn't matter.
      char * pvpatch_accumulate_type_i = strdup(pvpatch_accumulate_type);
      if (pvpatch_accumulate_type_i == NULL) {
         fprintf(stderr, "%s \"%s\" error: Rank %d process unable to copy pvpatchAccumulateType string.\n", parent->parameters()->groupKeywordFromName(name), name, parent->columnId());
         exit(EXIT_FAILURE);
      }
      for (char * c = pvpatch_accumulate_type_i; *c!='\0'; c++) {
         *c = (char) tolower((int) *c);
      }

      if (strcmp(pvpatch_accumulate_type_i,"convolve")==0) {
         pvpatchAccumulateType = ACCUMULATE_CONVOLVE;
      }
      else if (strcmp(pvpatch_accumulate_type_i,"stochastic")==0) {
         pvpatchAccumulateType = ACCUMULATE_STOCHASTIC;
      }
      else if (strcmp(pvpatch_accumulate_type_i,"maxpooling")==0 ||
               strcmp(pvpatch_accumulate_type_i,"max pooling")==0) {
         pvpatchAccumulateType = ACCUMULATE_MAXPOOLING;
      }
      else {
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: pvpatchAccumulateType \"%s\" unrecognized.  Allowed values are \"convolve\", \"stochastic\", or \"maxpooling\"\n",
                  parent->parameters()->groupKeywordFromName(name), name, pvpatch_accumulate_type);
         }
         MPI_Barrier(parent->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      free(pvpatch_accumulate_type_i);
   }
}

void HyPerConn::readPreActivityIsNotRate(PVParams * params) {
   preActivityIsNotRate = params->value(name, "preActivityIsNotRate", false, true) != 0;
}

void HyPerConn::readWriteStep(PVParams * params) {
   writeStep = params->value(name, "writeStep", parent->getDeltaTime());
}


void HyPerConn::readInitialWriteTime(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "writeStep"));
   double sim_time = parent->simulationTime();
   writeTime = sim_time;
   if (writeStep>=0) {
      if (!params->present(name, "writeStep")) {
         if (parent->columnId()==0) {
            fprintf(stderr, "HyPerConn::readInitialWriteTime warning for connection \"%s\": reading initialWriteTime using default for writeStep.\n", name);
         }
      }
      writeTime = params->value(name, "initialWriteTime", sim_time);
      while (writeTime < sim_time) {
         writeTime += writeStep;
      }
   }
}

void HyPerConn::readWriteCompressedWeights(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "writeStep"));
   if (writeStep>=0) {
      writeCompressedWeights = params->value(name, "writeCompressedWeights", writeCompressedWeights, /*warnifabsent*/true) != 0;
   }
}

void HyPerConn::readWriteCompressedCheckpoints(PVParams * params) {
   if (parent->getCheckpointWriteFlag() || !parent->getSuppresLastOutputFlag()) {
      writeCompressedCheckpoints = params->value(name, "writeCompressedCheckpoints", writeCompressedCheckpoints, /*warnifabsent*/true) != 0;
   }
}

void HyPerConn::readSelfFlag(PVParams * params) {
   // The default for selfFlag should be whether pre and post are the same HyPerLayer object,
   // but pre and post are not set until communicateInitInfo is called.
   // So we check for a selfFlag parameter but do not warn if it is absent, and use false as the
   // default.
   // In communicateInitInfo(), we check if the selfFlag is present in the params, and if not,
   // selfFlag becomes (pre == post)
   selfFlag = params->value(name, "selfFlag", selfFlag, false) != 0;
}

void HyPerConn::readCombine_dW_with_W_flag(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag){
      combine_dW_with_W_flag = params->value(name, "combine_dW_with_W_flag", combine_dW_with_W_flag, true) != 0;
   }
}

void HyPerConn::read_dWMax(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag){
      dWMax = params->value(getName(), "dWMax", dWMax, true);
   }
}

void HyPerConn::readDelay(PVParams * params) {
   //Grab delays in ms and change into timesteps
   //fDelayArray is the float representation from the array values
   fDelayArray = params->arrayValues(name, "delay", &delayArraySize);
   //Being called now in allocate
   //initializeDelays(fDelayArray, delayArraySize);
}

int HyPerConn::initializeDelays(const float * fDelayArray, int size){

   int status = PV_SUCCESS;
   assert(!parent->parameters()->presentAndNotBeenRead(name, "numAxonalArbors"));
   //Allocate delay data structure
   delays = (int *) calloc(numAxonalArborLists, sizeof(int));
   if( delays == NULL ) {
      createArborsOutOfMemory();
      assert(false);
   }

   //Initialize delays for each arbor
   //Using setDelay to convert ms to timesteps
   for (int arborId=0;arborId<numAxonalArborLists;arborId++) {
      if (size == 0){
         //No delay
         setDelay(arborId, 0);
      }
      else if (size == 1){
         setDelay(arborId, fDelayArray[0]);
      }
      else if (size == numAxonalArborLists){
         setDelay(arborId, fDelayArray[arborId]);
      }
      else{
         fprintf(stderr, "Delay must be either a single value or the same length as the number of arbors\n");
         abort();
      }
   }
   return status;
}

int HyPerConn::readPatchSize(PVParams * params) {
   int status = PV_SUCCESS;
   nxp = parent->parameters()->value(name, "nxp", 1); // These two lines could be moved to readPatchSizeFromParams,
   nyp = parent->parameters()->value(name, "nyp", 1); // but checkWeightHeader() expects that nxp,nyp will have been read.
   nxpShrunken = parent->parameters()->value(name,"nxpShrunken",nxp,false);
   nypShrunken = parent->parameters()->value(name,"nypShrunken",nyp,false);
   if (filename!=NULL) {
      status = readPatchSizeFromFile(filename);
   // }
   // else {
   //    status = readPatchSizeFromParams(params);
   }
   return status;
}

int HyPerConn::checkShrunkenSize() {
   int status = PV_SUCCESS;
   // nxpShrunken = parent->parameters()->value(name,"nxpShrunken",nxp,false);
   // nypShrunken = parent->parameters()->value(name,"nypShrunken",nyp,false);
   if(nxpShrunken>nxp) {
      if (parent->columnId()==0) {
         fprintf(stderr, "readPatchSize error in connection \"%s\": nxpShrunken cannot be greater than nxp.\n", name);
      }
      status = PV_FAILURE;
   }
   if(nypShrunken>nyp) {
      if (parent->columnId()==0) {
         fprintf(stderr, "readPatchSize error in connection \"%s\": nypShrunken cannot be greater than nyp.\n", name);
      }
      status = PV_FAILURE;
   }
   return status;
}

int HyPerConn::readPatchSizeFromFile(const char * filename) {
   assert(filename != NULL);
   int status = PV_SUCCESS;
   readUseListOfArborFiles(parent->parameters());
   readCombineWeightFiles(parent->parameters());
   if( !useListOfArborFiles && !combineWeightFiles) { // Should still get patch size from file if either of these flags is true
      status = patchSizeFromFile(filename);
   }
   // else {
   //    status = readPatchSizeFromParams(parent->parameters());
   // }
   return status;
}

int HyPerConn::readNfp(PVParams * params) {
   if (filename==NULL) { // If filename is set, nfp already will have been read by patchSizeFromFile.
      nfp = params->value(name, "nfp", -1);
   }
   return PV_SUCCESS;
}

void HyPerConn::readUseListOfArborFiles(PVParams * params) {
   assert(filename!=NULL);
   useListOfArborFiles = params->value(name, "useListOfArborFiles", false)!=0;
}

void HyPerConn::readCombineWeightFiles(PVParams * params) {
   assert(filename!=NULL);
   combineWeightFiles = params->value(name, "combineWeightFiles", false)!=0;
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

   int status = PV_SUCCESS;

   if (preLayerName==NULL) {
      assert(postLayerName==NULL);
      status = handleMissingPreAndPostLayerNames();
   }
   MPI_Barrier(parent->icCommunicator()->communicator());
   if (status != PV_SUCCESS) {
      assert(preLayerName==NULL && postLayerName==NULL);
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: Unable to determine pre- and post-layer names.  Exiting.\n", parent->parameters()->groupKeywordFromName(name), name);
      }
      exit(EXIT_FAILURE);
   }
   this->pre = parent->getLayerFromName(preLayerName);
   this->post = parent->getLayerFromName(postLayerName);
   if (this->pre==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "Connection \"%s\": preLayerName \"%s\" does not correspond to a layer in the column.\n", name, preLayerName);
      }
      status = PV_FAILURE;
   }

   if (this->post==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "Connection \"%s\": postLayerName \"%s\" does not correspond to a layer in the column.\n", name, postLayerName);
      }
      status = PV_FAILURE;
   }
   MPI_Barrier(parent->icCommunicator()->communicator());
   if (status != PV_SUCCESS) {
      exit(EXIT_FAILURE);
   }

   handleDefaultSelfFlag(); // If selfFlag was not specified in params, it should be set to pre==post
                            // but pre and post were not determined until now.

   // Find maximum delay over all the arbors and send it to the presynaptic layer
   int maxdelay = 0;
   for (int delayi = 0; delayi < delayArraySize; delayi++){
      if (fDelayArray[delayi] > maxdelay){
         maxdelay = fDelayArray[delayi];
      }
   }
   //for( int arborId=0; arborId<numberOfAxonalArborLists(); arborId++ ) {
   //   int curdelay = this->getDelay(arborId);
   //   if( maxdelay < curdelay ) maxdelay = curdelay;
   //}
   int allowedDelay = pre->increaseDelayLevels(maxdelay);
   if( allowedDelay < maxdelay ) {
      if( parent->icCommunicator()->commRank() == 0 ) {
         fflush(stdout);
         fprintf(stderr, "Connection \"%s\": attempt to set delay to %d, but the maximum allowed delay is %d.  Exiting\n", name, maxdelay, allowedDelay);
      }
      exit(EXIT_FAILURE);
   }

   // Make sure post-synaptic layer has enough channels.
   int num_channels_check;
   status = post->requireChannel((int) channel, &num_channels_check);
   if (status != PV_SUCCESS) { return status; }

   assert(num_channels_check > (int) channel);

   status = setPatchSize();
   status = checkPatchDimensions();

   if (nfp == -1) {
      nfp = post->getCLayer()->loc.nf;
      if (warnDefaultNfp && parent->columnId()==0) {
         printf("Connection \"%s\" setting nfp to number of postsynaptic features = %d.\n", name, nfp);
      }
   }
   if (nfp != post->getCLayer()->loc.nf) {
      if (parent->columnId()==0) {
         fprintf( stderr, "Params file specifies %d features for connection \"%s\",\n", nfp, name );
         fprintf( stderr, "but %d features for post-synaptic layer %s\n",
               post->getCLayer()->loc.nf, post->getName() );
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
   int margin = xmargin>=ymargin ? xmargin : ymargin;
   int receivedmargin = 0;
   status = pre->requireMarginWidth(margin, &receivedmargin);
   if (status != PV_SUCCESS) {
      status = PV_MARGINWIDTH_FAILURE;
      fprintf(stderr,"Margin Failure for layer %s.  Received margin is %d, but required margin is %d",name,receivedmargin,margin);
   }

   //Trigger stuff
   if(triggerFlag){
      triggerLayer = parent->getLayerFromName(triggerLayerName);
      if (triggerLayer==NULL) {
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: triggerLayer \"%s\" is not a layer in the HyPerCol.\n",
                    parent->parameters()->groupKeywordFromName(name), name, triggerLayerName);
         }
         MPI_Barrier(parent->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      //Use either triggering or weightUpdatePeriod
      //If not default values, print warning
      if(weightUpdatePeriod != 1 || weightUpdateTime != 0){
         std::cout << "Warning: Connection " << name << " trigger flag is set, ignoring weightUpdatePeriod and initialWeightUpdateTime\n";
      }
      //Although weightUpdatePeriod and weightUpdateTime is being set here, if trigger flag is set, they are not being used
      //Only updating for backwards compatibility
      //getDeltaUpdateTime can return -1 (if it never updates), so set placisity flag off if so
      weightUpdatePeriod = triggerLayer->getDeltaUpdateTime();
      if(weightUpdatePeriod <= 0){
         if(plasticityFlag == true){
            std::cout << "Warning: Connection " << name << "triggered layer " << triggerLayerName << " never updates, turning placisity flag off\n";
            plasticityFlag = false;
         }
      }
      weightUpdateTime = 1;
   }


   return status;
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
   if (filename) {
      const PVLayerLoc * loc = pre->getLayerLoc();
      status = checkPVPFileHeader(parent->icCommunicator(), loc, fileparams, NUM_WGT_PARAMS);
      if (status < 0) status = PV_FAILURE;
   }
   return status;
}

// returns handle to initialized weight patches
PVPatch *** HyPerConn::initializeWeights(PVPatch *** patches, pvdata_t ** dataStart, int numPatches, const char * filename)
{
   weightInitializer->initializeWeights(patches, dataStart, filename, this);
   // insert synchronization barrier to ensure that all processes have finished loading portions of shared memory for which they
   // might be responsible
#ifdef USE_SHMGET
#ifdef PV_USE_MPI
   //std::cout << "starting MPI_Barrier in HyPerConn::initializeWeights: " << this->name << ", rank = " << getParent()->icCommunicator()->commRank() << std::endl;
   MPI_Barrier(getParent()->icCommunicator()->communicator());
   //std::cout << "leaving MPI_Barrier in HyPerConn::initializeWeights: " << this->name << ", rank = " << getParent()->icCommunicator()->commRank() << std::endl;
#endif // PV_USE_MPI
#endif // USE_SHMGET
   initNormalize(); // Sets normalizeMethod; derived-class methods that override initNormalize must also set normalizeMethod
   normalizeWeights();
#ifdef OBSOLETE // Marked obsolete April 11, 2013.  Implementing the new NormalizeBase class hierarchy.
      for(int arborId=0; arborId<numberOfAxonalArborLists(); arborId++) {
         int status = normalizeWeights(patches ? patches[arborId] : NULL, dataStart, numPatches, arborId);
         if (status == PV_BREAK) break;
      } // arborId
#endif // OBSOLETE
   return patches;
}

int HyPerConn::allocateDataStructures() {
   initNumWeightPatches();
   initNumDataPatches();
   initPatchToDataLUT();
   initializeDelays(fDelayArray, delayArraySize);

   if (pvpatchAccumulateType == ACCUMULATE_STOCHASTIC) {
      bool from_post = getUpdateGSynFromPostPerspective();
      if (from_post) {
         randState = new Random(parent, postSynapticLayer()->getLayerLoc(), false/*isExtended*/);
      }
      else {
         randState = new Random(parent, preSynapticLayer()->getLayerLoc(), true/*isExtended*/);
      }
//      const PVLayerLoc * loc = (from_post ? post : pre)->getLayerLoc();
//      int nx = loc->nx;
//      int ny = loc->ny;
//      int nf = loc->nf;
//      if (!from_post) {
//         int nb2 = 2*loc->nb;
//         nx += nb2;
//         ny += nb2;
//      }
//      int neededRNGSeeds = from_post ? post->getNumGlobalNeurons() : pre->getNumGlobalExtended();
//      rngSeedBase = parent->getObjectSeed(neededRNGSeeds);
//      rnd_state = (uint4 *) malloc((size_t)neededRNGSeeds*sizeof(uint4));
//      for (int y=0; y<ny; y++) {
//         int localIndex = kIndex(0,y,0,nx,ny,nf);
//         int globalIndex = globalIndexFromLocal(localIndex, *loc);
//         cl_random_init(&rnd_state[localIndex], nx*nf, rngSeedBase+(unsigned int) globalIndex);
//      }
   }

   if (plasticityFlag) {
      if (parent->getCheckpointReadFlag()==false && weightUpdateTime < parent->simulationTime()) {
         while(weightUpdateTime <= parent->simulationTime()) {weightUpdateTime += weightUpdatePeriod;}
         if (parent->columnId()==0) {
            fprintf(stderr, "Warning: initialWeightUpdateTime of %s \"%s\" less than simulation start time.  Adjusting weightUpdateTime to %f\n",
                  parent->parameters()->groupKeywordFromName(name), name, weightUpdateTime);
         }
      }
      lastUpdateTime = weightUpdateTime - parent->getDeltaTime();
   }

   int status = constructWeights(filename);

   return status;
}

uint4 * HyPerConn::getRandState(int index) {
   uint4 * state = NULL;
   if (pvpatchAccumulateType==ACCUMULATE_STOCHASTIC) {
      state = randState->getRNG(index);
   }
   return state;
}


InitWeights * HyPerConn::getDefaultInitWeightsMethod(const char * keyword) {
   fprintf(stderr, "weightInitType not set or unrecognized.  Using default method.\n");
   InitWeights * initWeightsObj = new InitWeights();
   return initWeightsObj;
}

InitWeights * HyPerConn::handleMissingInitWeights(PVParams * params) {
   return new InitWeights();
}

// TODO: use templates and std::cerr for handleUnnecessary*Parameter functions
void HyPerConn::handleUnnecessaryIntParameter(const char * paramName, int correctValue) {
   int status = PV_SUCCESS;
   PVParams * params = parent->parameters();
   const char * classname = params->groupKeywordFromName(name);
   if (params->present(name, paramName)) {
      if (parent->columnId()==0) {
         fprintf(stderr, "Connection \"%s\" warning: class \"%s\" does not use parameter %s but determines from other param(s).\n",
               name, classname, paramName);
      }
      if ((int) params->value(name, paramName)!=correctValue) {
         status = PV_FAILURE;
         if (parent->columnId()==0) {
            fprintf(stderr, "    Value %d is inconsistent with correct value %d.  Exiting.\n",
                  (int) params->value(name, paramName), correctValue);
         }
      }
   }
   MPI_Barrier(parent->icCommunicator()->communicator());
   if (status != PV_SUCCESS) exit(EXIT_FAILURE);
}

void HyPerConn::handleUnnecessaryFloatingPointParameter(const char * paramName, double correctValue) {
   int status = PV_SUCCESS;
   PVParams * params = parent->parameters();
   const char * classname = params->groupKeywordFromName(name);
   if (params->present(name, paramName)) {
      if (parent->columnId()==0) {
         fprintf(stderr, "Connection \"%s\" warning: class \"%s\" does not use parameter %s but determines it from other param(s).\n",
               name, classname, paramName);
      }
      if (params->value(name, paramName)!=correctValue) {
         status = PV_FAILURE;
         if (parent->columnId()==0) {
            fprintf(stderr, "    Value %f is inconsistent with correct value %f.  Exiting.\n",
                  params->value(name, paramName), correctValue);
         }
      }
   }
   MPI_Barrier(parent->icCommunicator()->communicator());
   if (status != PV_SUCCESS) exit(EXIT_FAILURE);
}

void HyPerConn::handleUnnecessaryParameterString(const char * paramName, const char * correctValue, bool case_insensitive) {
   int status = PV_SUCCESS;
   PVParams * params = parent->parameters();
   const char * classname = params->groupKeywordFromName(name);
   if (params->stringPresent(name, paramName)) {
      char * param_value = strdup(params->stringValue(name, paramName)); // need mutable strings if case-insensitive;
      if (param_value == NULL) {
         fprintf(stderr, "%s \"%s\" error: Rank %d process unable to copy parameter string value: %s.\n", classname, name, parent->columnId(), strerror(errno));
         exit(EXIT_FAILURE);
      }
      char * correct_value = strdup(correctValue); // more convenient to use same string even if case-sensitive.
      if (correct_value == NULL) {
         fprintf(stderr, "%s \"%s\" error: Rank %d process unable to correct string value: %s.\n", classname, name, parent->columnId(), strerror(errno));
         exit(EXIT_FAILURE);
      }
      if (case_insensitive) {
         for (char * c = param_value; *c!='\0'; c++) {
            *c = (char) tolower((int) *c);
         }
         for (char * c = correct_value; *c!='\0'; c++) {
            *c = (char) tolower((int) *c);
         }
      }
      if (parent->columnId()==0) {
         fprintf(stderr, "Connection \"%s\" warning: class \"%s\" does not use string parameter %s but determines it from other param(s).\n",
               name, classname, paramName);
      }
      if (strcmp(param_value,correct_value)) {
         status = PV_FAILURE;
         if (parent->columnId()==0) {
            fprintf(stderr, "    Value \"%s\" is inconsistent with correct value \"%s\".  Exiting.\n",
                  params->stringValue(name, paramName), correct_value);
         }
      }
      free(param_value);
      free(correct_value);
   }
   MPI_Barrier(parent->icCommunicator()->communicator());
   if (status != PV_SUCCESS) exit(EXIT_FAILURE);
}

#ifdef PV_USE_OPENCL
void HyPerConn::initIgnoreGPUFlag() {
   PVParams * params = parent->parameters();
   ignoreGPUflag=false;
   ignoreGPUflag = params->value(name, "ignoreGPU", ignoreGPUflag);
}
//this method sets up GPU related variables and calls the
//initializeThreadBuffers and initializeThreadKernels
int HyPerConn::initializeGPU() {
   initIgnoreGPUFlag();
   //if((gpuAccelerateFlag)&&(ignoreGPUflag)) post->copyChannelToDevice();
   int totwait = numberOfAxonalArborLists();
   evRecvSynWaitList = (cl_event *) malloc(totwait*sizeof(cl_event));
   numWait = 0;

   nxl = 16;
   nyl = 8;

   const char* kernel_name = "HyPerLayer_recv_synaptic_input";
   initializeThreadBuffers(kernel_name);
   initializeThreadKernels(kernel_name);
   //pre->initializeDataStoreThreadBuffers();

   return PV_SUCCESS;
}
/**
 * Initialize OpenCL buffers.  This must be called after weights have
 * been allocated.
 */
int HyPerConn::initializeThreadBuffers(const char * kernel_name)
{
   int status = CL_SUCCESS;

   const size_t size = getNumDataPatches() * nxp*nyp*nfp * sizeof(pvdata_t);

   CLDevice * device = parent->getCLDevice();

   clWeights = NULL;
   if (numAxonalArborLists > 0) {
      clWeights = (CLBuffer **) malloc(numAxonalArborLists*sizeof(CLBuffer *));
      assert(clWeights != NULL);
   }

   // create device buffers for weights
   //
   for (int arbor = 0; arbor < numAxonalArborLists; arbor++) {
      pvdata_t * wBuf = get_wDataStart(arbor);
      clWeights[arbor] = device->createReadBuffer(size, (void*)wBuf);
   }

//   gSynSemaphors=(int*)calloc(sizeof(int), post->getNumNeurons());
//   const size_t sizeSem = sizeof(int)*post->getNumNeurons();
//   clGSynSemaphors= device->createBuffer(CL_MEM_COPY_HOST_PTR, sizeSem, gSynSemaphors);
   //float *gSemaphors=(float*)calloc(sizeof(float), postLoc->getNumNeurons();

   const int numWeightPatches = getNumWeightPatches();
   const size_t lutSize = numWeightPatches*sizeof(int);
   int * lutpointer = getLUTpointer();
   bool freelutpointer=false;
   if(lutpointer==NULL) {
      lutpointer = (int *) calloc(sizeof(int), numWeightPatches);
      freelutpointer=true;
      lutpointer[0]=-1;
   }
   clPatch2DataLookUpTable = device->createReadBuffer(lutSize, (void*)lutpointer);
   if(freelutpointer) free(lutpointer);

   //tell the presynaptic layer to copy its data store to the GPU after publishing
   pre->tellLayerToCopyDataStoreCLBuffer(/*&evCopyDataStore*/);

   return status;
}

int HyPerConn::initializeThreadKernels(const char * kernel_name)
{
   char kernelPath[PV_PATH_MAX+128];
   char kernelFlags[PV_PATH_MAX+128];

   int status = CL_SUCCESS;
   CLDevice * device = parent->getCLDevice();

   const char * pvRelPath = "../PetaVision";
   sprintf(kernelPath, "%s/%s/src/kernels/%s.cl", parent->getPath(), pvRelPath, kernel_name);
   sprintf(kernelFlags, "-D PV_USE_OPENCL -cl-fast-relaxed-math -I %s/%s/src/kernels/", parent->getPath(), pvRelPath);

   // create kernels
   //

   krRecvSyn = device->createKernel(kernelPath, kernel_name, kernelFlags);

   const PVLayerLoc * preLoc  = pre-> getLayerLoc();
   const PVLayerLoc * postLoc = post->getLayerLoc();

   int argid = 0;

   status |= krRecvSyn->setKernelArg(argid++, preLoc->nx);
   status |= krRecvSyn->setKernelArg(argid++, preLoc->ny);
   status |= krRecvSyn->setKernelArg(argid++, preLoc->nf);
   status |= krRecvSyn->setKernelArg(argid++, preLoc->nb);

   status |= krRecvSyn->setKernelArg(argid++, nxp);
   status |= krRecvSyn->setKernelArg(argid++, nyp);
   status |= krRecvSyn->setKernelArg(argid++, nfp);

   float fScale = (float)postLoc->nf/(float)preLoc->nf;
   float xScale = (float)postLoc->nx/(float)preLoc->nx;
   float yScale = (float)postLoc->ny/(float)preLoc->ny;
   status |= krRecvSyn->setKernelArg(argid++, fScale);
   status |= krRecvSyn->setKernelArg(argid++, xScale);
   status |= krRecvSyn->setKernelArg(argid++, yScale);

   clArgIdOffset = argid;  // offset into activity buffer (with delay)
   argid++;
   status |= krRecvSyn->setKernelArg(argid++, clPatch2DataLookUpTable);
   // activity buffer from DataStore
   clArgIdDataStore=argid;
   argid++;
   clArgIdWeights = argid; // weights
   status |= krRecvSyn->setKernelArg(argid++, clWeights[0]);
   // update variable, GSyn
   status |= krRecvSyn->setKernelArg(argid++, post->getNumNeurons()*getChannel());
   status |= krRecvSyn->setKernelArg(argid++, post->getChannelCLBuffer());

   return status;
}
#endif // PV_USE_OPENCL

int HyPerConn::checkPVPFileHeader(Communicator * comm, const PVLayerLoc * loc, int params[], int numParams)
{
   // use default header checker
   //
   return pvp_check_file_header(comm, loc, params, numParams);
}

int HyPerConn::checkWeightsHeader(const char * filename, const int * wgtParams)
{
   // extra weight parameters
   //
   const int nxpFile = wgtParams[NUM_BIN_PARAMS + INDEX_WGT_NXP];
   if (nxp != nxpFile) {
      fprintf(stderr,
              "ignoring nxp = %i in HyPerConn %s, using nxp = %i in binary file %s\n",
              nxp, name, nxpFile, filename);
      nxp = nxpFile;
   }

   const int nypFile = wgtParams[NUM_BIN_PARAMS + INDEX_WGT_NYP];
   if (nyp != nypFile) {
      fprintf(stderr,
              "ignoring nyp = %i in HyPerConn %s, using nyp = %i in binary file %s\n",
              nyp, name, nypFile, filename);
      nyp = nypFile;
   }

   nfp = wgtParams[NUM_BIN_PARAMS + INDEX_WGT_NFP];
   // const int nfpFile = wgtParams[NUM_BIN_PARAMS + INDEX_WGT_NFP];
   // if (nfp != nfpFile) {
   //    fprintf(stderr,
   //            "ignoring nfp = %i in HyPerConn %s, using nfp = %i in binary file %s\n",
   //            nfp, name, nfpFile, filename);
   //    nfp = nfpFile;
   // }
   return 0;
}

int HyPerConn::writeWeights(double time, bool last)
{
   const int numPatches = getNumWeightPatches();
   return writeWeights(wPatches, wDataStart, numPatches, NULL, time, writeCompressedWeights, last);
}

int HyPerConn::writeWeights(const char * filename) {
   return writeWeights(wPatches, wDataStart, getNumWeightPatches(), filename, parent->simulationTime(), writeCompressedWeights, true);
}

int HyPerConn::writeWeights(PVPatch *** patches, pvdata_t ** dataStart, int numPatches,
      const char * filename, double timef, bool compressWeights, bool last) {
   int status = PV_SUCCESS;
   char path[PV_PATH_MAX];

   // if (patches == NULL) return PV_SUCCESS; // KernelConn::writeWeights will call with patches set to NULL.

   float minVal = FLT_MAX;
   float maxVal = -FLT_MAX;
   for(int arbor=0; arbor<this->numberOfAxonalArborLists(); arbor++) {
      float minVal1 = minWeight(arbor);
      if( minVal1 < minVal ) minVal = minVal1;
      float maxVal1 = maxWeight(arbor);
      if( maxVal1 > maxVal ) maxVal = maxVal1;
   }

   const PVLayerLoc * loc = pre->getLayerLoc();

   // Is "_last" obsolete?  The data that used to be written to the _last files are now handled by checkpointing.
   const char * laststr = last ? "_last" : "";
   int chars_needed = 0;
   if (filename == NULL) {
      assert(parent->includeConnectionName()<=2 && parent->includeConnectionName()>=0);
      switch(parent->includeConnectionName()) {
      case 0:
         chars_needed = snprintf(path, PV_PATH_MAX, "%s/w%d%s.pvp", parent->getOutputPath(), getConnectionId(), laststr);
         break;
      case 1:
         chars_needed = snprintf(path, PV_PATH_MAX, "%s/w%d_%s%s.pvp", parent->getOutputPath(), getConnectionId(), name, laststr);
         break;
      case 2:
         chars_needed = snprintf(path, PV_PATH_MAX, "%s/%s%s.pvp", parent->getOutputPath(), name, laststr);
         break;
      default:
         assert(0);
         break;
      }
   }
   else {
      chars_needed = snprintf(path, PV_PATH_MAX, "%s", filename);
   }
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "HyPerConn::writeWeights in connection \"%s\": path is too long (it would be cut off as \"%s\")\n", name, path);
      abort();
   }

   Communicator * comm = parent->icCommunicator();

   bool append = last ? false : ioAppend;

	status = PV::writeWeights(path, comm, (double) timef, append, loc, nxp, nyp,
			nfp, minVal, maxVal, patches, dataStart, numPatches,
			numberOfAxonalArborLists(), compressWeights, fileType);
   assert(status == 0);

   return status;
}

int HyPerConn::writeTextWeights(const char * filename, int k)
{
   if (parent->icCommunicator()->commSize()>1) {
      fprintf(stderr, "writeTextWeights error for connection \"%s\": writeTextWeights is not compatible with MPI", name);
      abort();
      // NOTE : if run under MPI when more than one process sees the same file system, the contending processes will clobber each other.
   }
   PV_Stream * pvstream = NULL;

   if (filename != NULL) {
      char outfile[PV_PATH_MAX];
      snprintf(outfile, PV_PATH_MAX-1, "%s/%s", parent->getOutputPath(), filename);
      pvstream = PV_fopen(outfile, "w");
   }
   else {
      pvstream = PV_stdout();
   }
   if (pvstream == NULL) {
     fprintf(stderr, "writeWeights: ERROR opening file \"%s\"\n", filename);
     return PV_FAILURE;
   }

   FILE * fd = pvstream->fp;
   fprintf(fd, "Weights for connection \"%s\", neuron %d\n", name, k);
   fprintf(fd, "   (kxPre,kyPre,kfPre)   = (%i,%i,%i)\n",
           kxPos(k,pre->getLayerLoc()->nx + 2*pre->getLayerLoc()->nb,
                 pre->getLayerLoc()->ny + 2*pre->getLayerLoc()->nb, pre->getLayerLoc()->nf),
           kyPos(k,pre->getLayerLoc()->nx + 2*pre->getLayerLoc()->nb,
                 pre->getLayerLoc()->ny + 2*pre->getLayerLoc()->nb, pre->getLayerLoc()->nf),
           featureIndex(k,pre->getLayerLoc()->nx + 2*pre->getLayerLoc()->nb,
                 pre->getLayerLoc()->ny + 2*pre->getLayerLoc()->nb, pre->getLayerLoc()->nf) );
   fprintf(fd, "   (nxp,nyp,nfp)   = (%i,%i,%i)\n", (int) nxp, (int) nyp, (int) nfp);
   fprintf(fd, "   pre  (nx,ny,nf) = (%i,%i,%i)\n",
           pre->getLayerLoc()->nx, pre->getLayerLoc()->ny, pre->getLayerLoc()->nf);
   fprintf(fd, "   post (nx,ny,nf) = (%i,%i,%i)\n",
           post->getLayerLoc()->nx, post->getLayerLoc()->ny, post->getLayerLoc()->nf);
   fprintf(fd, "\n");


   //int arbor = 0;
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

//Input delay is in ms
void HyPerConn::setDelay(int arborId, float delay) {
   assert(arborId>=0 && arborId<numAxonalArborLists);
   int intDelay = round(delay/parent->getDeltaTime());
   if (fmod(delay, parent->getDeltaTime()) != 0){
      float actualDelay = intDelay * parent->getDeltaTime();
      std::cerr << name << ": A delay of " << delay << " will be rounded to " << actualDelay << "\n";
   }
   delays[arborId] = (int)(round(delay / parent->getDeltaTime()));
}

#ifdef PV_USE_OPENCL
int HyPerConn::deliverOpenCL(Publisher * pub, const PVLayerCube * cube)
{
   int status = PV_SUCCESS;


   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const size_t nxex = (preLoc->nx + 2*preLoc->nb)*preLoc->nf;
   const size_t nyex = preLoc->ny + 2*preLoc->nb;
   while((nxex%nxl!=0)&&(nxl>1)) {nxl--;}
   while((nyex%nyl!=0)&&(nyl>1)) {nyl--;}

   status |= krRecvSyn->setKernelArg(clArgIdDataStore, pre->getLayerDataStoreCLBuffer());

   status |= pre->waitForDataStoreCopy();


   // for all numextended in pre

   post->startTimer();


   int arborCnt=numberOfAxonalArborLists();
   for (int arbor = 0; arbor < arborCnt; arbor++) {
      int delay = getDelay(arbor);
      size_t activityOffset = pre->getLayerDataStoreOffset(delay);
      status |= krRecvSyn->setKernelArg(clArgIdOffset, activityOffset/sizeof(pvdata_t)); //need to convert offset to an array index offset
      status |= krRecvSyn->setKernelArg(clArgIdWeights, clWeights[arbor]);
      status |= krRecvSyn->run(nxex, nyex, nxl, nyl, 0, NULL, &evRecvSynWaitList[arbor]);
      numWait++;
   }

   // TODO - use events properly
   status |= clWaitForEvents(numWait, evRecvSynWaitList);
   for (int i = 0; i < numWait; i++) {
      clReleaseEvent(evRecvSynWaitList[i]);
   }
   numWait = 0;

   post->stopTimer();

   int arborId=0;
   int delay = getDelay(arborId);
   pub->readData(delay);
   const PVLayerLoc * postLoc = post->getLayerLoc();
   //define global location:
   int kx=nxex/2; int ky=nyex/2;
   //int kPre=ky*nxex+kx;
   int gstart=0;//post->getNumNeurons()*getChannel();
   float *gTempBuf=(float*)calloc(sizeof(float), post->getNumNeurons());
   int * lutpointer = getLUTpointer();
   const int numWeightPatches = getNumWeightPatches();
   bool freelutpointer=false;
   if(lutpointer==NULL) {
      lutpointer = (int *) calloc(sizeof(int), numWeightPatches);
      freelutpointer=true;
      lutpointer[0]=-1;
   }
   printf("nxex %lu\n",nxex);
   printf("nyex %lu\n",nyex);
   printf("nxl %lu\n",nxl);
   printf("nyl %lu\n",nyl);
   printf("nxex/nxl %lu\n",nxex/nxl);
   printf("nyex/nyl %lu\n",nyex/nyl);
   for(kx=0;kx<(int)(nxex/nxl);kx++) {
      for(ky=0;ky<(int)(nyex/nyl);ky++) {

         for(int lidx=0;lidx<(int)nxl;lidx++) {
            for(int lidy=0;lidy<(int)nyl;lidy++) {
               HyPerLayer_recv_synaptic_input(kx*nxl+nxl/2, ky*nyl+nyl/2, lidx, lidy, nxl, nyl,
                     preLoc->nx, preLoc->ny, preLoc->nf, preLoc->nb, nxp, nyp, nfp,
                     (float)postLoc->nf/(float)preLoc->nf,(float)postLoc->nx/(float)preLoc->nx,(float)postLoc->ny/(float)preLoc->ny,
                     0, lutpointer, cube->data, get_wDataStart(arborId), gstart, gTempBuf);
               //free(tempBuf);
            }
         }

      }
   }
   if(freelutpointer) {free(lutpointer);lutpointer=NULL;}
   //copy back to G:
//   float xScale = (float)postLoc->nx/(float)preLoc->nx;
//   float yScale = (float)postLoc->ny/(float)preLoc->ny;
//   const int kPostX = (int)(xScale*kx) - (int)(xScale*preLoc->nb); // kPostX==0 is left boundary non-extended
//   const int kPostY = (int)(yScale*ky) - (int)(yScale*preLoc->nb); // kPostY==0 is top  boundary non-extended
//   const int gStride = xScale*preLoc->nx;
//   int tempBufStride=nxl+nxp*nfp;
//
//   const int gx=kPostX-nxp/2;
//   const int gy=kPostY-nyp/2;
//   for(int clidx=0;clidx<nxp;clidx++){
//      for(int clidy=0;clidy<nyp;clidy++){
//         gTempBuf[(gy+clidy)*gStride + gx+clidx]+=tempBuf[clidy*tempBufStride + clidx];
//      }
//   }
   //free(tempBuf);

   //cl_event   tmpcopybackGevList;         // event list
   //cl_event   tmpevUpdate;
//   post->getChannelCLBuffer(getChannel())->copyFromDevice(1, &evRecvSyn, &tmpcopybackGevList);
//   status |= clWaitForEvents(1, &tmpcopybackGevList);
//   clReleaseEvent(tmpcopybackGevList);
   post->copyGSynFromDevice();

   //copyChannelExcFromDevice();
   float *gTempBuf2=getGSynPatchStart(0, arborId);

   int errcnt=0;
   for(int ix=0;ix<postLoc->nx; ix++) {
      for(int iy=0;iy<postLoc->ny; iy++) {
         if(fabs(gTempBuf[iy*postLoc->nx+ix]-gTempBuf2[iy*postLoc->nx+ix])>0.00001){
            printf("mismatch! C function version: %f \n",gTempBuf[iy*postLoc->nx+ix]);
            printf("opencl function version: %f \n",gTempBuf2[iy*postLoc->nx+ix]);
            printf("at loc x: %d y %d \n",ix, iy);
            printf("kpre %d \n",ix+preLoc->nb+ (iy+preLoc->nb)*(preLoc->nx*preLoc->nf + 2*preLoc->nb));
            errcnt++;
            if(errcnt>10) exit(1);
         }
//         if(gTempBuf[iy*postLoc->nx+ix]==4){
//            printf("value = 4! lutpointer: %f \n",gTempBuf[iy*postLoc->nx+ix]);
//            printf("opencl function version: %f \n",gTempBuf2[iy*postLoc->nx+ix]);
//            printf("at loc x: %d y %d \n",ix, iy);
//            printf("kpre %d \n",ix+preLoc->nb+ (iy+preLoc->nb)*(preLoc->nx*preLoc->nf + 2*preLoc->nb));
//            errcnt++;
//            if(errcnt>10) exit(1);
//         }
//         if((gTempBuf[iy*postLoc->nx+ix]>25)||(gTempBuf2[iy*postLoc->nx+ix]>25)){
//            printf("not equal to row! C function version: %f \n",gTempBuf[iy*postLoc->nx+ix]);
//            printf("opencl function version: %f \n",gTempBuf2[iy*postLoc->nx+ix]);
//            printf("at loc x: %d y %d \n",ix, iy);
//            errcnt++;
//            if(errcnt>500) exit(1);
//         }
//         if((gTempBuf[iy*postLoc->nx+ix]!=gTempBuf2[iy*postLoc->nx+ix])&&
//               (gTempBuf[iy*postLoc->nx+ix]!=0)){
//            printf("mismatch (2)! C function version: %d \n",gTempBuf[iy*postLoc->nx+ix]);
//            printf("opencl function version: %d \n",gTempBuf2[iy*postLoc->nx+ix]);
//            printf("at loc x: %d y %d \n",ix, iy);
//            errcnt++;
//            if(errcnt>10) exit(1);
//         }
//         if((gTempBuf[iy*postLoc->nx+ix]==gTempBuf2[iy*postLoc->nx+ix])&&
//               (gTempBuf[iy*postLoc->nx+ix]!=0)){
//            printf("nonzero match found! C function version: %d \n",gTempBuf[iy*postLoc->nx+ix]);
//            printf("opencl function version: %d \n",gTempBuf2[iy*postLoc->nx+ix]);
//            printf("at loc x: %d y %d \n",ix, iy);
//            //errcnt++;
//            //if(errcnt>10) exit(1);
//         }
      }
   }
   free(gTempBuf);

   return status;
}
#endif // PV_USE_OPENCL

#ifdef OBSOLETE // Marked obsolete July 25, 2013.  recvSynapticInput is now called by recvAllSynapticInput, called by HyPerCol, so deliver andtriggerReceive aren't needed.
int HyPerConn::deliver(Publisher * pub, const PVLayerCube * cube, int neighbor)
{
#ifdef DEBUG_OUTPUT
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   printf("[%d]: HyPerConn::deliver: neighbor=%d cube=%p post=%p this=%p\n", rank, neighbor, cube, post, this);
   fflush(stdout);
#endif // DEBUG_OUTPUT

#ifdef PV_USE_OPENCL
   if((gpuAccelerateFlag)&&(!ignoreGPUflag)) {
      deliverOpenCL(pub, cube);
   }
   else {
      //if((gpuAccelerateFlag)&&(ignoreGPUflag)) post->copyChannelFromDevice(getChannel());
      if((gpuAccelerateFlag)&&(ignoreGPUflag))
         post->copyGSynFromDevice();
#endif // PV_USE_OPENCL
      for(int arborId=0;arborId<numberOfAxonalArborLists();arborId++) {
         int delay = getDelay(arborId);
         pub->readData(delay);
         int status = post->recvSynapticInput(this, cube, arborId);
         if (status == PV_BREAK) break;
         assert(status == PV_SUCCESS);
      }
#ifdef PV_USE_OPENCL
      if((gpuAccelerateFlag)&&(ignoreGPUflag))
         post->copyGSynToDevice();

   }
#endif // PV_USE_OPENCL

#ifdef DEBUG_OUTPUT
   printf("[%d]: HyPerConn::delivered: \n", rank);
   fflush(stdout);
#endif // DEBUG_OUTPUT
   return 0;
}
#endif // OBSOLETE

int HyPerConn::checkpointRead(const char * cpDir, double * timef) {
   clearWeights(get_wDataStart(), getNumDataPatches(), nxp, nyp, nfp);

   char path[PV_PATH_MAX];
   int status = checkpointFilename(path, PV_PATH_MAX, cpDir);
   assert(status==PV_SUCCESS);
   InitWeights * weightsInitObject = new InitWeights();
   weightsInitObject->initializeWeights(wPatches, get_wDataStart(), path, this, timef);
   free(weightsInitObject); weightsInitObject = NULL;

   status = parent->readScalarFromFile(cpDir, getName(), "lastUpdateTime", &lastUpdateTime, lastUpdateTime);
   assert(status == PV_SUCCESS);
   status = parent->readScalarFromFile(cpDir, getName(), "weightUpdateTime", &weightUpdateTime, weightUpdateTime);
   assert(status == PV_SUCCESS);
   if (this->plasticityFlag &&  weightUpdateTime<parent->simulationTime()) {
      // simulationTime() may have been changed by HyPerCol::checkpoint, so this repeats the sanity check on weightUpdateTime in allocateDataStructures
      while(weightUpdateTime <= parent->simulationTime()) {weightUpdateTime += weightUpdatePeriod;}
      if (parent->columnId()==0) {
         fprintf(stderr, "Warning: initialWeightUpdateTime of %s \"%s\" less than simulation start time.  Adjusting weightUpdateTime to %f\n",
               parent->parameters()->groupKeywordFromName(name), name, weightUpdateTime);
      }
   }

   status = parent->readScalarFromFile(cpDir, getName(), "nextWrite", &writeTime, writeTime);
   assert(status == PV_SUCCESS);

   return status;
}

int HyPerConn::checkpointWrite(const char * cpDir) {
   char filename[PV_PATH_MAX];
   int status = checkpointFilename(filename, PV_PATH_MAX, cpDir);
   assert(status==PV_SUCCESS);
   status = writeWeights(wPatches, wDataStart, getNumWeightPatches(), filename, parent->simulationTime(), writeCompressedCheckpoints, /*last*/true);
   assert(status==PV_SUCCESS);
   status = parent->writeScalarToFile(cpDir, getName(), "nextWrite", writeTime);
   assert(status==PV_SUCCESS);
   status = parent->writeScalarToFile(cpDir, getName(), "lastUpdateTime", lastUpdateTime);
   assert(status==PV_SUCCESS);
   status = parent->writeScalarToFile(cpDir, getName(), "weightUpdateTime", weightUpdateTime);
   assert(status==PV_SUCCESS);
   return status;
}

int HyPerConn::checkpointFilename(char * cpFilename, int size, const char * cpDir) {
   int chars_needed = snprintf(cpFilename, size, "%s/%s_W.pvp", cpDir, name);
   if(chars_needed >= PV_PATH_MAX) {
      if ( parent->icCommunicator()->commRank()==0 ) {
         fprintf(stderr, "HyPerConn::checkpointFilename error: path \"%s/%s_W.pvp\" is too long.\n", cpDir, name);
      }
      abort();
   }
   return PV_SUCCESS;
}

#ifdef OBSOLETE // Marked obsolete May 1, 2013.  Use HyPerCol's writeScalarToFile instead
int HyPerConn::writeScalarFloat(const char * cp_dir, const char * val_name, double val) {
   int status = PV_SUCCESS;
   if (parent->columnId()==0)  {
      char filename[PV_PATH_MAX];
      int chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_%s.bin", cp_dir, name, val_name);
      if (chars_needed >= PV_PATH_MAX) {
         fprintf(stderr, "writeScalarFloat error: path %s/%s_%s.bin is too long.\n", cp_dir, name, val_name);
         abort();
      }
      PV_Stream * writeTimeStream = PV_fopen(filename, "w");
      if (writeTimeStream==NULL) {
         fprintf(stderr, "HyPerLayer::checkpointWrite error: unable to open path %s for writing.\n", filename);
         abort();
      }
      int num_written = PV_fwrite(&val, sizeof(val), 1, writeTimeStream);
      if (num_written != 1) {
         fprintf(stderr, "HyPerLayer::checkpointWrite error while writing to %s.\n", filename);
         abort();
      }
      PV_fclose(writeTimeStream);
      chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_%s.txt", cp_dir, name, val_name);
      assert(chars_needed < PV_PATH_MAX);
      writeTimeStream = PV_fopen(filename, "w");
      if (writeTimeStream==NULL) {
         fprintf(stderr, "HyPerLayer::checkpointWrite error: unable to open path %s for writing: %s\n", filename, strerror(errno));
         abort();
      }
      fprintf(writeTimeStream->fp, "%f\n", val);
      PV_fclose(writeTimeStream);
   }
   return status;
}
#endif // OBSOLETE

float HyPerConn::maxWeight(int arborId)
{
   const int num_data_patches = getNumDataPatches();
   float max_weight = -FLT_MAX;
   for (int i_weight = 0; i_weight < num_data_patches; i_weight++) {
      pvdata_t * w_data = this->get_wData(arborId, i_weight);
      PVPatch * w_patch = this->getWeights(i_weight, arborId);
      int num_weights = this->fPatchSize() * w_patch->nx * w_patch->ny;
      for (int iWeight = 0; iWeight < num_weights; iWeight++) {
         max_weight = (max_weight > w_data[iWeight]) ? max_weight
               : w_data[iWeight];
      }
   }
   return max_weight;
}

int HyPerConn::insertProbe(BaseConnectionProbe * p)
{
   if(p->getTargetConn() != this) {
      fprintf(stderr, "HyPerConn \"%s\": insertProbe called with probe %p, whose targetConn is not this connection.  Probe was not inserted.\n", name, p);
      return numProbes;
   }
   for( int i=0; i<numProbes; i++ ) {
      if( p == probes[i] ) {
         fprintf(stderr, "HyPerConn \"%s\": insertProbe called with probe %p, which has already been inserted as probe %d.\n", name, p, i);
         return numProbes;
      }
   }

   BaseConnectionProbe ** tmp;
   tmp = (BaseConnectionProbe **) malloc((numProbes + 1) * sizeof(BaseConnectionProbe *));
   assert(tmp != NULL);

   for (int i = 0; i < numProbes; i++) {
      tmp[i] = probes[i];
   }
   delete probes;

   probes = tmp;
   probes[numProbes] = p;

   return ++numProbes;
}

int HyPerConn::outputState(double timef, bool last)
{
   int status = 0;
   io_timer->start();

   if( !last ) {
      for (int i = 0; i < numProbes; i++) {
         probes[i]->outputState(timef);
      }
   }

   if (last) {
      status = writeWeights(timef, last);
      assert(status == 0);
   }
   else if ( (timef >= writeTime) && (writeStep >= 0) ) {
      writeTime += writeStep;

      status = writeWeights(timef, last);
      assert(status == 0);

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
   if(triggerFlag){
      assert(triggerLayer);
      //Never update on timestep 1 if triggered
      if(time == dt){
         return false;
      }
      double nextUpdateTime = triggerLayer->getNextUpdateTime();
      //never update flag
      if(nextUpdateTime == -1){
         return false;
      }
      //Check for equality
      if(abs(time - (nextUpdateTime - triggerOffset)) < (dt/2)){
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


int HyPerConn::updateStateWrapper(double time, double dt){
   int status = PV_SUCCESS;

   if(needUpdate(time, dt)){
      //std::cout << "Connection " << name << " updating on timestep " << time << "\n";
      status = updateState(time, dt);
      //Update lastUpdateTime
      lastUpdateTime = time;
      computeNewWeightUpdateTime(time, weightUpdateTime);

      //Sanity check, take this out once convinced layer's nextUpdateTime is the same as weightUpdateTime
      //No way to make this assertion, cause nextUpdateTime/weightUpdateTime updates happen at different times
      //if(triggerFlag){
      //   if(weightUpdateTime != triggerLayer->getNextUpdateTime()){
      //      std::cout << "Layer " << name << ": weightUpdateTime (" << weightUpdateTime << ") and layer's getNextUpdateTime (" << triggerLayer->getNextUpdateTime() << ") mismatch\n";
      //   }
      //   //assert(weightUpdateTime == triggerLayer->getNextUpdateTime());
      //}
   }
   return status;
}

int HyPerConn::updateState(double time, double dt)
{
   int status = PV_SUCCESS;
   if( !plasticityFlag ) {
      return status;
   }
   update_timer->start();

   //const int arborId = 0;       // assume only one for now
   for(int arborId=0;arborId<numberOfAxonalArborLists();arborId++) {
      status = calc_dW(arborId);        // Calculate changes in weights
      // TODO error handling
      status = updateWeights(arborId);  // Apply changes in weights
   }

   update_timer->stop();
   return status;
}

int HyPerConn::calc_dW(int arborId) {
   return PV_SUCCESS;
}

//
/* M (m or pDecr->data) is an extended post-layer variable
 *
 */
int HyPerConn::updateWeights(int arborId)
{
   return 0;
}

double HyPerConn::computeNewWeightUpdateTime(double time, double currentUpdateTime) {
   //Only called if placisity flag is set
   while(time >= weightUpdateTime){
      weightUpdateTime += weightUpdatePeriod;
   }
   return weightUpdateTime;
}

float HyPerConn::minWeight(int arborId)
{
   const int num_data_patches = getNumDataPatches();
   float min_weight = FLT_MAX;
   for (int i_patch = 0; i_patch < num_data_patches; i_patch++) {
      pvdata_t * w_data = this->get_wData(arborId, i_patch);
      PVPatch * w_patch = this->getWeights(i_patch, arborId);
      int num_weights = this->fPatchSize() * w_patch->nx * w_patch->ny;
      for (int iWeight = 0; iWeight < num_weights; iWeight++) {
         min_weight = (min_weight < w_data[iWeight]) ? min_weight
               : w_data[iWeight];
      }
   }
   return min_weight;
}

PVPatch * HyPerConn::getWeights(int k, int arbor)
{
   // a separate arbor/patch of weights for every neuron
   return wPatches[arbor][k];
}

int HyPerConn::createWeights(PVPatch *** patches, int nWeightPatches, int nDataPatches, int nxPatch,
      int nyPatch, int nfPatch, int arborId)
{
   // could create only a single patch with following call
   //   return createPatches(numAxonalArborLists, nxp, nyp, nfp);

   //assert(numAxonalArborLists == 1);
   assert(patches[arborId] == NULL);

   //patches = (PVPatch**) calloc(sizeof(PVPatch*), nPatches);
   //patches[arborId] = (PVPatch**) calloc(nPatches, sizeof(PVPatch*));
   if (shrinkPatches_flag || arborId == 0){
      patches[arborId] = createPatches(nWeightPatches, nxPatch, nyPatch);
      assert(patches[arborId] != NULL);
   }
   else{
      patches[arborId] = patches[0];
   }

   // allocate space for all weights at once (inplace), return pointer to beginning of weight array
   //pvdata_t * data_patches = allocWeights(patches, nDataPatches, nxPatch, nyPatch, nfPatch, arborId);
   return PV_SUCCESS;
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

int HyPerConn::clearWeights(pvdata_t ** dataStart, int numPatches, int nxp, int nyp, int nfp) {
   int status = PV_SUCCESS;
   for( int arborID = 0; arborID<numAxonalArborLists; arborID++ ) {
      if( clearWeights(dataStart[arborID], numPatches, nxp, nyp, nfp)!=PV_SUCCESS ) status = PV_FAILURE;
   }
   return status;
}

int HyPerConn::clearWeights(pvdata_t * arborDataStart, int numPatches, int nxp, int nyp, int nfp) {
   for( int w=0; w<numPatches*nxp*nyp*nfp; w++ ) {
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
#ifdef USE_SHMGET
				if (!shmget_flag) {
					if (wDataStart[arbor] != NULL) {
						free(this->wDataStart[arbor]);
					}
				} else {
					if (wDataStart[arbor] != NULL) {
						int shmget_status = shmdt(this->get_wDataStart(arbor));
						assert(shmget_status==0);
						if (shmget_owner[arbor]) {
							shmid_ds * shmget_ds = NULL;
							shmget_status = shmctl(shmget_id[arbor], IPC_RMID,
									shmget_ds);
							assert(shmget_status==0);
						}
					}
				}
#else
				if (wDataStart[arbor] != NULL) {
					free(this->wDataStart[arbor]);
				}
#endif // USE_SHMGET
			} // arbor == 0
			this->wDataStart[arbor] = NULL;
			if (!this->combine_dW_with_W_flag) {
				if (dwDataStart != NULL && dwDataStart[arbor] != NULL) {
					free(dwDataStart[arbor]);
					dwDataStart[arbor] = NULL;
				}
			}
		}  // arbor
		free(wDataStart);
		wDataStart = NULL;
		if (!this->combine_dW_with_W_flag) {
			free(dwDataStart);
		}
		dwDataStart = NULL;
	} // wDataStart != NULL

#ifdef USE_SHMGET
	if (shmget_flag) {
		free(shmget_id);
		free(shmget_owner);
	}
#endif

	if (wPostPatches != NULL) {
		for (int arborID = 0; arborID < numberOfAxonalArborLists(); arborID++) {
			if (wPostPatches[arborID] != NULL) {
				if (shrinkPatches_flag || arborID == 0) {
					deletePatches(wPostPatches[arborID]);
				}
				wPostPatches[arborID] = NULL;
			}

			if (wPostDataStart != NULL) {
				free(this->wPostDataStart[arborID]);
				this->wPostDataStart[arborID] = NULL;
			}
		}
		free(wPostPatches);
		wPostPatches = NULL;
		free(wPostDataStart);
		wPostDataStart = NULL;
	}  // wPostPatches != NULL

	if (gSynPatchStart != NULL) {
		free(gSynPatchStartBuffer); // All gSynPatchStart[k]'s were allocated together in a single malloc call.
		free(gSynPatchStart);
	}
	if (aPostOffset != NULL) {
		free(aPostOffsetBuffer); // All aPostOffset[k]'s were allocated together in a single malloc call.
		free(aPostOffset);
	}

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
   // activity is extended into margins
   //
   // Sets patches' offsets, ny, nx based on shrinking due to edge of the layer, and n{x,y}pShrunken
   // Doesn't do the shrinking turned on or off by the shrinkPatches parameter; that's handled by shrinkPatches().
#ifdef OBSOLETE // Marked obsolete Oct 10, 2013
   int numPatches = getNumWeightPatches();

   int dxPatchHead, dyPatchHead;
   dxPatchHead = (nxp - nxpShrunken)/2;
   dyPatchHead = (nyp - nypShrunken)/2;

   offsetShrunken = dxPatchHead * nfp + dyPatchHead * nxp * nfp;


   for (int kex = 0; kex < numPatches; kex++) {

      // kex is in extended frame, this makes transformations more difficult
      int kl, offset, nxpMargin, nypMargin, dxMargin, dyMargin;
      calcPatchSize(arborId, kex, &kl, &offset, &nxpMargin, &nypMargin, &dxMargin, &dyMargin);

      int dx = dxMargin;
      int dy = dyMargin;
      int nxPatch = nxpMargin;
      int nyPatch = nypMargin;

      if (nxpMargin > 0 && nypMargin > 0) {
    	  if (dxMargin > 0){ // left border
    		  if (dxMargin > dxPatchHead) {
    			  int dxShrunken = dxMargin - dxPatchHead; //dx from simulated shrunken patch head
    			  nxPatch = nxpShrunken - dxShrunken;
    			  dx = dxMargin;
    		  }
    		  else {
    			  nxPatch = nxpShrunken < nxpMargin ? nxpShrunken : nxpMargin;
    			  dx = dxPatchHead;
    		  } // dxMargin > dxPatchHead
    	  } // left border
    	  else {  // right border or inside
			  dx = dxPatchHead;
			  int physicalShrink = nxp - nxpMargin;
			  if (physicalShrink > dxPatchHead) { // right border
				  nxPatch = nxpShrunken - (physicalShrink - dxPatchHead);
			  }
			  else{ // inside
				  nxPatch = nxpShrunken;
			  }
    	  } // dxMargin > 0
    	  if (dyMargin > 0){ // left border
    		  if (dyMargin > dyPatchHead) {
    			  int dyShrunken = dyMargin - dyPatchHead; //dy from simulated shrunken patch head
    			  nyPatch = nypShrunken - dyShrunken;
    			  dy = dyMargin;
    		  }
    		  else {
    			  nyPatch = nypShrunken < nypMargin ? nypShrunken : nypMargin;
    			  dy = dyPatchHead;
    		  } // dyMargin > dyPatchHead
    	  } // left border
    	  else {  // right border or inside
			  dy = dyPatchHead;
			  int physicalShrink = nyp - nypMargin;
			  if (physicalShrink > dyPatchHead) { // right border
				  nyPatch = nypShrunken - (physicalShrink - dyPatchHead);
			  }
			  else{ // inside
				  nyPatch = nypShrunken;
			  }
    	  } // dyMargin > 0
      } //  nxPatch > 0 && nyPatch > 0
      if (nxPatch <= 0  || nyPatch <= 0) {
    	  nxPatch = 0; nyPatch = 0;
    	  dx = 0; dy = 0;
    	  dxMargin = 0; dyMargin = 0;
      }

      // arbor->offset = offset;
      const PVLayerLoc * post_loc = post->getLayerLoc();
      int offsetDiffExtended =
    		  	  (dx - dxMargin) * post_loc->nf + (dy - dyMargin) * (post_loc->nx + 2*post_loc->nb) * post_loc->nf;
  	  assert(offsetDiffExtended >= 0); assert(offsetDiffExtended <= post->getNumNeurons());
      aPostOffset[arborId][kex] = offset + offsetDiffExtended;

      // initialize the receiving gSyn variable
      int offsetDiffRestricted =
    		  	  (dx - dxMargin) * post_loc->nf + (dy - dyMargin) * post_loc->nx * post_loc->nf;
  	  assert(offsetDiffRestricted >= 0); assert(offsetDiffRestricted <= post->getNumNeurons());
  	  //GTK:  gSynPatchStart redefined as offset from head of gSynBuffer
      //pvdata_t * gSyn = post->getChannel(channel) + kl + offsetDiffRestricted;
  	  int gSyn = kl + offsetDiffRestricted;
      gSynPatchStart[arborId][kex] = gSyn;

      // adjust patch dimensions
      pvpatch_adjust(getWeights(kex,arborId), sxp, syp, nxPatch, nyPatch, dx, dy);

   } // loop over patches
#endif // OBSOLETE

   const int nxPre = pre->getLayerLoc()->nx;
   const int nyPre = pre->getLayerLoc()->ny;
   const int nfPre = pre->getLayerLoc()->nf;
   const int nbPre = pre->getLayerLoc()->nb;
   const int nxPost = post->getLayerLoc()->nx;
   const int nyPost = post->getLayerLoc()->ny;
   const int nfPost = post->getLayerLoc()->nf;
   const int nbPost = post->getLayerLoc()->nb;

   const int xPostNeuronsPerPreNeuron = nxPre < nxPost ? nxPost/nxPre : 1;
   assert(nxPre>=nxPost || nxPre*xPostNeuronsPerPreNeuron==nxPost);
   const int xPreNeuronsPerPostNeuron = nxPre > nxPost ? nxPre/nxPost : 1;
   assert(nxPre<=nxPost || nxPost*xPreNeuronsPerPostNeuron==nxPre);
   const int yPostNeuronsPerPreNeuron = nyPre < nyPost ? nyPost/nyPre : 1;
   assert(nyPre>=nyPost || nyPre*yPostNeuronsPerPreNeuron==nyPost);
   const int yPreNeuronsPerPostNeuron = nyPre > nyPost ? nyPre/nyPost : 1;
   assert(nyPre<=nyPost || nyPost*yPreNeuronsPerPostNeuron==nyPre);

   int xPatchHead = (nxp-nxpShrunken)/2;
   assert(2*xPatchHead == nxp-nxpShrunken);
   int yPatchHead = (nyp-nypShrunken)/2;
   assert(2*yPatchHead == nyp-nypShrunken);
   offsetShrunken = kIndex(xPatchHead, yPatchHead, 0, nxp, nyp, nfp);

   for (int kex=0; kex<getNumWeightPatches(); kex++) {
      // calculate xPostStart, xPostStop, xPatchStart, xPatchStop
      int xHalfLength = (nxpShrunken-xPostNeuronsPerPreNeuron)/2;
      assert(2*xHalfLength+xPostNeuronsPerPreNeuron==nxpShrunken);
      int xPre = kxPos(kex, nxPre+2*nbPre, nyPre+2*nbPre, nfPre)-nbPre; // x-coordinate of presynaptic neuron tied to patch kex, in restricted coordinates.
      // xCellStartInPostCoords will be the x-coordinate of the first neuron in the unit cell pre-synaptic site xPre,
      // in postsynaptic restricted coordinates (i.e. leftmost restricted neuron is at x=0; rightmost is at x=post->getLayerLoc()->nx - 1.
      // For a 1-1 connection, this is the same as xPre, but for 1-many or many-1 connections, we have to multiply or divide by "many".
      int xCellStartInPostCoords = xPre;
      if (xPostNeuronsPerPreNeuron>1) {
         xCellStartInPostCoords *= xPostNeuronsPerPreNeuron;
      }
      else if (xPreNeuronsPerPostNeuron>1) {
         // For a many-to-one connection, need to divide by "many", and discard the remainder,
         // but in the left boundary region xPre is negative, so xPre/xPreNeuronsPerPostNeuron is not what we want.
         if (xCellStartInPostCoords>=0) {
            xCellStartInPostCoords /= xPreNeuronsPerPostNeuron;
         }
         else {
            xCellStartInPostCoords = -(-xCellStartInPostCoords-1)/xPreNeuronsPerPostNeuron - 1;
         }
      }
      int xPostStart = xCellStartInPostCoords - xHalfLength;
      int xPostStop = xPostStart + nxpShrunken;
      int xPatchStart = xPatchHead;
      int xPatchStop = xPatchStart + nxpShrunken;

      if (xPostStart < 0) {
         int shrinkamount = -xPostStart;
         xPatchStart += shrinkamount;
         xPostStart = 0;
      }
      if (xPostStart > nxPost) { // This can happen if the pre-layer's boundary region is big and the patch size is small
         int shrinkamount = xPostStart - nxPost;
         xPatchStart -= shrinkamount;
         xPostStart = nxPost;
      }
      if (xPostStop > nxPost) {
         int shrinkamount = xPostStop - nxPost;
         xPatchStop -= shrinkamount;
         xPostStop = nxPost;
      }
      if (xPostStop < 0) {
         int shrinkamount = -xPostStop;
         xPatchStop += shrinkamount;
         xPostStop = 0;
      }
      if (xPatchStart < 0) {
         assert(xPatchStart==xPatchStop);
         xPatchStart = 0;
         xPatchStop = 0;
      }
      if (xPatchStop > (nxp+nxpShrunken)/2) {
         assert(xPatchStart==xPatchStop);
         xPatchStop = (nxp+nxpShrunken)/2;
         xPatchStart = xPatchStop;
      }
      assert(xPostStop-xPostStart==xPatchStop-xPatchStart);

      int nx = xPatchStop - xPatchStart;
      assert(nx>=0 && nx<=nxpShrunken);
      assert(xPatchStart>=0 && (xPatchStart<nxp || (nx==0 && xPatchStart==nxp)));

      // calculate yPostStart, yPostStop, yPatchStart, yPatchStop
      int yHalfLength = (nypShrunken-yPostNeuronsPerPreNeuron)/2;
      assert(2*yHalfLength+yPostNeuronsPerPreNeuron==nypShrunken);
      int yPre = kyPos(kex, nxPre+2*nbPre, nyPre+2*nbPre, nfPre)-nbPre;
      int yCellStartInPostCoords = yPre;
      if (yPostNeuronsPerPreNeuron>1) {
         yCellStartInPostCoords *= yPostNeuronsPerPreNeuron;
      }
      else if (yPreNeuronsPerPostNeuron>1) {
         // For a many-to-one connection, need to divide by "many", and discard the remainder,
         // but in the top boundary region yPre is negative, so yPre/yPreNeuronsPerPostNeuron is not what we want.
         if (yCellStartInPostCoords>=0) {
            yCellStartInPostCoords /= yPreNeuronsPerPostNeuron;
         }
         else {
            yCellStartInPostCoords = -(-yCellStartInPostCoords-1)/yPreNeuronsPerPostNeuron - 1;
         }
      }
      int yPostStart = yCellStartInPostCoords - yHalfLength;
      int yPostStop = yPostStart + nypShrunken;
      int yPatchStart = yPatchHead;
      int yPatchStop = yPatchStart + nypShrunken;

      if (yPostStart < 0) {
         int shrinkamount = -yPostStart;
         yPatchStart += shrinkamount;
         yPostStart = 0;
      }
      if (yPostStart > nyPost) { // This can happen if the pre-layer's boundary region is big and the patch size is small
         int shrinkamount = yPostStart - nyPost;
         yPatchStart -= shrinkamount;
         yPostStart = nyPost;
      }
      if (yPostStop > nyPost) {
         int shrinkamount = yPostStop - nyPost;
         yPatchStop -= shrinkamount;
         yPostStop = nyPost;
      }
      if (yPostStop < 0) {
         int shrinkamount = -yPostStop;
         yPatchStop += shrinkamount;
         yPostStop = 0;
      }
      if (yPatchStart < 0) {
         assert(yPatchStart==yPatchStop);
         yPatchStart = 0;
         yPatchStop = 0;
      }
      if (yPatchStop > (nyp+nypShrunken)/2) {
         assert(yPatchStart==yPatchStop);
         yPatchStop = (nyp+nypShrunken)/2;
         yPatchStart = yPatchStop;
      }
      assert(yPostStop-yPostStart==yPatchStop-yPatchStart);

      int ny = yPatchStop - yPatchStart;
      assert(ny>=0 && ny<=nypShrunken);
      assert(yPatchStart>=0 && (yPatchStart<nxp || (ny==0 && yPatchStart==nyp)));

      gSynPatchStart[arborId][kex] = (size_t) kIndex(xPostStart,yPostStart,0,nxPost,nyPost,nfPost);
      aPostOffset[arborId][kex] = (size_t) kIndex(xPostStart+nbPost,yPostStart+nbPost,0,nxPost+2*nbPost,nyPost+2*nbPost,nfPost);
      PVPatch * w = getWeights(kex, arborId);
      assert(w->offset==0);
      pvpatch_adjust(w, sxp, syp, nx, ny, xPatchStart, yPatchStart);

   } // loop over patches

   return PV_SUCCESS;
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

// fixed?
// TODO - fix this
//   assert(xScale <= 0);
//   assert(yScale <= 0);

   const int prePad = preLoc->nb;

   // pre-synaptic weights are in extended layer reference frame
   const int nxPre = preLoc->nx + 2 * prePad;
   const int nyPre = preLoc->ny + 2 * prePad;
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
      wPostPatches = (PVPatch***) calloc(numAxonalArborLists, sizeof(PVPatch**));
      assert(wPostPatches!=NULL);
      assert(wPostDataStart == NULL);
      wPostDataStart = (pvdata_t **) calloc(numAxonalArborLists, sizeof(pvdata_t *));
      assert(wPostDataStart!=NULL);
      wPostDataStart[0] = allocWeights(numPost, nxpPost, nypPost, nfpPost);
      assert(wPostDataStart[0] != NULL);
      for(int arborID=0;arborID<numberOfAxonalArborLists();arborID++) {
         int status = createWeights(wPostPatches, numPost, numPost, nxpPost, nypPost, nfpPost, arborID);
         assert(status == PV_SUCCESS);
         if (arborID > 0){  // wDataStart already allocated
            wPostDataStart[arborID] = wPostDataStart[0] + spPost * numPost * arborID;
            assert(wPostDataStart[arborID] != NULL);
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
         kxPreHead += prePad;
         kyPreHead += prePad;

         // TODO - FIXME for powXScale > 1
   //      int ax = (int) (1.0f / powXScale);
   //      int ay = (int) (1.0f / powYScale);
   //      int xShift = (ax - 1) - (kxPost + (int) (0.5f * ax)) % ax;
   //      int yShift = (ay - 1) - (kyPost + (int) (0.5f * ay)) % ay;

         pvdata_t * postData = wPostDataStart[arborID] + nxpPost*nypPost*nfpPost*kPost + wPostPatches[arborID][kPost]->offset;
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
               assert(kxPre < 0 || kyPre < 0 || kxPre >= nxPre || kyPre >= nyPre);
               postData[kp] = 0.0; // wPostPatches[arborID][kPost]->data[kp] = 0.0;
            }
            else {
               //int arbor = 0;
               //PVPatch * p = wPatches[arborID][kPre];
               //PVPatch * p = c->getWeights(kPre, arbor);

               //const int nfp = p->nf;

               // get strides for possibly shrunken patch
               //const int sxp = p->sx;
               //const int syp = p->sy;
               //const int sfp = p->sf;

               // *** Old Method (fails test_post_weights) *** //
               // The patch from the pre-synaptic layer could be smaller at borders.
               // At top and left borders, calculate the offset back to the original
               // data pointer for the patch.  This make indexing uniform.
               //
   //            int dx = (kxPre < nxPre / 2) ? nxPrePatch - p->nx : 0;
   //            int dy = (kyPre < nyPre / 2) ? nyPrePatch - p->ny : 0;
   //            int prePatchOffset = - p->sx * dx - p->sy * dy;

   //            int kxPrePatch = (nxPrePatch - 1) - ax * kxPostPatch - xShift;
   //            int kyPrePatch = (nyPrePatch - 1) - ay * kyPostPatch - yShift;
   //            int kPrePatch = kIndex(kxPrePatch, kyPrePatch, kfPost, nxPrePatch, nyPrePatch, p->nf);
   //            wPostPatches[kPost]->data[kp] = p->data[kPrePatch + prePatchOffset];
               // ** //

               // *** New Method *** //
               // {kPre, kzPre} store the extended indices of the presynaptic cell
               // {kPost, kzPost} store the restricted indices of the postsynaptic cell

               // {kzPostHead} store the restricted indices of the postsynaptic patch head
               int kxPostHead, kyPostHead, kfPostHead;
               int nxp_post, nyp_post;  // shrunken patch dimensions
               int dx_nxp, dy_nyp;  // shrinkage

               postSynapticPatchHead(kPre, &kxPostHead, &kyPostHead, &kfPostHead, &dx_nxp,
                                        &dy_nyp,  &nxp_post,   &nyp_post);

               //assert(nxp_post == p->nx);
               //assert(nyp_post == p->ny);
               //assert(nfp == postLoc->nf);

               int kxPrePatch, kyPrePatch; // relative index in shrunken patch
               kxPrePatch = kxPost - kxPostHead;
               kyPrePatch = kyPost - kyPostHead;
               int kPrePatch = kfPost * sfp + kxPrePatch * sxp + kyPrePatch * syp;
               pvdata_t * preData = get_wDataStart(arborID) + nxp*nyp*nfp*kPre + getWeights(kPre,arborID)->offset;
               postData[kp] = preData[kPrePatch];
               // wPostPatches[arborID][kPost]->data[kp] = preData[kPrePatch];

            }
         }
      }
   }
   return wPostPatches;
}

//PVPatch **** HyPerConn::point2PreSynapticWeights2(){
//
//   const PVLayer * lPre  = pre->getCLayer();
//   const PVLayer * lPost = post->getCLayer();
//
//   //xScale is in log format, powScale is post/pre
//   const int xScale = post->getXScale() - pre->getXScale();
//   const int yScale = post->getYScale() - pre->getYScale();
//   const double powXScale = pow(2.0f, (double) xScale);
//   const double powYScale = pow(2.0f, (double) yScale);
//
//// fixed?
//// TODO - fix this
////   assert(xScale <= 0);
////   assert(yScale <= 0);
//
//   const int prePad = lPre->loc.nb;
//
//   // pre-synaptic weights are in extended layer reference frame
//   const int nxPre = lPre->loc.nx + 2 * prePad;
//   const int nyPre = lPre->loc.ny + 2 * prePad;
//   const int nfPre = lPre->loc.nf;
//
//   // post-synaptic weights are in restricted layer
//   const int nxPost  = lPost->loc.nx;
//   const int nyPost  = lPost->loc.ny;
//   const int nfPost  = lPost->loc.nf;
//   const int numPost = lPost->numNeurons;
//
//   nxpPost = (int) (nxp * powXScale);
//   nypPost = (int) (nyp * powYScale);
//   nfpPost = lPre->loc.nf;
//   float z = 0;
//
//   // the number of features is the end-point value (normally post-synaptic)
//   const int numPostPatch = nxpPost * nypPost * nfpPost; // Post-synaptic weights are never shrunken
//
//   if (wPostPatchesp == NULL) {
//
//      //Return data structure
//      wPostPatchesp = (PVPatch****) calloc(numAxonalArborLists, sizeof(PVPatch***));
//      assert(wPostPatchesp!=NULL);
//      assert(wPostDataStartp == NULL);
//      wPostDataStartp = (pvdata_t ***) calloc(numAxonalArborLists, sizeof(pvdata_t **));
//      assert(wPostDataStartp!=NULL);
//
//      for(int arborID=0;arborID<numberOfAxonalArborLists();arborID++) {
//
//         wPostPatchesp[arborID] = (PVPatch***) calloc(numPost, sizeof(PVPatch**));
//
//         int sx = nfpPost;
//         int sy = sx * nxpPost;
//         int sp = sy * nypPost;
//
//         size_t patchSize = sp * sizeof(pvdata_t);
//         size_t dataSize = numPost * patchSize;
//
//         wPostDataStartp[arborID] = (pvdata_t **) calloc(dataSize, sizeof(char*));
//
//
//         PVPatch** patcharray = (PVPatch**) (calloc(numPost, sizeof(PVPatch*)));
//         PVPatch ** curpatch = patcharray;
//         for (int i = 0; i < numPost; i++) {
//            wPostPatchesp[arborID][i] = curpatch;
//            curpatch++;
//         }
//        //createWeights(wPostPatches, numPost, nxpPost, nypPost, nfpPost, arborID);
//      }
//   }
//
//}


PVPatch **** HyPerConn::point2PreSynapticWeights()
{

   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const PVLayerLoc * postLoc = post->getLayerLoc();

   const int xScale = post->getXScale() - pre->getXScale();
   const int yScale = post->getYScale() - pre->getYScale();
   const double powXScale = pow(2.0f, (double) xScale);
   const double powYScale = pow(2.0f, (double) yScale);

// fixed?
// TODO - fix this
//   assert(xScale <= 0);
//   assert(yScale <= 0);

   const int prePad = preLoc->nb;

   // pre-synaptic weights are in extended layer reference frame
   const int nxPre = preLoc->nx + 2 * prePad;
   const int nyPre = preLoc->ny + 2 * prePad;
   const int nfPre = preLoc->nf;

   const int nxPost  = postLoc->nx;
   const int nyPost  = postLoc->ny;
   const int nfPost  = postLoc->nf;
   const int numPost = post->getNumNeurons();

   nxpPost = (int) (nxp * powXScale);
   nypPost = (int) (nyp * powYScale);
   nfpPost = preLoc->nf;
   float z = 0;

   // the number of features is the end-point value (normally post-synaptic)
   const int numPostPatch = nxpPost * nypPost * nfpPost; // Post-synaptic weights are never shrunken

   if (wPostPatchesp == NULL) {

      //Return data structure
      wPostPatchesp = (PVPatch****) calloc(numAxonalArborLists, sizeof(PVPatch***));
      assert(wPostPatchesp!=NULL);
      assert(wPostDataStartp == NULL);
      wPostDataStartp = (pvdata_t ***) calloc(numAxonalArborLists, sizeof(pvdata_t **));
      assert(wPostDataStartp!=NULL);

      for(int arborID=0;arborID<numberOfAxonalArborLists();arborID++) {

         wPostPatchesp[arborID] = (PVPatch***) calloc(numPost, sizeof(PVPatch**));

         int sx = nfpPost;
         int sy = sx * nxpPost;
         int sp = sy * nypPost;

         size_t patchSize = sp * sizeof(pvdata_t);
         size_t dataSize = numPost * patchSize;

         wPostDataStartp[arborID] = (pvdata_t **) calloc(dataSize, sizeof(char*));


         PVPatch** patcharray = (PVPatch**) (calloc(numPost, sizeof(PVPatch*)));
         PVPatch ** curpatch = patcharray;
         for (int i = 0; i < numPost; i++) {
            wPostPatchesp[arborID][i] = curpatch;
            curpatch++;
         }
        //createWeights(wPostPatches, numPost, nxpPost, nypPost, nfpPost, arborID);
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
         kxPreHead += prePad;
         kyPreHead += prePad;

         // TODO - FIXME for powXScale > 1
   //      int ax = (int) (1.0f / powXScale);
   //      int ay = (int) (1.0f / powYScale);
   //      int xShift = (ax - 1) - (kxPost + (int) (0.5f * ax)) % ax;
   //      int yShift = (ay - 1) - (kyPost + (int) (0.5f * ay)) % ay;

         //Accessing by patch offset through wPostDataStart by x,y,and feature of a patch
         pvdata_t ** postData = wPostDataStartp[arborID] + nxpPost*nypPost*nfpPost*kPost + 0;
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
               assert(kxPre < 0 || kyPre < 0 || kxPre >= nxPre || kyPre >= nyPre);
               postData[kp] = &z; // wPostPatches[arborID][kPost]->data[kp] = 0.0;
            }
            else {
               //int arbor = 0;
               //PVPatch * p = wPatches[arborID][kPre];
               //PVPatch * p = c->getWeights(kPre, arbor);

               //const int nfp = p->nf;

               // get strides for possibly shrunken patch
               //const int sxp = p->sx;
               //const int syp = p->sy;
               //const int sfp = p->sf;

               // *** Old Method (fails test_post_weights) *** //
               // The patch from the pre-synaptic layer could be smaller at borders.
               // At top and left borders, calculate the offset back to the original
               // data pointer for the patch.  This make indexing uniform.
               //
   //            int dx = (kxPre < nxPre / 2) ? nxPrePatch - p->nx : 0;
   //            int dy = (kyPre < nyPre / 2) ? nyPrePatch - p->ny : 0;
   //            int prePatchOffset = - p->sx * dx - p->sy * dy;

   //            int kxPrePatch = (nxPrePatch - 1) - ax * kxPostPatch - xShift;
   //            int kyPrePatch = (nyPrePatch - 1) - ay * kyPostPatch - yShift;
   //            int kPrePatch = kIndex(kxPrePatch, kyPrePatch, kfPost, nxPrePatch, nyPrePatch, p->nf);
   //            wPostPatches[kPost]->data[kp] = p->data[kPrePatch + prePatchOffset];
               // ** //

               // *** New Method *** //
               // {kPre, kzPre} store the extended indices of the presynaptic cell
               // {kPost, kzPost} store the restricted indices of the postsynaptic cell

               // {kzPostHead} store the restricted indices of the postsynaptic patch head
               int kxPostHead, kyPostHead, kfPostHead;
               int nxp_post, nyp_post;  // shrunken patch dimensions
               int dx_nxp, dy_nyp;  // shrinkage

               postSynapticPatchHead(kPre, &kxPostHead, &kyPostHead, &kfPostHead, &dx_nxp,
                                        &dy_nyp,  &nxp_post,   &nyp_post);

               //assert(nxp_post == p->nx);
               //assert(nyp_post == p->ny);
               //assert(nfp == lPost->loc.nf);

               int kxPrePatch, kyPrePatch; // relative index in shrunken patch
               kxPrePatch = kxPost - kxPostHead;
               kyPrePatch = kyPost - kyPostHead;
               int kPrePatch = kfPost * sfp + kxPrePatch * sxp + kyPrePatch * syp;
               pvdata_t * preData = get_wDataStart(arborID) + nxp*nyp*nfp*kPre + getWeights(kPre,arborID)->offset;
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

   const int prePad  = preLoc->nb;

   const int nxPre  = preLoc->nx;
   const int nyPre  = preLoc->ny;
   const int kx0Pre = preLoc->kx0;
   const int ky0Pre = preLoc->ky0;
   const int nfPre  = preLoc->nf;

   const int nxexPre = nxPre + 2 * prePad;
   const int nyexPre = nyPre + 2 * prePad;

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
   kxPre += kx0Pre - prePad;
   kyPre += ky0Pre - prePad;

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
      fprintf(stderr, "HyPerConn::postSynapticPatchHead: WARNING patch size is zero\n");
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
   for(int arbor=0; arbor<this->numberOfAxonalArborLists(); arbor++) {
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
   assert(parent->includeConnectionName()<=2 && parent->includeConnectionName()>=0);
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
	  assert(0);
	  break;
   }

   const PVLayerLoc * postLoc = post->getLayerLoc();
   Communicator * comm = parent->icCommunicator();

   bool append = (last) ? false : ioAppend;

	status = PV::writeWeights(path, comm, (double) timef, append, postLoc,
			nxPostPatch, nyPostPatch, nfPostPatch, minVal, maxVal, wPostPatches,
			wPostDataStart, numPostPatches, numberOfAxonalArborLists(),
			writeCompressedWeights, fileType);

   if(status != PV_SUCCESS) {
      fflush(stdout);
      fprintf(stderr, "Connection \"%s\": writePostSynapticWeights failed at time %f.  Exiting.\n", name, timef);
      abort();
   }

   return PV_SUCCESS;
}

int HyPerConn::initNormalize() {
   PVParams * params = parent->parameters();

   normalizeMethod = NULL;
   normalizeMethod = params->stringValue(name, "normalizeMethod");
   if (!normalizeMethod || normalizeMethod[0]=='\0') {
      if (params->present(name, "normalize")) {
         if (parent->columnId()==0) {
            fprintf(stderr, "initNormalize warning for connection \"%s\": normalize_flag is deprecated.\n", name);
            fprintf(stderr, "Please use the string parameter normalizeMethod.\n");
            fprintf(stderr, "'normalize = false;' should be replaced by 'normalizeMethod = \"none\"';\n");
            fprintf(stderr, "and 'normalize = true;' should be replaced by setting normalizeMethod to one of\n");
            fprintf(stderr, "\"normalizeSum\", \"normalizeL2\", \"normalizeMax\", or \"normalizeContrastZeroMean\".\n");
         }
      }
      bool normalize_flag = params->value(name, "normalize", true/*default*/);
      if (normalize_flag) {
         if (params->value(name, "normalize_max", false/*default*/) != 0.0f) {
            normalizeMethod = "normalizeMax";
         }
         bool L2flag = params->value(name, "normalize_RMS_amp", false/*default*/) != 0;
         if (L2flag) {
            normalizeMethod = "normalizeL2";
         }
         else {
            normalizeMethod = "normalizeSum";
         }
      }
      else {
         normalizeMethod = "";
      }
   }
   if (normalizeMethod && normalizeMethod[0]!='\0') {
      if (!strcmp(normalizeMethod, "normalizeSum")) {
         normalizer = new NormalizeSum(name, params);
      }
      else if (!strcmp(normalizeMethod, "normalizeL2"))  {
         normalizer = new NormalizeL2(name, params);
      }
      else if (!strcmp(normalizeMethod, "normalizeMax")) {
         normalizer = new NormalizeMax(name, params);
      }
      else if (!strcmp(normalizeMethod, "normalizeContrastZeroMean")) {
         normalizer = new NormalizeContrastZeroMean(name, params);
      }
      else if (!strcmp(normalizeMethod, "none")) {
         normalizer = NULL;
      }
      else {
         if (parent->columnId()==0) {
            fprintf(stderr, "HyPerConn::initNormalize error: unrecognized normalizeMethod \"%s\".\n", normalizeMethod);
            exit(EXIT_FAILURE);
         }
      }
   }
   else {
      normalizer = NULL;
   }

#ifdef OBSOLETE // Marked obsolete April 11, 2013.  Implementing the NormalizeBase class hierarchy for normalizing
   normalize_flag = params->value(name, "normalize", normalize_flag);
   if( normalize_flag ) {
      normalize_strength = params->value(name, "strength", 1.0f);
      normalizeTotalToPost = params->value(name, "normalizeTotalToPost", /*default*/false);
      if (normalizeTotalToPost) {
         float scale_factor = ((float) postSynapticLayer()->getNumNeurons())/((float) preSynapticLayer()->getNumNeurons());
         normalize_strength *= scale_factor;
      }
      normalize_max = params->value(name, "normalize_max", normalize_max) != 0.0f;
      normalize_zero_offset = params->value(name, "normalize_zero_offset", normalize_zero_offset) != 0.0f;
      normalize_cutoff = params->value(name, "normalize_cutoff", normalize_cutoff) * normalize_strength;
      normalize_RMS_amp = params->value(name, "normalize_RMS_amp", normalize_RMS_amp) != 0.0f;
      if (this->numberOfAxonalArborLists() > 1) {
         normalizeArborsIndividually = params->value(name, "normalize_arbors_individually", normalizeArborsIndividually) != 0.0f;
      }
   }
#endif // OBSOLETE

   return PV_SUCCESS;
}

int HyPerConn::sumWeights(int nx, int ny, int offset, pvdata_t * dataStart, double * sum, double * sum2, pvdata_t * maxVal)
{
   // assert(wp != NULL);
   volatile pvdata_t * w = dataStart + offset;
   // assert(w != NULL);
   // const int nx = wp->nx;
   // const int ny = wp->ny;
   //const int nfp = wp->nf;
   //const int syp = wp->sy;
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

int HyPerConn::scaleWeights(int nx, int ny, int offset, pvdata_t * dataStart, pvdata_t sum, pvdata_t sum2, pvdata_t maxVal)
{
   // assert(wp != NULL);
   int num_weights = nx * ny * nfp; //wp->nf;
   if (!this->normalizeArborsIndividually){
      num_weights *= numberOfAxonalArborLists(); // assumes all arbors shrunken equally at this point (shrink patches should occur after normalize)
   }
   float sigma2 = ( sum2 / num_weights ) - ( sum / num_weights ) * ( sum / num_weights );
   double zero_offset = 0.0f;
   if (normalize_zero_offset){
      // set sum to zero and normalize std of weights to sigma
      zero_offset = sum / num_weights;
      sum = 0.0f;
      maxVal -= zero_offset;
   }
   float scale_factor = 1.0f;
   if (normalize_max) {
      // set maximum weight to normalize_strength
      scale_factor = normalize_strength / ( fabs(maxVal) + (maxVal == 0.0f) );
   }
   else if (sum != 0.0f && !normalize_RMS_amp) {
      scale_factor = normalize_strength / sum;
   }
   else if (!normalize_RMS_amp && (sum == 0.0f) && (sigma2 > 0.0f)) {
      scale_factor = normalize_strength / sqrt(sigma2) ;
   }
   else if (normalize_RMS_amp && (sum2 > 0.0f)) {
      scale_factor = normalize_strength / sqrt(sum2) ;
   }
   else{
	  std::cout << "can't normalize HyPerConn:" << this->name << std::endl;
	  return PV_FAILURE;
   }

   pvdata_t * w = dataStart + offset;
   assert(w != NULL);
   for (int ky = 0; ky < ny; ky++) {
      for(int iWeight = 0; iWeight < syp; iWeight++ ){
         w[iWeight] = ( w[iWeight] - zero_offset ) * scale_factor;
      }
      w += syp;
   }
   maxVal = ( maxVal - zero_offset ) * scale_factor;
   w = dataStart + offset;
   if (fabs(normalize_cutoff) > 0.0f){
      for (int ky = 0; ky < ny; ky++) {
         for(int iWeight = 0; iWeight < syp; iWeight++ ){
            w[iWeight] = ( fabs(w[iWeight]) > fabs(normalize_cutoff) * maxVal) ? w[iWeight] : 0.0f;
         }
         w += syp;
      }
   }
   //maxVal = ( fabs(maxVal) > fabs(normalize_cutoff) ) ? maxVal : 0.0f;
   this->wMax = maxVal > this->wMax ? maxVal : this->wMax;
   return PV_SUCCESS;
} // scaleWeights

// only checks for certain combinations of normalize parameter settings
int HyPerConn::checkNormalizeWeights(float sum, float sum2, float sigma2, float maxVal)
{
   assert( sum != 0 || sigma2 != 0 ); // Calling routine should make sure this condition is met.
   float tol = 0.01f;
   if (normalize_zero_offset && (normalize_cutoff == 0.0f)){  // condition may be violated is normalize_cutoff != 0.0f
      // set sum to zero and normalize std of weights to sigma
      assert((sum > -tol) && (sum < tol));
      assert((sqrt(sigma2) > (1-tol)*normalize_strength) && (sqrt(sigma2) < (1+tol)*normalize_strength));
   }
   else if (normalize_max) {
      // set maximum weight to normalize_strength
      assert((maxVal > (1-tol)*normalize_strength) && ((maxVal < (1+tol)*normalize_strength)));
    }
   else if (normalize_cutoff == 0.0f && !normalize_RMS_amp){  // condition may be violated if normalize_cutoff != 0.0f
      assert((sum > (1-sign(normalize_strength)*tol)*normalize_strength) && ((sum < (1+sign(normalize_strength)*tol)*normalize_strength)));
   }
   else if (normalize_RMS_amp){
	      assert((sqrt(sum2) > (1-tol)*normalize_strength) && (sqrt(sum2) < (1+tol)*normalize_strength));
   }
   return PV_SUCCESS;

} // checkNormalizeWeights

int HyPerConn::checkNormalizeArbor(PVPatch ** patches, pvdata_t ** dataStart, int numPatches, int arborId)
{
   int status = PV_SUCCESS;
   int nx = nxp;
   int ny = nyp;
   int offset = 0;
   if (this->normalizeArborsIndividually) {
      for (int k = 0; k < numPatches; k++) {
         if (patches != NULL) {
            PVPatch * wp = patches[k];
            nx = wp->nx;
            ny = wp->ny;
            offset = wp->offset;
         }
//      PVPatch * wp = patches[k];
//      if( wp->nx < nxp || wp->ny < nyp ) {
//         continue;  // Normalization of shrunken patches used unshrunken part, which is no longer available
//      }
         double sum = 0;
         double sum2 = 0;
         float maxVal = -FLT_MAX;
         status = sumWeights(nx, ny, offset, dataStart[arborId] + k * nxp * nyp * nfp,
               &sum, &sum2, &maxVal);
         int num_weights = nx * ny * nfp; //wp->nf;
         float sigma2 = (sum2 / num_weights) - (sum / num_weights) * (sum / num_weights);
         if (sum != 0 || sigma2 != 0) {
            status = checkNormalizeWeights(sum, sum2, sigma2, maxVal);
            assert( status == PV_SUCCESS);
         }
         else {
            fprintf(stderr,
                  "checkNormalizeArbor: connection \"%s\", arbor %d, patch %d has all zero weights.\n",
                  name, arborId, k);
         }
      }
      return PV_SUCCESS;
   }
   else{
      for (int kPatch = 0; kPatch < numPatches; kPatch++) {
         double sumAll = 0.0f;
         double sum2All = 0.0f;
         float maxAll = 0.0f;
         for(int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++){
            if (patches != NULL) {
               PVPatch * wp = patches[kArbor];
               nx = wp->nx;
               ny = wp->ny;
               offset = wp->offset;
            }
            double sum, sum2;
            float maxVal;
            // PVPatch * p = patches[kPatch];
            status = sumWeights(nx, ny, offset, dataStart[kArbor] + kPatch*nxp*nyp*nfp, &sum, &sum2, &maxVal);
            assert( (status == PV_SUCCESS) || (status == PV_BREAK) );
            sumAll += sum;
            sum2All += sum2;
            maxAll = maxVal > maxAll ? maxVal : maxAll;
         } // kArbor
         int num_weights = nx * ny * nfp * numberOfAxonalArborLists();
         float sigma2 = ( sum2All / num_weights ) - ( sumAll / num_weights ) * ( sumAll / num_weights );
         for(int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++){
            if( sumAll != 0 || sigma2 != 0 ) {
               status = checkNormalizeWeights(sumAll, sum2All, sigma2, maxAll);
               assert(status == PV_SUCCESS );
            }
            else {
					fprintf(stderr,
							"checkNormalizeArbor: connection \"%s\", arbor %d, kernel %d has all zero weights.\n",
							name, kArbor, kPatch);
            }
         }
      }
      return PV_BREAK;
   } // normalizeArborsIndividually
} // checkNormalizeArbor

int HyPerConn::normalizeWeights() {
   int status = PV_SUCCESS;
   if (normalizer) {
      status = normalizer->normalizeWeights(this);
   }
   return status;
}

#ifdef OBSOLETE // Marked obsolete April 11, 2013.  Implementing the new NormalizeBase class hierarchy
int HyPerConn::normalizeWeights(PVPatch ** patches, pvdata_t ** dataStart, int numPatches, int arborId)
{
   if (dataStart == NULL){
      dataStart = this->get_wDataStart();
   }
   int status = PV_SUCCESS;
   this->wMax = -FLT_MAX;
   int nx = nxp;
   int ny = nyp;
   int offset = 0;
   if (this->normalizeArborsIndividually) {
#ifdef USE_SHMGET
        	 if (shmget_flag && !shmget_owner[arborId]) return PV_SUCCESS;
#endif
      for (int k = 0; k < numPatches; k++) {
         if (patches != NULL) {
            PVPatch * wp = patches[k];
            nx = wp->nx;
            ny = wp->ny;
            offset = wp->offset;
         }
         float maxVal = -FLT_MAX;
         double sum = 0;
         double sum2 = 0;
         pvdata_t * dataStartPatch = dataStart[arborId] + k * nxp * nyp * nfp;
         status = sumWeights(nx, ny, offset, dataStartPatch, &sum, &sum2, &maxVal);
         assert( (status == PV_SUCCESS) || (status == PV_BREAK));
         // don't need synchronization barrier here because only this process writes to patches in this arbor
         if (sum2 != 0) {
            status = scaleWeights(nx, ny, offset, dataStartPatch, sum, sum2, maxVal);
            assert( (status == PV_SUCCESS) || (status == PV_BREAK));
         }
      } // k < numPatches
      status = HyPerConn::checkNormalizeArbor(patches, dataStart, numPatches, arborId); // no polymorphism here until HyPerConn generalized to normalize_arbor_individually == false
      assert( (status == PV_SUCCESS) || (status == PV_BREAK));
      return PV_SUCCESS;
   } // normalizeArborsIndividually
   else{
      for (int kPatch = 0; kPatch < numPatches; kPatch++) {
         if (patches != NULL) {
            PVPatch * wp = patches[kPatch];
            nx = wp->nx;
            ny = wp->ny;
            offset = wp->offset;
         }
         double sumAll = 0.0f;
         double sum2All = 0.0f;
         float maxAll = 0.0f;
         for(int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++){
            double sum, sum2;
            float maxVal;
            status = sumWeights(nx, ny, offset, dataStart[kArbor]+kPatch*nxp*nyp*nfp, &sum, &sum2, &maxVal);
            assert( (status == PV_SUCCESS) || (status == PV_BREAK) );
            sumAll += sum;
            sum2All += sum2;
            maxAll = maxVal > maxAll ? maxVal : maxAll;
         } // kArbor
         // insert synchronization barrier to ensure that all processes have finished computing sums over shared memory before any
         // process begins writing to shared memory
#ifdef USE_SHMGET
#ifdef PV_USE_MPI
         //std::cout << "starting MPI_Barrier in HyPerConn::normalizeWeights: " << this->name << ", rank = " << getParent()->icCommunicator()->commRank() << std::endl;
         MPI_Barrier(getParent()->icCommunicator()->communicator());
         //std::cout << "leaving MPI_Barrier in HyPerConn::normalizeWeights: " << this->name << ", rank = " << getParent()->icCommunicator()->commRank() << std::endl;
#endif // PV_USE_MPI
#endif // USE_SHMGET
         for(int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++){
#ifdef USE_SHMGET
        	 if (shmget_flag && !shmget_owner[kArbor]) continue;
#endif
        	if (sum2All != 0) {
               status = scaleWeights(nx, ny, offset, dataStart[kArbor]+kPatch*nxp*nyp*nfp, sumAll, sum2All, maxAll);
               assert( (status == PV_SUCCESS) || (status == PV_BREAK) );
        	}
         }
      } // kPatch < numPatches

      // insert synchronization barrier to ensure that all processes have finished writing to shared memory before checking
      // normalization
#ifdef USE_SHMGET
#ifdef PV_USE_MPI
      //std::cout << "starting MPI_Barrier in HyPerConn::normalizeWeights: " << this->name << ", rank = " << getParent()->icCommunicator()->commRank() << std::endl;
      MPI_Barrier(getParent()->icCommunicator()->communicator());
      //std::cout << "leaving MPI_Barrier in HyPerConn::normalizeWeights: " << this->name << ", rank = " << getParent()->icCommunicator()->commRank() << std::endl;
#endif // PV_USE_MPI
#endif // USE_SHMGET
      status = checkNormalizeArbor(patches, dataStart, numPatches, arborId);
      assert( (status == PV_SUCCESS) || (status == PV_BREAK) );
      return PV_BREAK;
   }
} // normalizeWeights
#endif // OBSOLETE

#ifdef OBSOLETE // Marked obsolete Oct 10, 2013
int HyPerConn::calcPatchSize(int arbor_index, int kex,
                             int * kl_out, int * offset_out,
                             int * nxPatch_out, int * nyPatch_out,
                             int * dx_out, int * dy_out)
{
   int status = PV_SUCCESS;

   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const PVLayerLoc * postLoc = post->getLayerLoc();

   const int prePad  = preLoc->nb;
   const int postPad = postLoc->nb;

   const int nxPre  = preLoc->nx;
   const int nyPre  = preLoc->ny;
   const int kx0Pre = preLoc->kx0;
   const int ky0Pre = preLoc->ky0;
   const int nfPre  = preLoc->nf;

   const int nxexPre = nxPre + 2 * prePad;
   const int nyexPre = nyPre + 2 * prePad;

   const int nxPost  = postLoc->nx;
   const int nyPost  = postLoc->ny;
   const int kx0Post = postLoc->kx0;
   const int ky0Post = postLoc->ky0;
   const int nfPost  = postLoc->nf;

   const int nxexPost = nxPost + 2 * postPad;
   const int nyexPost = nyPost + 2 * postPad;

   // local indices in extended frame
   int kxPre = kxPos(kex, nxexPre, nyexPre, nfPre);
   int kyPre = kyPos(kex, nxexPre, nyexPre, nfPre);

   // convert to global non-extended frame
   kxPre += kx0Pre - prePad;
   kyPre += ky0Pre - prePad;

   // global non-extended post-synaptic frame
   int kxPost = zPatchHead( kxPre, nxp, pre->getXScale(), post->getXScale() );
   int kyPost = zPatchHead( kyPre, nyp, pre->getYScale(), post->getYScale() );

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
         kxPost = nxPost - 1;
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
   if (nxPatch == 0 || nyPatch == 0) {
      dx = 0;
      dy = 0;
      nxPatch = 0;
      nyPatch = 0;
   }

   //If higher than post, set equal to npost
   if (nxPatch > nxPost) {
      nxPatch = nxPost;
   }
   if (nyPatch > nyPost) {
         nyPatch = nyPost;
   }

   // local non-extended index but shifted to be in bounds
   int kl = kIndex(kxPost, kyPost, kfPost, nxPost, nyPost, nfPost);
   assert(kl >= 0);
   assert(kl < post->getNumNeurons());

   // get offset in extended frame
   kxPost += postPad;
   kyPost += postPad;

   int offset = kIndex(kxPost, kyPost, kfPost, nxexPost, nyexPost, nfPost);
   assert(offset >= 0);
   assert(offset < post->getNumExtended());

   // set return variables
   *kl_out = kl;
   *offset_out = offset;
   *nxPatch_out = nxPatch;
   *nyPatch_out = nyPatch;
   *dx_out = dx;
   *dy_out = dy;

   return status;
}
#endif // OBSOLETE

int HyPerConn::patchSizeFromFile(const char * filename) {
   // use patch dimensions from file if (filename != NULL)
   //
   int status = PV_SUCCESS;
   int filetype, datatype;
   double timed = 0.0;

   int numWgtParams = NUM_WGT_PARAMS;

   Communicator * comm = parent->icCommunicator();

   char nametmp[PV_PATH_MAX];
   for (int arborId = 0; arborId < this->numberOfAxonalArborLists(); arborId++){
      snprintf(nametmp, PV_PATH_MAX-1, "%s", filename);

      status = pvp_read_header(nametmp, comm, &timed, &filetype, &datatype, fileparams, &numWgtParams);
      if (status < 0) return status;
      assert(numWgtParams==NUM_WGT_PARAMS);

      // const PVLayerLoc loc = pre->getCLayer()->loc; // checkPVPFileHeader moved to communicate since pre needs to be defined.
      // status = checkPVPFileHeader(comm, &loc, wgtParams, numWgtParams);
      // if (status < 0) return status;

      // reconcile differences with inputParams
      status = checkWeightsHeader(nametmp, fileparams);
   }
   return status;
}

int HyPerConn::checkPatchDimensions() {
   int statusx = checkPatchSize(nxp, pre->getXScale(), post->getXScale(), 'x');
   int statusy = checkPatchSize(nyp, pre->getYScale(), post->getYScale(), 'y');
   int status = statusx==PV_SUCCESS && statusy==PV_SUCCESS ? PV_SUCCESS : PV_FAILURE;
   return status;
}

int HyPerConn::checkPatchSize(int patchSize, int scalePre, int scalePost, char dim) {
   int scaleDiff = scalePre - scalePost;
   bool goodsize;

   if( scaleDiff > 0) {
      // complain if patchSize is not an odd number times 2^xScaleDiff
      int scaleFactor = (int) pow(2, (double) scaleDiff);
      int shouldbeodd = patchSize/scaleFactor;
      goodsize = shouldbeodd > 0 && shouldbeodd % 2 == 1 && patchSize == shouldbeodd*scaleFactor;
   }
   else {
      // complain if patchSize is not an odd number
      goodsize = patchSize > 0 && patchSize % 2 == 1;
   }
   if( !goodsize ) {
      fprintf(stderr, "Error:  Connection: %s\n",name);
      fprintf(stderr, "Presynaptic layer:  %s\n", pre->getName());
      fprintf(stderr, "Postsynaptic layer: %s\n", post->getName());
      fprintf(stderr, "Patch size n%cp=%d is not compatible with presynaptic n%cScale %f\n",
              dim,patchSize,dim,pow(2,-scalePre));
      fprintf(stderr, "and postsynaptic n%cScale %f.\n",dim,pow(2,-scalePost));
      if( scaleDiff > 0) {
         int scaleFactor = (int) pow(2, (float) scaleDiff);
         fprintf(stderr, "(postsynaptic scale) = %d * (presynaptic scale);\n", scaleFactor);
         fprintf(stderr, "therefore compatible sizes are %d times an odd number.\n", scaleFactor);
      }
      else {
         fprintf(stderr, "(presynaptic scale) >= (postsynaptic scale);\n");
         fprintf(stderr, "therefore patch size must be odd\n");
      }
      fprintf(stderr, "Exiting.\n");
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
   postExtStrides.sy = nfp * (post->getLayerLoc()->nx+2*post->getLayerLoc()->nb);

   return PV_SUCCESS;
}

//pvdata_t * HyPerConn::allocWeights(PVPatch *** patches, int nPatches, int nxPatch,
//      int nyPatch, int nfPatch, int arborId)
pvdata_t * HyPerConn::allocWeights(int nPatches, int nxPatch, int nyPatch, int nfPatch)
{
   int sx = nfPatch;
   int sy = sx * nxPatch;
   int sp = sy * nyPatch;

   size_t patchSize = sp * sizeof(pvdata_t);
   size_t dataSize = nPatches * patchSize;
   //if (arborId > 0){  // wDataStart already allocated
	//   assert(this->get_wDataStart(0) != NULL);
	//   return (this->get_wDataStart(0) + sp * nPatches * arborId);
	//}
   // arborID == 0
   size_t arborSize = dataSize * this->numberOfAxonalArborLists();
   pvdata_t * dataPatches = NULL;
   dataPatches = (pvdata_t *) calloc(arborSize, sizeof(char));
   assert(dataPatches != NULL);
   return dataPatches;
}

int HyPerConn::patchToDataLUT(int patchIndex) {
   return patchIndex;
}

int HyPerConn::patchIndexToDataIndex(int patchIndex, int * kx/*default=NULL*/, int * ky/*default=NULL*/, int * kf/*default=NULL*/) {
   const PVLayerLoc * preLoc = pre->getLayerLoc();
   if(kx) *kx = kxPos(patchIndex, preLoc->nx + 2*preLoc->nb, preLoc->ny + 2*preLoc->nb, preLoc->nf);
   if(ky) *ky = kyPos(patchIndex, preLoc->nx + 2*preLoc->nb, preLoc->ny + 2*preLoc->nb, preLoc->nf);
   if(kf) *kf = featureIndex(patchIndex, preLoc->nx + 2*preLoc->nb, preLoc->ny + 2*preLoc->nb, preLoc->nf);
   return patchIndex;
}

int HyPerConn::dataIndexToUnitCellIndex(int dataIndex, int * kx/*default=NULL*/, int * ky/*default=NULL*/, int * kf/*default=NULL*/) {
   return calcUnitCellIndex(dataIndex, kx, ky, kf);
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
   fprintf(stderr, "Out of memory error in %s for connection \"%s\"\n", funcname, name);
   exit(EXIT_FAILURE);
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
