/*
 * HyPerConn.cpp
 *
 *  Created on: Oct 21, 2008
 *      Author: Craig Rasmussen
 */

#include "HyPerConn.hpp"
#include "../layers/LIF.hpp"
#include "../layers/PVLayer.h"
#include "../include/default_params.h"
#include "../io/ConnectionProbe.hpp"
#include "../io/io.h"
#include "../io/fileio.hpp"
#include "../utils/conversions.h"
#include "../utils/pv_random.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "../weightinit/InitWeights.hpp"
#include "../weightinit/InitCocircWeights.hpp"
#include "../weightinit/InitSmartWeights.hpp"
#include "../weightinit/InitUniformRandomWeights.hpp"
#include "../weightinit/InitGaussianRandomWeights.hpp"

namespace PV {

// default values

//PVConnParams defaultConnParams =
//{
//   /*delay*/ 0
//   // Commenting out the same parameters that are commented out in setParams()
//   // , /*fixDelay*/ 0, /*varDelayMin*/ 0, /*varDelayMax*/ 0, /*numDelay*/ 1,
//   // /*isGraded*/ 0, /*vel*/ 45.248, /*rmin*/ 0.0, /*rmax*/ 4.0
//};

HyPerConn::HyPerConn()
{
   initialize_base();
}

HyPerConn::HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, ChannelType channel)
{
   initialize_base();
   initialize(name, hc, pre, post, channel, NULL, NULL);
}

HyPerConn::HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, ChannelType channel, InitWeights *weightInit)
{
   initialize_base();
   initialize(name, hc, pre, post, channel, NULL, weightInit);
}

// provide filename or set to NULL
HyPerConn::HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, ChannelType channel, const char * filename)
{
   initialize_base();
   initialize(name, hc, pre, post, channel, filename, NULL);
}

HyPerConn::HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, ChannelType channel, const char * filename, InitWeights *weightInit)
{
   initialize_base();
   initialize(name, hc, pre, post, channel, filename, weightInit);
}


HyPerConn::~HyPerConn()
{
   if (parent->columnId() == 0) {
      printf("%32s: total time in %6s %10s: ", name, "conn", "update ");
      update_timer->elapsed_time();
      fflush(stdout);
   }
   delete update_timer;

   free(name);

#ifdef PV_USE_OPENCL
   delete krRecvSyn;

   if (clWeights != NULL) {
      for (int arbor = 0; arbor < numAxonalArborLists; arbor++) {
         delete clWeights[arbor];
      }
      free(clWeights);
      clWeights = NULL;
   }
#endif // PV_USE_OPENCL

   // assert(params != NULL); // freeing a null pointer is not an error
   //free(params);

   deleteWeights();

   // free the task information

//   for (int l = 0; l < numAxonalArborLists; l++) {
//      if ( axonalArborList[l] ) {
//         free(axonalArbor(0, l)->data);
//         // axonalArbor(0,l) frees all data patches for arbor l because all
//         // axonalArbor patches for that l were created in a single calloc().
//         free(axonalArborList[l]);
//      }
//   }
   free(*gSynPatchStart); // All gSynPatchStart[k]'s were allocated together in a single malloc call.
   free(gSynPatchStart);
   free(*aPostOffset); // All aPostOffset[k]'s were allocated together in a single malloc call.
   free(aPostOffset);

   free(delays);

   // delete weightInitializer; // weightInitializer should be deleted by whoever called the HyPerConn constructor

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
   this->nfp = 1;
   this->sxp = 1;
   this->syp = 1;
   this->sfp = 1;
   this->parent = NULL;
   this->connId = 0;
   this->pre = NULL;
   this->post = NULL;
   this->numAxonalArborLists = 1;
   this->channel = CHANNEL_EXC;
   this->ioAppend = false;

   this->weightInitializer = NULL;

   this->probes = NULL;
   this->numProbes = 0;

   this->update_timer = new Timer();

   this->wMin = 0.0;
   this->wMax = 1.0;
   this->wPostTime = -1.0;
   this->wPostPatches = NULL;
   this->writeCompressedWeights = true;
   this->fileType = PVP_WGT_FILE_TYPE; // Subclass's initialize_base() gets called after HyPerConn's initialize_base(), so this can be changed in subclasses.

   wPatches=NULL;
   // axonalArborList=NULL;
   dwPatches = NULL;
   aPostOffset = NULL;

   this->selfFlag = false;  // determines whether connections are made to neuron at same location (i.e. could be a self-connection)

   this->normalize_flag = true; // default value, overridden by params file parameter "normalize" in initNormalize()
   this->plasticityFlag = false;
   this->shrinkPatches_flag = false; // default value, overridden by params file parameter "normalize" in initNormalize()
   this->normalize_arbors_individually = true;
   this->normalize_max = false;
   this->normalize_max = false;
   this->normalize_zero_offset = false;
   this->normalize_cutoff = 0.0f;

   return PV_SUCCESS;
}

int HyPerConn::createArbors() {
   wPatches = (PVPatch***) calloc(numAxonalArborLists, sizeof(PVPatch**));
   if( wPatches == NULL ) {
      createArborsOutOfMemory();
      assert(false);
   }
   gSynPatchStart = (pvdata_t ***) calloc( numAxonalArborLists, sizeof(pvdata_t **) );
   if( gSynPatchStart == NULL ) {
      createArborsOutOfMemory();
      assert(false);
   }
   pvdata_t ** gSynPatchStartBuffer = (pvdata_t **) calloc(
         (this->shrinkPatches_flag ? numAxonalArborLists : 1)
               * preSynapticLayer()->getNumExtended(), sizeof(pvdata_t *));
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
   size_t * aPostOffsetBuffer = (size_t *) calloc(
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
   delays = (int *) calloc(numAxonalArborLists, sizeof(int));
   if( delays == NULL ) {
      createArborsOutOfMemory();
      assert(false);
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
   int status = PV_SUCCESS;

   initShrinkPatches(); // Sets shrinkPatches; derived-class methods that override initShrinkPatches must also set shrinkPatches
   // createArbors() uses the value of shrinkPatches.
   //allocate the arbor arrays:
   createArbors();


   //const int arbor = 0;
   //numAxonalArborLists = 1;

   setPatchSize(filename);
   setPatchStrides();

   //allocate weightpatches and axonal arbors for each arbor
   for (int arborId=0;arborId<numAxonalArborLists;arborId++) {
      wDataStart[arborId] = createWeights(wPatches, arborId);
      assert(wPatches[arborId] != NULL);
      setWPatches(wPatches[arborId], arborId);
      //wPatches[arborId] = createWeights(wPatches[arborId]); //allocates memory for weights

      // Create list of axonal arbors containing pointers to {phi,w,P,M} patches.
      //  weight patches may shrink
      status |= createAxonalArbors(arborId);
      assert(weights(arborId) != NULL);
   }  // arborId

   //initialize weights for patches:
   assert( initializeWeights(wPatches, numWeightPatches(), filename) != NULL);
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
   int numPatches = numWeightPatches();
   for (int kex = 0; kex < numPatches; kex++) {
      shrinkPatch(kex, arborId /* arbor */ );
   } // loop over pre-synaptic neurons

   return 0;
}

int HyPerConn::shrinkPatch(int kExt, int arborId /* PVAxonalArbor * arbor */) {
   //int kl, offset, nxPatch, nyPatch, dx, dy;
   //calcPatchSize(arborId, kex, &kl, &offset, &nxPatch, &nyPatch, &dx, &dy);

   // PVAxonalArbor * arbor = axonalArbor(kExt, arborId);

   PVPatch *weights = getWeights(kExt,arborId);
   pvdata_t * w = weights->data;

   int nxp = weights->nx;
   int nyp = weights->ny;
   //int nfp = weights->nf;

   //int sxp = weights->sx;
   //int syp = weights->sy;
   //int sfp = weights->sf;

   int maxnx = -999999;
   int minnx = 999999;
   int maxny = -999999;
   int minny = 999999;

   bool nonZeroWeightFound = false;
   // loop over all post-synaptic cells in patch
   for (int y = 0; y < nyp; y++) {
      for (int x = 0; x < nxp; x++) {
         for (int f = 0; f < nfp; f++) {
            if(w[x * sxp + y * syp + f * sfp] != 0) {
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
      int nxNew = maxnx - minnx;
      int nyNew = maxny - minny;
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


int HyPerConn::initShrinkPatches() {
   PVParams * params = parent->parameters();
   shrinkPatches_flag = params->value(name, "shrinkPatches", shrinkPatches_flag);
   return PV_SUCCESS;
}

#ifdef OBSOLETE // marked Obsolete Oct 1, 2011.  We can use a default argument for weightInit instead of defining an extra form of the method
int HyPerConn::initialize(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, ChannelType channel, const char * filename) {
   return initialize(name, hc, pre, post, channel, filename, NULL);
}
#endif // OBSOLETE

int HyPerConn::initialize(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, ChannelType channel, const char * filename, InitWeights *weightInit)
{
   int status = PV_SUCCESS;

   int postnumchannels = post->getNumChannels();
   if(postnumchannels <= 0) {
      fprintf(stderr, "Connection \"%s\": layer \"%s\" has no channels and cannot be a post-synaptic layer.  Exiting.\n",
              name, post->getName());
      exit(EXIT_FAILURE);
   }
   if( channel < 0 || channel >= postnumchannels ) {
      fprintf(stderr, "Connection \"%s\": given channel is %d but channels for post-synaptic layer \"%s\" are 0 through %d. Exiting.\n",
              name, channel, post->getName(), post->getNumChannels()-1);
      exit(EXIT_FAILURE);
   }
   this->parent = hc;
   this->pre = pre;
   this->post = post;
   this->channel = channel;

   free(this->name);  // name will already have been set in initialize_base()
   this->name = strdup(name);
   assert(this->name != NULL);

   PVParams * inputParams = parent->parameters();
   //if a weight initializer hasn't been created already, use the default--> either 2D Gauss or read from file
   if(weightInit==NULL) {
      this->weightInitializer = handleMissingInitWeights(inputParams);
   }
   else {
      this->weightInitializer = weightInit;
   }
   assert(this->weightInitializer != NULL);

   status = setParams(inputParams /*, &defaultConnParams*/);
   defaultDelay = (int) inputParams->value(name, "delay", 0);

//   stochasticReleaseFlag = inputParams->value(name, "stochasticReleaseFlag", 0, true) != 0;
   accumulateFunctionPointer = stochasticReleaseFlag ? &pvpatch_accumulate_stochastic : &pvpatch_accumulate;

   this->connId = parent->addConnection(this);

   writeTime = parent->simulationTime();
   writeStep = parent->parameters()->value(name, "writeStep", parent->getDeltaTime());

   constructWeights(filename);

   // Find maximum delay over all the arbors and send it to the presynaptic layer
   int maxdelay = 0;
   for( int arborId=0; arborId<numberOfAxonalArborLists(); arborId++ ) {
      int curdelay = this->getDelay(arborId);
      if( maxdelay < curdelay ) maxdelay = curdelay;
   }
   int allowedDelay = pre->increaseDelayLevels(maxdelay);
   if( allowedDelay < maxdelay ) {
      if( parent->icCommunicator()->commRank() == 0 ) {
         fflush(stdout);
         fprintf(stderr, "Connection \"%s\": attempt to set delay to %d, but the maximum allowed delay is %d.  Exiting\n", name, maxdelay, allowedDelay);
      }
      exit(EXIT_FAILURE);
   }

#ifdef PV_USE_OPENCL
   initializeThreadBuffers("HyPerLayer_recv_synaptic_input");
   initializeThreadKernels("HyPerLayer_recv_synaptic_input");
#endif // PV_USE_OPENCL

   return status;
}

int HyPerConn::initPlasticityPatches()
{
   if (!plasticityFlag) return PV_SUCCESS;

   const int numAxons = numberOfAxonalArborLists();

   dwPatches = (PVPatch***) calloc(numAxons, sizeof(PVPatch**));
   assert(dwPatches != NULL);

   int numArbors = numWeightPatches();
   for (int arborId = 0; arborId < numAxons; arborId++) {

      this->set_dwDataStart(arborId, createWeights(dwPatches, numWeightPatches(), nxp, nyp, nfp, arborId));
      assert(dwPatches[arborId] != NULL);
      // PVPatch** dWPatch = createWeights(NULL, numWeightPatches(), nxp, nyp, nfp, 0);
      // assert(dWPatch != NULL);


      // kex is in extended frame
      for (int kex = 0; kex < numArbors; kex++) {
         int kl, offset, nxPatch, nyPatch, dx, dy;

         calcPatchSize(arborId, kex, &kl, &offset, &nxPatch, &nyPatch, &dx, &dy);

         // adjust patch size (shrink) to fit within interior of post-synaptic layer
         //
         //arbor->plasticIncr = dwPatches[n][kex];
         //arbor->plasticIncr = dwPatches[arborId][kex];
         pvpatch_adjust(dwPatches[arborId][kex], sxp, syp, nxPatch, nyPatch, dx, dy);

      } // loop over pre-synaptic neurons
      setdWPatches(dwPatches[arborId], arborId);

   } // loop over arbors

   return PV_SUCCESS;
}

// set member variables specified by user
int HyPerConn::setParams(PVParams * inputParams /*, PVConnParams * p*/)
{
   const char * name = getName();

   numAxonalArborLists=(int) inputParams->value(name, "numAxonalArbors", 1, true);
   plasticityFlag = inputParams->value(name, "plasticityFlag", plasticityFlag, true);
   stochasticReleaseFlag = inputParams->value(name, "stochasticReleaseFlag", false, true) != 0;

   writeCompressedWeights = inputParams->value(name, "writeCompressedWeights", true);

   selfFlag = inputParams->value(name, "selfFlag", selfFlag, true);

   return 0;
}

// returns handle to initialized weight patches
PVPatch *** HyPerConn::initializeWeights(PVPatch *** arbors, int numPatches, const char * filename)
{
   weightInitializer->initializeWeights(arbors, numPatches, filename, this);
   // for(int arborId=0; arborId<numberOfAxonalArborLists(); arborId++) {
   //    weightInitializer->initializeWeights(arbors[arborId], arborId, numPatches, filename, this);
   // }
   initNormalize(); // Sets normalize_flag; derived-class methods that override initNormalize must also set normalize_flag
   if (normalize_flag) {
      for(int arborId=0; arborId<numberOfAxonalArborLists(); arborId++) {
         int status = normalizeWeights(arbors[arborId], numPatches, arborId);
         if (status == PV_BREAK) break;
      } // arborId
   } // normalize_flag
   return arbors;
}

InitWeights * HyPerConn::handleMissingInitWeights(PVParams * params) {
   int rank = parent->icCommunicator()->commRank();
   bool randomFlag = params->value(name, "randomFlag", 0.0f, false) != 0;
   bool smartWeights = params->value(name, "smartWeights",0.0f, false) != 0;
   bool cocircWeights = params->value(name, "cocircWeights",0.0f, false) != 0;

   if( rank == 0 ) {
      bool using_legacy_flags = randomFlag || smartWeights || cocircWeights;
      if( using_legacy_flags ) {
         fprintf(stderr, "Connection \"%s\": This method of initializing weights has been deprecated.\n"
                         "  Please pass an InitWeights object to the constructor.\n"
                         "  In buildandrun(), use the string parameter \"weightInitType\" to set the InitWeights object.\n", name);
      }
      else {
         fprintf(stderr, "Connection \"%s\": Please pass an InitWeights object to the constructor.\n"
                         "  In buildandrun(), use the string parameter \"weightInitType\" to set the InitWeights object.\n", name);

      }
   }

   if( randomFlag ) {
      if( rank==0 && (smartWeights || cocircWeights) ) {
         fprintf(stderr, "Connection \"%s\": Conflict in specifying initial weights.  randomFlag will be used\n", name);
      }
      bool uniform_weights = params->value(name, "uniformWeights", 1.0f, false) != 0;
      bool gaussian_weights = params->value(name, "gaussianWeights", 0.0f, false) != 0;
      if( uniform_weights ) {
         if( gaussian_weights ) {
            if( rank==0 ) {
               fprintf(stderr, "Connection \"%s\": Conflict in specifying distribution for random weights.  uniformWeights will be used\n", name);
            }
         }
         return new InitUniformRandomWeights();
      }
      else if( gaussian_weights ) {
         return new InitGaussianRandomWeights();
      }
      else {
         if( rank==0 ) {
            fprintf(stderr, "Connection \"%s\": No distribution for random weights specified.  gaussianWeights will be used.\n", name);
         }
         return new InitGaussianRandomWeights();
      }
   }
   else if (smartWeights) {
      if( rank==0 && (cocircWeights) ) {
         fprintf(stderr, "Connection \"%s\": Conflict in specifying initial weights.  smartWeights will be used\n", name);
      }
      return new InitSmartWeights();
   }
   else if (cocircWeights) {
      return new InitCocircWeights();
   }
   else {
      return new InitWeights();
   }
   return NULL;
}

#ifdef PV_USE_OPENCL
/**
 * Initialize OpenCL buffers.  This must be called after weights have
 * been allocated.
 */
int HyPerConn::initializeThreadBuffers(const char * kernel_name)
{
   int status = CL_SUCCESS;

   const size_t size = numWeightPatches() * nxp*nyp*nfp * sizeof(pvdata_t);

   CLDevice * device = parent->getCLDevice();

   clWeights = NULL;
   if (numAxonalArborLists > 0) {
      clWeights = (CLBuffer **) malloc(numAxonalArborLists*sizeof(CLBuffer *));
      assert(clWeights != NULL);
   }

   // create device buffers for weights
   //
   for (int arbor = 0; arbor < numAxonalArborLists; arbor++) {
      pvdata_t * wBuf = getWeights(0, arbor)->data;
      clWeights[arbor] = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, wBuf);
   }

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

   float xScale = (float)postLoc->nx/(float)preLoc->nx;
   float yScale = (float)postLoc->ny/(float)preLoc->ny;
   status |= krRecvSyn->setKernelArg(argid++, xScale);
   status |= krRecvSyn->setKernelArg(argid++, yScale);

   clArgIdOffset = argid;  // offset into activity buffer (with delay)
   status |= krRecvSyn->setKernelArg(argid++, pre->getLayerDataStoreOffset());
   // activity buffer from DataStore
   status |= krRecvSyn->setKernelArg(argid++, pre->getLayerDataStoreCLBuffer());
   clArgIdWeights = argid; // weights
   status |= krRecvSyn->setKernelArg(argid++, clWeights[0]);
   // update variable, GSyn
   status |= krRecvSyn->setKernelArg(argid++, post->getChannelCLBuffer(getChannel()));

   return status;
}
#endif // PV_USE_OPENCL

int HyPerConn::checkPVPFileHeader(Communicator * comm, const PVLayerLoc * loc, int params[], int numParams)
{
   // use default header checker
   //
   return pvp_check_file_header(comm, loc, params, numParams);
}

int HyPerConn::checkWeightsHeader(const char * filename, int * wgtParams)
{
   // extra weight parameters
   //
   const int nxpFile = wgtParams[NUM_BIN_PARAMS + INDEX_WGT_NXP];
   const int nypFile = wgtParams[NUM_BIN_PARAMS + INDEX_WGT_NYP];
   const int nfpFile = wgtParams[NUM_BIN_PARAMS + INDEX_WGT_NFP];

   if (nxp != nxpFile) {
      fprintf(stderr,
              "ignoring nxp = %i in HyPerConn %s, using nxp = %i in binary file %s\n",
              nxp, name, nxpFile, filename);
      nxp = nxpFile;
   }
   if (nyp != nypFile) {
      fprintf(stderr,
              "ignoring nyp = %i in HyPerConn %s, using nyp = %i in binary file %s\n",
              nyp, name, nypFile, filename);
      nyp = nypFile;
   }
   if (nfp != nfpFile) {
      fprintf(stderr,
              "ignoring nfp = %i in HyPerConn %s, using nfp = %i in binary file %s\n",
              nfp, name, nfpFile, filename);
      nfp = nfpFile;
   }
   return 0;
}
int HyPerConn::correctPIndex(int patchIndex) {
   return patchIndex;
}

int HyPerConn::writeWeights(float time, bool last)
{
   const int numPatches = numWeightPatches();
   return writeWeights(wPatches, numPatches, NULL, time, last);
}

int HyPerConn::writeWeights(const char * filename) {
   return writeWeights(wPatches, numWeightPatches(), filename, parent->simulationTime(), true);
}

#ifdef OBSOLETE_NBANDSFORARBORS
int HyPerConn::writeWeights(float time, bool last)
{
   //const int arbor = 0;
   const int numPatches = numWeightPatches();
   for(int arborId=0;arborId<numberOfAxonalArborLists();arborId++) {
      if(writeWeights(wPatches[arborId], numPatches, NULL, time, last, arborId))
         return 1;
   }
   return 0;
}
#endif OBSOLETE_NBANDSFORARBORS

int HyPerConn::writeWeights(PVPatch *** patches, int numPatches, const char * filename, float timef, bool last) {
   int status = PV_SUCCESS;
   char path[PV_PATH_MAX];

   if (patches == NULL) return PV_SUCCESS;

   float minVal = FLT_MAX;
   float maxVal = -FLT_MAX;
   for(int arbor=0; arbor<this->numberOfAxonalArborLists(); arbor++) {
      float minVal1 = minWeight(arbor);
      if( minVal1 < minVal ) minVal = minVal1;
      float maxVal1 = maxWeight(arbor);
      if( maxVal1 > maxVal ) maxVal = maxVal1;
   }

   const PVLayerLoc * loc = pre->getLayerLoc();

   if (filename == NULL) {
      if (last) {
         snprintf(path, PV_PATH_MAX-1, "%s/w%d_last.pvp", parent->getOutputPath(), getConnectionId());
      }
      else {
         snprintf(path, PV_PATH_MAX - 1, "%s/w%d.pvp", parent->getOutputPath(), getConnectionId());
      }
   }
   else {
      snprintf(path, PV_PATH_MAX-1, "%s", filename);
   }

   Communicator * comm = parent->icCommunicator();

   bool append = last ? false : ioAppend;

   status = PV::writeWeights(path, comm, (double) timef, append,
                             loc, nxp, nyp, nfp, minVal, maxVal,
                             patches, numPatches, numberOfAxonalArborLists(), writeCompressedWeights, fileType);
   assert(status == 0);

   return status;
}

#ifdef OBSOLETE_NBANDSFORARBORS
int HyPerConn::writeWeights(PVPatch ** patches, int numPatches,
                            const char * filename, float time, bool last, int arborId)
{
   int status = 0;
   char path[PV_PATH_MAX];

   if (patches == NULL) return 0;

   const float minVal = minWeight(arborId);
   const float maxVal = maxWeight(arborId);

   const PVLayerLoc * loc = pre->getLayerLoc();

   if (filename == NULL) {
      if (last) {
         if(numberOfAxonalArborLists()>1)
            snprintf(path, PV_PATH_MAX-1, "%s/w%d_a%d_last.pvp", parent->getOutputPath(), getConnectionId(), arborId);
         else
            snprintf(path, PV_PATH_MAX-1, "%s/w%d_last.pvp", parent->getOutputPath(), getConnectionId());
      }
      else {
         if(numberOfAxonalArborLists()>1)
            snprintf(path, PV_PATH_MAX - 1, "%s/w%d_a%d.pvp", parent->getOutputPath(), getConnectionId(), arborId);
         else
            snprintf(path, PV_PATH_MAX - 1, "%s/w%d.pvp", parent->getOutputPath(), getConnectionId());
      }
   }
   else {
      snprintf(path, PV_PATH_MAX-1, "%s", filename);
   }

   Communicator * comm = parent->icCommunicator();

   bool append = (last) ? false : ioAppend;

   status = PV::writeWeights(path, comm, (double) time, append,
                             loc, nxp, nyp, nfp, minVal, maxVal,
                             patches, numPatches, writeCompressedWeights, fileType);
   assert(status == 0);

#ifdef DEBUG_WEIGHTS
   char outfile[PV_PATH_MAX];

   // only write first weight patch

   sprintf(outfile, "%s/w%d.tif", parent->getOutputPath(), getConnectionId());
   FILE * fd = fopen(outfile, "wb");
   if (fd == NULL) {
      fprintf(stderr, "writeWeights: ERROR opening file %s\n", outfile);
      return 1;
   }
   int arbor = 0;
   pv_tiff_write_patch(fd, patches);
   fclose(fd);
#endif // DEBUG_WEIGHTS

   return status;
}
#endif // OBSOLETE_NBANDSFORARBORS

int HyPerConn::writeTextWeights(const char * filename, int k)
{
   FILE * fd = stdout;
   char outfile[PV_PATH_MAX];

   if (filename != NULL) {
      snprintf(outfile, PV_PATH_MAX-1, "%s/%s", parent->getOutputPath(), filename);
      fd = fopen(outfile, "w");
      if (fd == NULL) {
         fprintf(stderr, "writeWeights: ERROR opening file %s\n", filename);
         return 1;
      }
   }

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
      writeTextWeightsExtra(fd, k, arbor);
      pv_text_write_patch(fd, wPatches[arbor][k], nfp, sxp, syp, sfp);
      fprintf(fd, "----------------------------\n");
   }

   if (fd != stdout) {
      fclose(fd);
   }

   return 0;
}

void HyPerConn::setDelay(int arborId, int delay) {
   assert(arborId>=0 && arborId<numAxonalArborLists);
   delays[arborId] = delay;
//   int numPatches = numWeightPatches();
//    for(int pID=0;pID<numPatches; pID++) {
//       axonalArbor(pID, arborId)->delay = delay;
//    }
}

// NOTE: this should be temporary until delivery interface is straightened out
//
#ifdef PV_USE_OPENCL
int HyPerConn::deliverOpenCL(Publisher * pub)
{
   int status = PV_SUCCESS;

   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const size_t nxex = preLoc->nx + 2*preLoc->nb;
   const size_t nyex = preLoc->ny + 2*preLoc->nb;
   const size_t nxl = 16;
   const size_t nyl = 8;

   // for all numextended in pre

   for (int arbor = 0; arbor < numberOfAxonalArborLists(); arbor++) {
      size_t activityOffset = pre->getLayerDataStoreOffset(arbor);
      status |= krRecvSyn->setKernelArg(clArgIdOffset, activityOffset);
      status |= krRecvSyn->setKernelArg(clArgIdWeights, clWeights[arbor]);
      status |= krRecvSyn->run(nxex, nyex, nxl, nyl, 0, NULL, &evRecvSyn);
   }

   // TODO - use events properly
   status |= clWaitForEvents(1, &evRecvSyn);
   clReleaseEvent(evRecvSyn);

   return status;
}
#endif // PV_USE_OPENCL

int HyPerConn::deliver(Publisher * pub, const PVLayerCube * cube, int neighbor)
{
#ifdef DEBUG_OUTPUT
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   printf("[%d]: HyPerConn::deliver: neighbor=%d cube=%p post=%p this=%p\n", rank, neighbor, cube, post, this);
   fflush(stdout);
#endif // DEBUG_OUTPUT

   for(int arborId=0;arborId<numberOfAxonalArborLists();arborId++) {
      int delay = getDelay(arborId);
      pub->readData(delay);
      int status = post->recvSynapticInput(this, cube, arborId);
      if (status == PV_BREAK) break;
      assert(status == PV_SUCCESS);
   }

#ifdef DEBUG_OUTPUT
   printf("[%d]: HyPerConn::delivered: \n", rank);
   fflush(stdout);
#endif // DEBUG_OUTPUT
   return 0;
}

int HyPerConn::checkpointRead(float * timef) {
   char * filename = checkpointFilename();
   InitWeights * weightsInitObject = new InitWeights();
   weightsInitObject->initializeWeights(wPatches, numDataPatches(), filename, this, timef);
   free(filename);
   return PV_SUCCESS;
}

int HyPerConn::checkpointWrite() {
   char * filename = checkpointFilename();
   int status = writeWeights(wPatches, numWeightPatches(), filename, parent->simulationTime(), true);
   free(filename);
   return status;
}

char * HyPerConn::checkpointFilename() {
   char * filename = (char *) malloc( (strlen(name)+12)*sizeof(char) );
   // routine that calls checkpointFilename should free filename when done
   if( filename != NULL ) {
      sprintf(filename, "%s_W.pvp", name);
   }
   else {
         fprintf(stderr, "Connection \"%s\", Rank %d process: unable to allocate memory for checkpointFilename.  Exiting.\n", name, parent->icCommunicator()->commRank());
         abort();
   }
   return filename;
}

int HyPerConn::insertProbe(BaseConnectionProbe * p)
{
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

int HyPerConn::outputState(float time, bool last)
{
   int status = 0;

   if( !last ) {
      for (int i = 0; i < numProbes; i++) {
         probes[i]->outputState(time, this);
      }
   }

   if (last) {
      status = writeWeights(time, last);
      assert(status == 0);
   }
   else if ( (time >= writeTime) && (writeStep >= 0) ) {
      writeTime += writeStep;

      status = writeWeights(time, last);
      assert(status == 0);

      // append to output file after original open
      ioAppend = true;
   }

   return status;
}

int HyPerConn::updateState(float time, float dt)
{
   update_timer->start();

   int status;
   //const int axonId = 0;       // assume only one for now
   for(int axonId=0;axonId<numberOfAxonalArborLists();axonId++) {
      status = calc_dW(axonId);        // Calculate changes in weights
      // TODO error handling
      status = updateWeights(axonId);  // Apply changes in weights
   }
   update_timer->stop();
   return status;
}

int HyPerConn::calc_dW(int axonId) {
   return PV_SUCCESS;
}

//
/* M (m or pDecr->data) is an extended post-layer variable
 *
 */
int HyPerConn::updateWeights(int axonId)
{
   return 0;
}

int HyPerConn::numDataPatches()
{
   return numWeightPatches();
}

/**
 * returns the number of weight patches for the given neighbor
 * @param neighbor the id of the neighbor (0 for interior/self)
 */
int HyPerConn::numWeightPatches()
{
   // for now there is just one axonal arbor
   // extending to all neurons in extended layer
   return pre->getNumExtended();
}

PVPatch * HyPerConn::getWeights(int k, int arbor)
{
   // a separate arbor/patch of weights for every neuron
   return wPatches[arbor][k];
}

pvdata_t * HyPerConn::createWeights(PVPatch *** patches, int nPatches, int nxPatch,
      int nyPatch, int nfPatch, int axonId)
{
   // could create only a single patch with following call
   //   return createPatches(numAxonalArborLists, nxp, nyp, nfp);

   //assert(numAxonalArborLists == 1);
   assert(patches[axonId] == NULL);

   //patches = (PVPatch**) calloc(sizeof(PVPatch*), nPatches);
   patches[axonId] = (PVPatch**) calloc(nPatches, sizeof(PVPatch*));
   assert(patches[axonId] != NULL);

   // TODO - allocate space for all weights at once (inplace), return pointer to beginning of weight array
   pvdata_t * data_patches = allocWeights(patches, nPatches, nxPatch, nyPatch, nfPatch, axonId);
   return data_patches;
}

/**
 * Create a separate patch of weights for every neuron
 */
pvdata_t * HyPerConn::createWeights(PVPatch *** patches, int axonId)
{
   //const int arbor = 0;
   int nPatches = numWeightPatches();
   int nxPatch = nxp;
   int nyPatch = nyp;
   int nfPatch = nfp;

   pvdata_t * data_patches = createWeights(patches, nPatches, nxPatch, nyPatch, nfPatch, axonId);
   return data_patches;
}

int HyPerConn::deleteWeights()
{
   // to be used if createPatches is used above
   // HyPerConn::deletePatches(numAxonalArborLists, wPatches);

   for (int arbor = 0; arbor < numAxonalArborLists; arbor++) {
      int numPatches = numWeightPatches();
      if (wPatches != NULL) {
         if (wPatches[arbor] != NULL) {
            for (int k = 0; k < numPatches; k++) {
               pvpatch_inplace_delete(wPatches[arbor][k]);
            }
            free(wPatches[arbor]);
            wPatches[arbor] = NULL;
         }
      }
      if (wDataStart != NULL) {
         free(this->wDataStart[arbor]);
         this->wDataStart[arbor] = NULL;
      }
      if (dwPatches != NULL) {
         if (dwPatches[arbor] != NULL) {
            for (int k = 0; k < numPatches; k++) {
               pvpatch_inplace_delete(dwPatches[arbor][k]);
            }
            free(dwPatches[arbor]);
            dwPatches[arbor] = NULL;
         }
      }
      if (dwDataStart != NULL){
         free(this->dwDataStart[arbor]);
         this->dwDataStart[arbor] = NULL;
      }
   }
   free(wPatches);
   wPatches = NULL;
   free(wDataStart);
   wDataStart = NULL;
   free(dwPatches);
   dwPatches = NULL;
   free(dwDataStart);
   dwDataStart = NULL;

   if (wPostPatches != NULL) {
      for(int axonID=0;axonID<numberOfAxonalArborLists();axonID++) {
         const int numPostNeurons = post->getNumNeurons();
         for (int k = 0; k < numPostNeurons; k++) {
            pvpatch_inplace_delete(wPostPatches[axonID][k]);
         }
         free(wPostPatches[axonID]);
      }
      free(wPostPatches);
   }

   return 0;
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
 *      This is the total number of weight patches for a given axon.
 *      Is the number of pre-synaptic neurons including margins.
 *      - activity and STDP M variable are extended into margins
 *      .
 *
 */
int HyPerConn::createAxonalArbors(int arborId)
{
   //PVParams * inputParams = parent->parameters();

   // these strides are for post-synaptic phi variable, a non-extended layer variable
   //
   // const int psf = 1;
   // const int psx = nfp;
   // const int psy = psx * post->getLayerLoc()->nx;
   postNonextStrides.sf = 1;
   postNonextStrides.sx = nfp;
   postNonextStrides.sy = nfp * post->getLayerLoc()->nx;
   postExtStrides.sf = 1;
   postExtStrides.sx = nfp;
   postExtStrides.sy = nfp * (post->getLayerLoc()->nx+2*post->getLayerLoc()->nb);

   // activity is extended into margins
   //
   //for (int n = 0; n < numAxons; n++) {
   int numPatches = numWeightPatches();
   // PVAxonalArbor* newArbor = (PVAxonalArbor*) calloc(numPatches, sizeof(PVAxonalArbor));
   // assert(newArbor != NULL);
   // setArbor(newArbor, arborId);
   // assert(axonalArborList[arborId] != NULL);
   //}

   for (int kex = 0; kex < numPatches; kex++) {
      // PVAxonalArbor * arbor = axonalArbor(kex, arborId);

      // kex is in extended frame, this makes transformations more difficult
      int kl, offset, nxPatch, nyPatch, dx, dy;
      calcPatchSize(arborId, kex, &kl, &offset, &nxPatch, &nyPatch, &dx, &dy);

      // arbor->data = &dataPatches[kex];
      // arbor->plasticIncr = NULL;   // set later by initPlasticityPatches

      // arbor->delay=(int) inputParams->value(name, "delay", 0);
      // initialize the receiving (of spiking data) gSyn variable
      pvdata_t * gSyn = post->getChannel(channel) + kl;
      gSynPatchStart[arborId][kex] = gSyn;
      // pvpatch_init(arbor->data, nxPatch, nyPatch, nfp, psx, psy, psf, gSyn);

      // arbor->offset = offset;
      aPostOffset[arborId][kex] = offset;

      // adjust patch size (shrink) to fit within interior of post-synaptic layer
      //
      pvpatch_adjust(getWeights(kex,arborId), sxp, syp, nxPatch, nyPatch, dx, dy);

   } // loop over arbors (pre-synaptic neurons)
   //} // loop over neighbors
   //delays[arborId] = (int) inputParams->value(name, "delay", 0);
   delays[arborId] = defaultDelay;

   return 0;
}

PVPatch *** HyPerConn::convertPreSynapticWeights(float time)
{
   if (time <= wPostTime) {
      return wPostPatches;
   }
   wPostTime = time;

   const PVLayer * lPre  = pre->getCLayer();
   const PVLayer * lPost = post->getCLayer();

   const int xScale = post->getXScale() - pre->getXScale();
   const int yScale = post->getYScale() - pre->getYScale();
   const double powXScale = pow(2.0f, (double) xScale);
   const double powYScale = pow(2.0f, (double) yScale);

// fixed?
   // TODO - fix this
//   assert(xScale <= 0);
//   assert(yScale <= 0);

   const int prePad = lPre->loc.nb;

   // pre-synaptic weights are in extended layer reference frame
   const int nxPre = lPre->loc.nx + 2 * prePad;
   const int nyPre = lPre->loc.ny + 2 * prePad;
   const int nfPre = lPre->loc.nf;

   const int nxPost  = lPost->loc.nx;
   const int nyPost  = lPost->loc.ny;
   const int nfPost  = lPost->loc.nf;
   const int numPost = lPost->numNeurons;

   const int nxPostPatch = (int) (nxp * powXScale);
   const int nyPostPatch = (int) (nyp * powYScale);
   const int nfPostPatch = lPre->loc.nf;

   // the number of features is the end-point value (normally post-synaptic)
   const int numPostPatch = nxPostPatch * nyPostPatch * nfPostPatch;

   if (wPostPatches == NULL) {
      wPostPatches = (PVPatch***) calloc(numAxonalArborLists, sizeof(PVPatch**));
      assert(wPostPatches!=NULL);
      for(int axonID=0;axonID<numberOfAxonalArborLists();axonID++) {
         // probable memory leak here since PVPatches and data are now allocated separately
         createWeights(wPostPatches, numPost, nxPostPatch, nyPostPatch, nfPostPatch, axonID);
      }
   }

   //loop through all axons:
   for(int axonID=0;axonID<numberOfAxonalArborLists();axonID++) {

      // loop through post-synaptic neurons (non-extended indices)

      for (int kPost = 0; kPost < numPost; kPost++) {
         int kxPost = kxPos(kPost, nxPost, nyPost, nfPost);
         int kyPost = kyPos(kPost, nxPost, nyPost, nfPost);
         int kfPost = featureIndex(kPost, nxPost, nyPost, nfPost);

         int kxPreHead = zPatchHead(kxPost, nxPostPatch, post->getXScale(), pre->getXScale());
         int kyPreHead = zPatchHead(kyPost, nyPostPatch, post->getYScale(), pre->getYScale());

         // convert kxPreHead and kyPreHead to extended indices
         kxPreHead += prePad;
         kyPreHead += prePad;

         // TODO - FIXME for powXScale > 1
   //      int ax = (int) (1.0f / powXScale);
   //      int ay = (int) (1.0f / powYScale);
   //      int xShift = (ax - 1) - (kxPost + (int) (0.5f * ax)) % ax;
   //      int yShift = (ay - 1) - (kyPost + (int) (0.5f * ay)) % ay;

         for (int kp = 0; kp < numPostPatch; kp++) {

            // calculate extended indices of presynaptic neuron {kPre, kzPre}
            int kxPostPatch = (int) kxPos(kp, nxPostPatch, nyPostPatch, nfPre);
            int kyPostPatch = (int) kyPos(kp, nxPostPatch, nyPostPatch, nfPre);
            int kfPostPatch = (int) featureIndex(kp, nxPostPatch, nyPostPatch, nfPre);

            int kxPre = kxPreHead + kxPostPatch;
            int kyPre = kyPreHead + kyPostPatch;
            int kfPre = kfPostPatch;
            int kPre = kIndex(kxPre, kyPre, kfPre, nxPre, nyPre, nfPre);

            // if {kPre, kzPre} out of bounds, set post weight to zero
            if (kxPre < 0 || kyPre < 0 || kxPre >= nxPre || kyPre >= nyPre) {
               assert(kxPre < 0 || kyPre < 0 || kxPre >= nxPre || kyPre >= nyPre);
               wPostPatches[axonID][kPost]->data[kp] = 0.0;
            }
            else {
               //int arbor = 0;
               PVPatch * p = wPatches[axonID][kPre];
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

               assert(nxp_post == p->nx);
               assert(nyp_post == p->ny);
               assert(nfp == lPost->loc.nf);

               int kxPrePatch, kyPrePatch; // relative index in shrunken patch
               kxPrePatch = kxPost - kxPostHead;
               kyPrePatch = kyPost - kyPostHead;
               int kPrePatch = kfPost * sfp + kxPrePatch * sxp + kyPrePatch * syp;
               wPostPatches[axonID][kPost]->data[kp] = p->data[kPrePatch];

            }
         }
      }
   }
   return wPostPatches;
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

   const PVLayer * lPre  = pre->getCLayer();
   const PVLayer * lPost = post->getCLayer();

   const int prePad  = lPre->loc.nb;

   const int nxPre  = lPre->loc.nx;
   const int nyPre  = lPre->loc.ny;
   const int kx0Pre = lPre->loc.kx0;
   const int ky0Pre = lPre->loc.ky0;
   const int nfPre  = lPre->loc.nf;

   const int nxexPre = nxPre + 2 * prePad;
   const int nyexPre = nyPre + 2 * prePad;

   const int nxPost  = lPost->loc.nx;
   const int nyPost  = lPost->loc.ny;
   const int kx0Post = lPost->loc.kx0;
   const int ky0Post = lPost->loc.ky0;

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

int HyPerConn::writePostSynapticWeights(float timef, bool last) {
   int status = PV_SUCCESS;
   char path[PV_PATH_MAX];

   const PVLayer * lPre  = pre->getCLayer();
   const PVLayer * lPost = post->getCLayer();

   float minVal = FLT_MAX;
   float maxVal = -FLT_MAX;
   for(int arbor=0; arbor<this->numberOfAxonalArborLists(); arbor++) {
      float minVal1 = minWeight(arbor);
      if( minVal1 < minVal ) minVal = minVal1;
      float maxVal1 = maxWeight(arbor);
      if( maxVal1 > maxVal ) maxVal = maxVal1;
   }

   const int numPostPatches = lPost->numNeurons;

   const int xScale = post->getXScale() - pre->getXScale();
   const int yScale = post->getYScale() - pre->getYScale();
   const double powXScale = pow(2, (double) xScale);
   const double powYScale = pow(2, (double) yScale);

   const int nxPostPatch = (int) (nxp * powXScale);
   const int nyPostPatch = (int) (nyp * powYScale);
   const int nfPostPatch = lPre->loc.nf;

   const char * last_str = (last) ? "_last" : "";
   snprintf(path, PV_PATH_MAX-1, "%s/w%d_post%s.pvp", parent->getOutputPath(), getConnectionId(), last_str);

   const PVLayerLoc * loc  = post->getLayerLoc();
   Communicator * comm = parent->icCommunicator();

   bool append = (last) ? false : ioAppend;

   status = PV::writeWeights(path, comm, (double) timef, append,
                             loc, nxPostPatch, nyPostPatch, nfPostPatch, minVal, maxVal,
                             wPostPatches, numPostPatches, numberOfAxonalArborLists(), writeCompressedWeights, PVP_WGT_FILE_TYPE);
   if(status != PV_SUCCESS) {
      if( parent->icCommunicator()->commRank() != 0 ) {
         fflush(stdout);
         fprintf(stderr, "Connection \"%s\": writePostSynapticWeights failed at time %f.  Exiting.\n", name, timef);
      }
      abort();
   }

   return PV_SUCCESS;
}

#ifdef OBSOLETE // Marked obsolete Nov 29, 2011.
int HyPerConn::writePostSynapticWeights(float time, bool last) {
   for(int axonID=0;axonID<numberOfAxonalArborLists();axonID++) {
      writePostSynapticWeights(time, last, axonID);
   }
   return PV_SUCCESS;
}

int HyPerConn::writePostSynapticWeights(float time, bool last, int axonID)
{
   int status = 0;
   char path[PV_PATH_MAX];

   const PVLayer * lPre  = pre->getCLayer();
   const PVLayer * lPost = post->getCLayer();

   const float minVal = minWeight(axonID);
   const float maxVal = maxWeight(axonID);

   const int numPostPatches = lPost->numNeurons;

   const int xScale = post->getXScale() - pre->getXScale();
   const int yScale = post->getYScale() - pre->getYScale();
   const float powXScale = powf(2, (float) xScale);
   const float powYScale = powf(2, (float) yScale);

   const int nxPostPatch = (int) (nxp * powXScale);
   const int nyPostPatch = (int) (nyp * powYScale);
   const int nfPostPatch = lPre->loc.nf;

   const char * last_str = (last) ? "_last" : "";
   if(numberOfAxonalArborLists()>1)
      snprintf(path, PV_PATH_MAX-1, "%s/w%d_a%1.1d_post%s.pvp", parent->getOutputPath(), getConnectionId(), axonID, last_str);
   else
      snprintf(path, PV_PATH_MAX-1, "%s/w%d_post%s.pvp", parent->getOutputPath(), getConnectionId(), last_str);


   const PVLayerLoc * loc  = post->getLayerLoc();
   Communicator   * comm = parent->icCommunicator();

   bool append = (last) ? false : ioAppend;

   status = PV::writeWeights(path, comm, (double) time, append,
                             loc, nxPostPatch, nyPostPatch, nfPostPatch, minVal, maxVal,
                             wPostPatches[axonID], numPostPatches, writeCompressedWeights, PVP_WGT_FILE_TYPE);
   assert(status == 0);

   return 0;
}
#endif // OBSOLETE

int HyPerConn::initNormalize() {
   PVParams * params = parent->parameters();
   normalize_flag = params->value(name, "normalize", normalize_flag);
   if( normalize_flag ) {
      normalize_strength = params->value(name, "strength", 1.0f);
      normalize_max = params->value(name, "normalize_max", normalize_max) != 0.0f;
      normalize_zero_offset = params->value(name, "normalize_zero_offset", normalize_zero_offset) != 0.0f;
      normalize_cutoff = params->value(name, "normalize_cutoff", normalize_cutoff) * normalize_strength;
      if (this->numberOfAxonalArborLists() > 1) {
         normalize_arbors_individually = params->value(name, "normalize_arbors_individually", normalize_arbors_individually) != 0.0f;
      }
   }
   return PV_SUCCESS;
}

int HyPerConn::sumWeights(PVPatch * wp, double * sum, double * sum2, pvdata_t * maxVal)
{
   assert(wp != NULL);
   pvdata_t * w = wp->data;
   assert(w != NULL);
   const int nx = wp->nx;
   const int ny = wp->ny;
   //const int nfp = wp->nf;
   //const int syp = wp->sy;
   double sum_tmp = 0;
   double sum2_tmp = 0;
   pvdata_t max_tmp = -FLT_MAX;
   for (int ky = 0; ky < ny; ky++) {
      for(int iWeight = 0; iWeight < nfp * nx; iWeight++ ){
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

int HyPerConn::scaleWeights(PVPatch * wp, pvdata_t sum, pvdata_t sum2, pvdata_t maxVal)
{
   assert(wp != NULL);
   int num_weights = wp->nx * wp->ny * nfp; //wp->nf;
   if (!this->normalize_arbors_individually){
      num_weights *= numberOfAxonalArborLists(); // assumes all arbors shrunken equally at this point (shrink patches should occur after normalize)
   }
   float sigma2 = ( sum2 / num_weights ) - ( sum / num_weights ) * ( sum / num_weights );
   float zero_offset = 0.0f;
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
   else if (sum != 0.0f) {
      scale_factor = normalize_strength / sum;
   }
   else if (sum == 0.0f && sigma2 > 0.0f) {
      scale_factor = normalize_strength / sqrt(sigma2);
   }
   pvdata_t * w = wp->data;
   assert(w != NULL);
   for (int ky = 0; ky < wp->ny; ky++) {
      for(int iWeight = 0; iWeight < nfp * wp->nx; iWeight++ ){
         w[iWeight] = ( w[iWeight] - zero_offset ) * scale_factor;
         w[iWeight] = ( fabs(w[iWeight]) > fabs(normalize_cutoff) ) ? w[iWeight] : 0.0f;
      }
      w += syp;
   }
   maxVal = ( maxVal - zero_offset ) * scale_factor;
   maxVal = ( fabs(maxVal) > fabs(normalize_cutoff) ) ? maxVal : 0.0f;
   this->wMax = maxVal > this->wMax ? maxVal : this->wMax;
   return PV_SUCCESS;
} // scaleWeights

// only checks for certain combinations of normalize parameter settings
int HyPerConn::checkNormalizeWeights(PVPatch * wp, float sum, float sigma2, float maxVal)
{
   assert( sum != 0 || sigma2 != 0 ); // Calling routine should make sure this condition is met.
   float tol = 0.01f;
   assert(wp != NULL);
   if (normalize_zero_offset && (normalize_cutoff == 0.0f)){  // condition may be violated is normalize_cutoff != 0.0f
      // set sum to zero and normalize std of weights to sigma
      assert((sum > -tol) && (sum < tol));
      assert((sqrt(sigma2) > (1-tol)*normalize_strength) && (sqrt(sigma2) < (1+tol)*normalize_strength));
   }
   else if (normalize_max) {
      // set maximum weight to normalize_strength
      assert((maxVal > (1-tol)*normalize_strength) && ((maxVal < (1+tol)*normalize_strength)));
    }
   else if (normalize_cutoff == 0.0f){  // condition may be violated is normalize_cutoff != 0.0f
      assert((sum > (1-sign(normalize_strength)*tol)*normalize_strength) && ((sum < (1+sign(normalize_strength)*tol)*normalize_strength)));
   }
   return PV_SUCCESS;

} // checkNormalizeWeights

int HyPerConn::checkNormalizeArbor(PVPatch ** patches, int numPatches, int arborId)
{
   int status = PV_SUCCESS;
   for (int k = 0; k < numPatches; k++) {
      PVPatch * wp = patches[k];
      if( wp->nx < nxp || wp->ny < nyp ) {
         continue;  // Normalization of shrunken patches used unshrunken part, which is no longer available
      }
      double sum = 0;
      double sum2 = 0;
      float maxVal = -FLT_MAX;
      status = sumWeights(wp, &sum, &sum2, &maxVal);
      int num_weights = wp->nx * wp->ny * nfp; //wp->nf;
      float sigma2 = ( sum2 / num_weights ) - ( sum / num_weights ) * ( sum / num_weights );
      if( sum != 0 || sigma2 != 0 ) {
         status = checkNormalizeWeights(wp, sum, sigma2, maxVal);
         assert( status == PV_SUCCESS );
      }
      else {
         fprintf(stderr, "checkNormalizeArbor: connection \"%s\", arbor %d, patch %d has all zero weights.\n", name, arborId, k);
      }
   }
   return PV_SUCCESS;
} // checkNormalizeArbor

int HyPerConn::normalizeWeights(PVPatch ** patches, int numPatches, int arborId)
{
   int status = PV_SUCCESS;
   this->wMax = -FLT_MAX;
   for (int k = 0; k < numPatches; k++) {
      PVPatch * wp = patches[k];
      float maxVal = -FLT_MAX;
      double sum = 0;
      double sum2 = 0;
      status = sumWeights(wp, &sum, &sum2, &maxVal);
      assert( (status == PV_SUCCESS) || (status == PV_BREAK) );
      status = scaleWeights(wp, sum, sum2, maxVal);
      assert( (status == PV_SUCCESS) || (status == PV_BREAK) );
   } // k < numPatches
   status = HyPerConn::checkNormalizeArbor(patches, numPatches, arborId); // no polymorphism here until HyPerConn generalized to normalize_arbor_individually == false
   assert( (status == PV_SUCCESS) || (status == PV_BREAK) );
   return PV_SUCCESS;
} // normalizeWeights


int HyPerConn::calcPatchSize(int axon_index, int kex,
                             int * kl_out, int * offset_out,
                             int * nxPatch_out, int * nyPatch_out,
                             int * dx_out, int * dy_out)
{
   int status = PV_SUCCESS;

   const PVLayer * lPre  = pre->getCLayer();
   const PVLayer * lPost = post->getCLayer();

   const int prePad  = lPre->loc.nb;
   const int postPad = lPost->loc.nb;

   const int nxPre  = lPre->loc.nx;
   const int nyPre  = lPre->loc.ny;
   const int kx0Pre = lPre->loc.kx0;
   const int ky0Pre = lPre->loc.ky0;
   const int nfPre  = lPre->loc.nf;

   const int nxexPre = nxPre + 2 * prePad;
   const int nyexPre = nyPre + 2 * prePad;

   const int nxPost  = lPost->loc.nx;
   const int nyPost  = lPost->loc.ny;
   const int kx0Post = lPost->loc.kx0;
   const int ky0Post = lPost->loc.ky0;
   const int nfPost  = lPost->loc.nf;

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

   // local non-extended index but shifted to be in bounds
   int kl = kIndex(kxPost, kyPost, kfPost, nxPost, nyPost, nfPost);
   assert(kl >= 0);
   assert(kl < lPost->numNeurons);

   // get offset in extended frame
   kxPost += postPad;
   kyPost += postPad;

   int offset = kIndex(kxPost, kyPost, kfPost, nxexPost, nyexPost, nfPost);
   assert(offset >= 0);
   assert(offset < lPost->numExtended);

   // set return variables
   *kl_out = kl;
   *offset_out = offset;
   *nxPatch_out = nxPatch;
   *nyPatch_out = nyPatch;
   *dx_out = dx;
   *dy_out = dy;

   return status;
}

int HyPerConn::setPatchSize(const char * filename)
{
   int status;
   PVParams * inputParams = parent->parameters();

   nxp = (int) inputParams->value(name, "nxp", post->getCLayer()->loc.nx);
   nyp = (int) inputParams->value(name, "nyp", post->getCLayer()->loc.ny);
   nfp = (int) inputParams->value(name, "nfp", post->getCLayer()->loc.nf);
   if( nfp != post->getCLayer()->loc.nf ) {
      fprintf( stderr, "Params file specifies %d features for connection \"%s\",\n", nfp, name );
      fprintf( stderr, "but %d features for post-synaptic layer %s\n",
               post->getCLayer()->loc.nf, post->getName() );
      exit(PV_FAILURE);
   }
   int xScalePre = pre->getXScale();
   int xScalePost = post->getXScale();
   status = checkPatchSize(nxp, xScalePre, xScalePost, 'x');
   if( status != PV_SUCCESS) return status;

   int yScalePre = pre->getYScale();
   int yScalePost = post->getYScale();
   status = checkPatchSize(nyp, yScalePre, yScalePost, 'y');
   if( status != PV_SUCCESS) return status;

   status = PV_SUCCESS;
   if( filename != NULL ) {
      bool useListOfArborFiles = inputParams->value(name, "useListOfArborFiles", false)!=0;
      if( !useListOfArborFiles ) status = patchSizeFromFile(filename);
   }

   return status;
}

int HyPerConn::patchSizeFromFile(const char * filename) {
   // use patch dimensions from file if (filename != NULL)
   //
   int status;
   int filetype, datatype;
   double time = 0.0;
   const PVLayerLoc loc = pre->getCLayer()->loc;

   int wgtParams[NUM_WGT_PARAMS];
   int numWgtParams = NUM_WGT_PARAMS;

   Communicator * comm = parent->icCommunicator();

   char nametmp[PV_PATH_MAX];
   for (int arborId = 0; arborId < this->numberOfAxonalArborLists(); arborId++){
      snprintf(nametmp, PV_PATH_MAX-1, "%s", filename);

      status = pvp_read_header(nametmp, comm, &time, &filetype, &datatype, wgtParams, &numWgtParams);
      if (status < 0) return status;

      status = checkPVPFileHeader(comm, &loc, wgtParams, numWgtParams);
      if (status < 0) return status;

      // reconcile differences with inputParams
      status = checkWeightsHeader(nametmp, wgtParams);
   }
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
         fprintf(stderr, "(postsynaptic scale) = %d * (postsynaptic scale);\n", scaleFactor);
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
   sfp = 1;
   sxp = nfp;
   syp = nfp * nxp;
   return PV_SUCCESS;
}

pvdata_t * HyPerConn::allocWeights(PVPatch *** patches, int nPatches, int nxPatch,
      int nyPatch, int nfPatch, int axonId)
{
//   set_wDataStart(axonId, pvpatches_inplace_new(patches, nxPatch, nyPatch, nfPatch, nPatches));
   pvdata_t * dataPatches = pvpatches_inplace_new(patches[axonId], nxPatch, nyPatch, nfPatch, nPatches);
//   for (int k = 0; k < nPatches; k++) {
//      patches[k] = pvpatch_inplace_new(nxPatch, nyPatch, nfPatch);
//   }
   return dataPatches;
}

//PVPatch ** HyPerConn::allocWeights(PVPatch ** patches)
//{
//   int arbor = 0;
//   int nPatches = numWeightPatches();
//   int nxPatch = nxp;
//   int nyPatch = nyp;
//   int nfPatch = nfp;
//
//   return allocWeights(patches, nPatches, nxPatch, nyPatch, nfPatch);
//}

// one to many mapping, chose first patch index in restricted space
// kernelIndex for unit cell
// patchIndex in extended space
int HyPerConn::kernelIndexToPatchIndex(int kernelIndex, int * kxPatchIndex,
      int * kyPatchIndex, int * kfPatchIndex)
{
   return kernelIndex;
}

// many to one mapping from weight patches to kernels
// patchIndex always in extended space
// kernelIndex always for unit cell
int HyPerConn::patchIndexToKernelIndex(int patchIndex, int * kxKernelIndex,
      int * kyKernelIndex, int * kfKernelIndex)
{
   const PVLayerLoc * loc = preSynapticLayer()->getLayerLoc();
   if(kxKernelIndex) *kxKernelIndex = kxPos(patchIndex,loc->nx+2*loc->nb,loc->ny+2*loc->nb,loc->nf);
   if(kyKernelIndex) *kyKernelIndex = kyPos(patchIndex,loc->nx+2*loc->nb,loc->ny+2*loc->nb,loc->nf);
   if(kfKernelIndex) *kfKernelIndex = featureIndex(patchIndex,loc->nx+2*loc->nb,loc->ny+2*loc->nb,loc->nf);
   return patchIndex;
}

void HyPerConn::connOutOfMemory(const char * funcname) {
   fprintf(stderr, "Out of memory error in %s for connection \"%s\"\n", funcname, name);
   exit(EXIT_FAILURE);
}

} // namespace PV
