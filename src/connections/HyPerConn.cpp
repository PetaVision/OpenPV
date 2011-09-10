/*
 * HyPerConn.cpp
 *
 *  Created on: Oct 21, 2008
 *      Author: Craig Rasmussen
 */

#include "HyPerConn.hpp"
#include "../layers/LIF.hpp"
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
#include "../connections/InitWeights.hpp"
#include "../connections/InitCocircWeights.hpp"
#include "../connections/InitSmartWeights.hpp"
#include "../connections/InitUniformRandomWeights.hpp"
#include "../connections/InitGaussianRandomWeights.hpp"

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

   // assert(params != NULL); // freeing a null pointer is not an error
   //free(params);

   deleteWeights();

   // free the task information

   for (int l = 0; l < numAxonalArborLists; l++) {
      if ( axonalArborList[l] ) {
         free(axonalArbor(0, l)->data);
         // axonalArbor(0,l) frees all data patches for arbor l because all
         // axonalArbor patches for that l were created in a single calloc().
         free(axonalArborList[l]);
      }
   }

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

#ifdef OBSOLETE_STDP
   // STDP parameters for modifying weights
   this->pIncr = NULL;
   this->pDecr = NULL;
   this->ampLTP = 1.0;
   this->ampLTD = 1.1;
   this->tauLTP = 20;
   this->tauLTD = 20;
   this->dWMax = 0.1;
   this->stdpFlag = false;
   this->localWmaxFlag = false;
#endif
   this->wMin = 0.0;
   this->wMax = 1.0;
   this->wPostTime = -1.0;
   this->wPostPatches = NULL;
   this->writeCompressedWeights = true;

   wPatches=NULL;
   axonalArborList=NULL;
   pIncr = NULL;
//   for (int i = 0; i < MAX_ARBOR_LIST; i++) {
//      wPatches[i] = NULL;
//      axonalArborList[i] = NULL;
//   }
   this->normalize_flag = true; // default value, overridden by params file parameter "normalize" in initNormalize()

   return PV_SUCCESS;
}

int HyPerConn::createArbors() {
   PVParams * inputParams = parent->parameters();
   numAxonalArborLists=(int) inputParams->value(name, "numAxonalArbors", 1);
   wPatches = (PVPatch***) calloc(numAxonalArborLists, sizeof(PVPatch**));
   assert(wPatches != NULL);
   axonalArborList = (PVAxonalArbor**) calloc(numAxonalArborLists, sizeof(PVAxonalArbor*));
   assert(axonalArborList != NULL);
   return PV_SUCCESS;
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

   //allocate the arbor arrays:
   createArbors();


   //const int arbor = 0;
   //numAxonalArborLists = 1;

   setPatchSize(filename);

   //allocate weightpatches and axonal arbors for each arbor
   for (int arborId=0;arborId<numAxonalArborLists;arborId++) {
      PVPatch **arborPatch = weights(arborId);
      arborPatch = createWeights(arborPatch, arborId);
      assert(arborPatch != NULL);
      setWPatches(arborPatch, arborId);
      //wPatches[arborId] = createWeights(wPatches[arborId]); //allocates memory for weights

   #ifdef OBSOLETE_STDP
      initializeSTDP();
   #endif

      // Create list of axonal arbors containing pointers to {phi,w,P,M} patches.
      //  weight patches may shrink
      status |= createAxonalArbors(arborId);
      assert(weights(arborId) != NULL);
   }  // arborId

   //initialize weights for patches:
   assert( initializeWeights(wPatches, numWeightPatches(), filename) != NULL);
   status |= initPlasticityPatches();
   assert(status == 0);

   return status;
}

#ifdef OBSOLETE_STDP
/*
 * Using a dynamic_cast operator to convert (downcast) a pointer to a base class (HyPerLayer)
 * to a pointer to a derived class (LIF). This way I do not need to define a virtual
 * function getWmax() in HyPerLayer which only returns a NULL pointer in the base class.
 * .
 */
int HyPerConn::initializeSTDP()
{
   int arbor = 0;
   if (stdpFlag) {
      int numPatches = numWeightPatches(arbor);
      pIncr = createWeights(NULL, numPatches, nxp, nyp, nfp);
      assert(pIncr != NULL);
      pDecr = pvcube_new(&post->getCLayer()->loc, post->getNumExtended());
      assert(pDecr != NULL);
      
      if(localWmaxFlag){
         LIF * LIF_layer = dynamic_cast<LIF *>(post);
         assert(LIF_layer != NULL);
         Wmax = LIF_layer->getWmax();
         assert(Wmax != NULL);
      } else {
         Wmax = NULL;
      }
   }
   else {
      pIncr = NULL;
      pDecr = NULL;
      Wmax  = NULL;
   }
   return 0;
}
#endif

int HyPerConn::initialize(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, ChannelType channel, const char * filename) {
   return initialize(name, hc, pre, post, channel, filename, NULL);
}


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

   //if a weightinitializer hasn't been created already, use the default--> either 2D Gauss or read from file
   if(weightInit==NULL) {
      PVParams * inputParams = parent->parameters();
      bool randomFlag = inputParams->value(name, "randomFlag", 0.0f, false) != 0;
      bool smartWeights = inputParams->value(name, "smartWeights",0.0f, false) != 0;
      bool cocircWeights = inputParams->value(name, "cocircWeights",0.0f, false) != 0;
      bool uniform_weights = inputParams->value(name, "uniformWeights", 1.0f, false) != 0;
      bool gaussian_weights = inputParams->value(name, "gaussianWeights", 0.0f, false) != 0;

      if ((randomFlag)&&(uniform_weights)) { // if (randomFlag != 0 || randomSeed != 0) {
         this->weightInitializer = new InitUniformRandomWeights();
         //initializeRandomWeights(patches, numPatches);
      }
      else if ((randomFlag)&&(gaussian_weights)) { // if (randomFlag != 0 || randomSeed != 0) {
         this->weightInitializer = new InitGaussianRandomWeights();
         //initializeRandomWeights(patches, numPatches);
      }
      else if (smartWeights) {
         this->weightInitializer = new InitSmartWeights();
         //initializeSmartWeights(patches, numPatches);
      }
      else if (cocircWeights) {
         this->weightInitializer = new InitCocircWeights();
         //initializeCocircWeights(patches, numPatches);
      }
      else {
         this->weightInitializer = new InitWeights();
         //inputParams->value(getName(), "gauss2DCalcWeights", 1.0f, true); // generate message if no method was set in params.
         //initializeDefaultWeights(patches, numPatches);
      }
      fprintf(stderr, "Connection \"%s: This method of initializing weights has been deprecated.\n"
                      "  Please pass an InitWeights object to the constructor.\n"
                      "  In buildandrun(), use the string parameter \"weightInitType\" to set the InitWeights object.\n", name);

   }
   else {
      this->weightInitializer = weightInit;
   }


   this->connId = parent->addConnection(this);

   writeTime = parent->simulationTime();
   writeStep = parent->parameters()->value(name, "writeStep", parent->getDeltaTime());

   status = setParams(hc->parameters() /*, &defaultConnParams*/);
   constructWeights(filename);


   return status;
}

int HyPerConn::initPlasticityPatches()
{
   if (!plasticityFlag) return PV_SUCCESS;

   //const int arbor = 0;
   const int numAxons = numberOfAxonalArborLists();

   pIncr = (PVPatch***) calloc(numAxons, sizeof(PVPatch**));
   assert(pIncr != NULL);

   int numArbors = numWeightPatches();
   for (int arborId = 0; arborId < numAxons; arborId++) {

      //pIncr[n] = createWeights(NULL, numWeightPatches(), nxp, nyp, nfp, 0);
      //assert(pIncr[n] != NULL);
      PVPatch** dWPatch = createWeights(NULL, numWeightPatches(), nxp, nyp, nfp, 0);
      assert(dWPatch != NULL);


      // kex is in extended frame
      for (int kex = 0; kex < numArbors; kex++) {
         int kl, offset, nxPatch, nyPatch, dx, dy;
         PVAxonalArbor * arbor = axonalArbor(kex, arborId);

         calcPatchSize(arborId, kex, &kl, &offset, &nxPatch, &nyPatch, &dx, &dy);

         // adjust patch size (shrink) to fit within interior of post-synaptic layer
         //
         //arbor->plasticIncr = pIncr[n][kex];
         arbor->plasticIncr = dWPatch[kex];
         pvpatch_adjust(arbor->plasticIncr, nxPatch, nyPatch, dx, dy);

      } // loop over pre-synaptic neurons
      setdWPatches(dWPatch, arborId);

   } // loop over arbors

   return PV_SUCCESS;
}

// set member variables specified by user
int HyPerConn::setParams(PVParams * filep /*, PVConnParams * p*/)
{
   const char * name = getName();

//   params = (PVConnParams *) malloc(sizeof(*p));
//   assert(params != NULL);
//   memcpy(params, p, sizeof(*p));
//
//   numParams = sizeof(*p) / sizeof(float);
//   assert(numParams == 1); // catch changes in structure
//
//   params->delay    = (int) filep->value(name, "delay", params->delay);
//   //params->fixDelay = (int) filep->value(name, "fixDelay", params->fixDelay);
//
//   //params->vel      = filep->value(name, "vel", params->vel);
//   //params->rmin     = filep->value(name, "rmin", params->rmin);
//   //params->rmax     = filep->value(name, "rmax", params->rmax);
//
//   //params->varDelayMin = (int) filep->value(name, "varDelayMin", params->varDelayMin);
//   //params->varDelayMax = (int) filep->value(name, "varDelayMax", params->varDelayMax);
//   //params->numDelay    = (int) filep->value(name, "numDelay"   , params->numDelay);
//   //params->isGraded    = (int) filep->value(name, "isGraded"   , params->isGraded);
//
//   assert(params->delay < MAX_F_DELAY);
   //params->numDelay = params->varDelayMax - params->varDelayMin + 1;

   //
   // now set params that are not in the params struct (instance variables)

#ifdef OBSOLETE_STDP
   stdpFlag = (bool) filep->value(name, "stdpFlag", (float) stdpFlag);
   if (stdpFlag) {
      ampLTP = filep->value(name, "ampLTP", ampLTP);
      ampLTD = filep->value(name, "ampLTD", ampLTD);
      tauLTP = filep->value(name, "tauLTP", tauLTP);
      tauLTD = filep->value(name, "tauLTD", tauLTD);
#endif

      plasticityFlag = filep->value(name, "plasticityFlag", plasticityFlag, false);

      // moved to STDPConn (not used elsewhere in project) !!! whoever keeps putting wMax back STOP!!!!
      // let wMax override strength if user provides it
      //wMax = filep->value(name, "wMax", wMax);
      //wMin = filep->value(name, "wMin", wMin);

#ifdef OBSOLETE_STDP
      dWMax = filep->value(name, "dWMax", dWMax);

      // set params for rate dependent Wmax
      localWmaxFlag = (bool) filep->value(name, "localWmaxFlag", (float) localWmaxFlag);
   }
#endif

   writeCompressedWeights = filep->value(name, "writeCompressedWeights", true);

   return 0;
}

// returns handle to initialized weight patches
PVPatch *** HyPerConn::initializeWeights(PVPatch *** arbors, int numPatches, const char * filename)
{
   // TODO  Implement InitWeightsMethod class.  The constructor for HyPerConn would take an InitWeightsMethod
   //       instantiation as an argument.  The routines called below would be put into derived classes
   //       of InitWeightsMethod.
//   PVParams * inputParams = parent->parameters();
//
//   int initFromLastFlag = inputParams->value(getName(), "initFromLastFlag", 0.0f, false) != 0;
//   int randomFlag = inputParams->value(getName(), "randomFlag", 0.0f, false) != 0;
//   int smartWeights = inputParams->value(getName(), "smartWeights",0.0f, false) != 0;
//   int cocircWeights = inputParams->value(getName(), "cocircWeights",0.0f, false) != 0;

//   if( filename != NULL ) {
//      //readWeights(patches, numPatches, filename);
//   }
//   else if (initFromLastFlag) {
//      char name[PV_PATH_MAX];
//      snprintf(name, PV_PATH_MAX-1, "%s/w%1.1d_last.pvp", parent->getOutputPath(), getConnectionId());
//      //readWeights(patches, numPatches, name);
//   }
//   else if (randomFlag) { // if (randomFlag != 0 || randomSeed != 0) {
//       initializeRandomWeights(patches, numPatches);
//   }
//   else if (smartWeights) {
//       initializeSmartWeights(patches, numPatches);
//   }
//   else if (cocircWeights) {
//       initializeCocircWeights(patches, numPatches);
//   }
//   else {
//      inputParams->value(getName(), "gauss2DCalcWeights", 1.0f, true); // generate message if no method was set in params.
//      initializeDefaultWeights(patches, numPatches);
//   }

   for(int arborId=0; arborId<numberOfAxonalArborLists(); arborId++) {

//      if (initFromLastFlag) {
//         char nametmp[PV_PATH_MAX];
//         snprintf(nametmp, PV_PATH_MAX-1, "%s/w%1.1d_last.pvp", parent->getOutputPath(), getConnectionId());
//         weightInitializer->initializeWeights(arbors[arborId], numPatches, nametmp, this);
//      }
//      else {
         weightInitializer->initializeWeights(arbors[arborId], arborId, numPatches, filename, this);

         //call to original for comparing result with my new one:
         //initializeDefaultWeights(patches, numPatches);
         //initializeCocircWeights(patches, numPatches);
         //initializeSmartWeights(patches, numPatches);
//      }

      // bool normalize_flag = (bool) inputParams->value(getName(), "normalize", 0.0f, true);
      initNormalize(); // Sets normalize_flag; derived-class methods that override initNormalize must also set normalize_flag
      if (normalize_flag) {
         normalizeWeights(arbors[arborId], numPatches, arborId);
      }
   }
   return arbors;
}

#ifdef PV_USE_OPENCL
/**
 * Initialize OpenCL buffers.  This must be called after weights have
 * been allocated.
 */
int HyPerConn::initializeThreadBuffers(const char * kernel_name)
{
   int status = CL_SUCCESS;

//   const size_t size    = getNumNeurons()  * sizeof(pvdata_t);
//   const size_t size_ex = getNumExtended() * sizeof(pvdata_t);

   CLDevice * device = parent->getCLDevice();

   // these buffers are shared between host and device
   //

   // TODO - create device buffers for weights

   return status;
}

int HyPerConn::initializeThreadKernels(const char * kernel_name)
{
   char kernelPath[PV_PATH_MAX+128];
   char kernelFlags[PV_PATH_MAX+128];

   int status = CL_SUCCESS;
   CLDevice * device = parent->getCLDevice();

   sprintf(kernelPath, "%s/src/kernels/%s.cl", parent->getPath(), kernel_name);
   sprintf(kernelFlags, "-D PV_USE_OPENCL -cl-fast-relaxed-math -I %s/src/kernels/", parent->getPath());

   // create kernels
   //

   return status;
}
#endif

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
              "ignoring nxp = %i in HyPerCol %s, using nxp = %i in binary file %s\n",
              nxp, name, nxpFile, filename);
      nxp = nxpFile;
   }
   if (nyp != nypFile) {
      fprintf(stderr,
              "ignoring nyp = %i in HyPerCol %s, using nyp = %i in binary file %s\n",
              nyp, name, nypFile, filename);
      nyp = nypFile;
   }
   if (nfp != nfpFile) {
      fprintf(stderr,
              "ignoring nfp = %i in HyPerCol %s, using nfp = %i in binary file %s\n",
              nfp, name, nfpFile, filename);
      nfp = nfpFile;
   }
   return 0;
}
int HyPerConn::correctPIndex(int patchIndex) {
   return patchIndex;
}
#ifdef OBSOLETE //The following methods have been added to the new InitWeights classes.  Please
                //use the param "weightInitType" to choose an initialization type
/*!
 * NOTES:
 *    - numPatches also counts the neurons in the boundary layer. It gives the size
 *    of the extended neuron space.
 *
 */
// PVPatch ** HyPerConn::initializeRandomWeights(PVPatch ** patches, int numPatches,
//      int seed)
PVPatch ** HyPerConn::initializeRandomWeights(PVPatch ** patches, int numPatches)
{
   PVParams * inputParams = parent->parameters();

   float uniform_weights = inputParams->value(getName(), "uniformWeights", 1.0f, false);
   float gaussian_weights = inputParams->value(getName(), "gaussianWeights", 0.0f, false);

   if (uniform_weights && gaussian_weights) {
      fprintf(stderr,"multiple random weights distributions defined:  Exiting\n");
      exit(PV_FAILURE);
   }

   if( !(uniform_weights || gaussian_weights) ) {
      fprintf(stderr,"When randomFlag is set, either uniformWeights or gaussianWeights must be specified.  Exiting\n");
      exit(PV_FAILURE);
   }

   if(uniform_weights) {
      inputParams->value(getName(), "uniformWeights", uniform_weights, true); // generate warning if uniformWeights set by default
   }

   if (uniform_weights) {
      float wMinInit = inputParams->value(getName(), "wMinInit", (float) wMin);
      float wMaxInit = inputParams->value(getName(), "wMaxInit", (float) wMax);

      for (int k = 0; k < numPatches; k++) {
         uniformWeights(patches[k], wMinInit, wMaxInit);
         // uniformWeights(patches[k], wMinInit, wMaxInit, &seed); // MA
      }
   }
   else if (gaussian_weights) {
         float wGaussMean = inputParams->value(getName(), "wGaussMean", 0.5f);
         float wGaussStdev = inputParams->value(getName(), "wGaussStdev", 0.1f);
         // int seed = (int) inputParams->value(getName(), "randomSeed", 0);
         // randomSeed now part of HyPerCol
         for (int k = 0; k < numPatches; k++) {
            gaussianWeights(patches[k], wGaussMean, wGaussStdev);
            // gaussianWeights(patches[k], wGaussMean, wGaussStdev, &seed); // MA (seed not used)
         }
      }
   else{
      // fprintf(stderr,"no random weights distribution was defined: exit\n");
      // exit(-1);
      assert(0); // Because of the checking done before this if-statement, it should be impossible to reach here.
   }
   return patches;
}

/*!
 * NOTES:
 *    - numPatches also counts the neurons in the boundary layer. It gives the size
 *    of the extended neuron space.
 *
 */
PVPatch ** HyPerConn::initializeSmartWeights(PVPatch ** patches, int numPatches)
{

   for (int k = 0; k < numPatches; k++) {
      smartWeights(patches[k], k); // MA
   }
   return patches;
}


PVPatch ** HyPerConn::initializeDefaultWeights(PVPatch ** patches, int numPatches)
{
   return initializeGaussian2DWeights(patches, numPatches);
}

PVPatch ** HyPerConn::initializeGaussian2DWeights(PVPatch ** patches, int numPatches)
{
   PVParams * params = parent->parameters();

   // default values (chosen for center on cell of one pixel)
   int noPost = nfp;
   float aspect = 1.0; // circular (not line oriented)
   float sigma = 0.8;
   float rMax = 1.4;
   float strength = 1.0;
   float deltaThetaMax = 2.0f * PI;  // max difference in orientation between pre and post
   float thetaMax = 1.0;  // max orientation in units of PI
   int numFlanks = 1;
   float shift = 0.0f;
   float rotate = 0.0f;   // rotate so that axis isn't aligned
   float bowtieFlag = 0.0f;  // flag for setting bowtie angle
   float bowtieAngle = PI * 2.0f;  // bowtie angle

   aspect   = params->value(name, "aspect", aspect);
   sigma    = params->value(name, "sigma", sigma);
   rMax     = params->value(name, "rMax", rMax);
   strength = params->value(name, "strength", strength);
   if (nfp > 1) {
      noPost = (int) params->value(post->getName(), "no", nfp);
      deltaThetaMax = params->value(name, "deltaThetaMax", deltaThetaMax);
      thetaMax = params->value(name, "thetaMax", thetaMax);
      numFlanks = (int) params->value(name, "numFlanks", (float) numFlanks);
      shift = params->value(name, "flankShift", shift);
      rotate = params->value(name, "rotate", rotate);
      bowtieFlag = params->value(name, "bowtieFlag", bowtieFlag);
      if (bowtieFlag == 1.0f) {
         bowtieAngle = params->value(name, "bowtieAngle", bowtieAngle);
      }
   }

   float r2Max = rMax * rMax;

   for (int patchIndex = 0; patchIndex < numPatches; patchIndex++) {
      gauss2DCalcWeights(patches[patchIndex], patchIndex, noPost, numFlanks, shift, rotate,
            aspect, sigma, r2Max, strength, deltaThetaMax, thetaMax, bowtieFlag, bowtieAngle);
   }

   return patches;
}

PVPatch ** HyPerConn::initializeCocircWeights(PVPatch ** patches, int numPatches)
{
   PVParams * params = parent->parameters();
   float aspect = 1.0; // circular (not line oriented)
   float sigma = 0.8;
   float rMax = 1.4;
   float strength = 1.0;

   aspect = params->value(name, "aspect", aspect);
   sigma = params->value(name, "sigma", sigma);
   rMax = params->value(name, "rMax", rMax);
   strength = params->value(name, "strength", strength);

   float r2Max = rMax * rMax;

   int numFlanks = 1;
   float shift = 0.0f;
   float rotate = 0.0f; // rotate so that axis isn't aligned

   numFlanks = (int) params->value(name, "numFlanks", numFlanks);
   shift = params->value(name, "flankShift", shift);
   rotate = params->value(name, "rotate", rotate);

   int noPre = pre->getLayerLoc()->nf;
   noPre = (int) params->value(name, "noPre", noPre);
   assert(noPre > 0);
   assert(noPre <= pre->getLayerLoc()->nf);

   int noPost = post->getLayerLoc()->nf;
   noPost = (int) params->value(name, "noPost", noPost);
   assert(noPost > 0);
   assert(noPost <= post->getLayerLoc()->nf);

   float sigma_cocirc = PI / 2.0;
   sigma_cocirc = params->value(name, "sigmaCocirc", sigma_cocirc);

   float sigma_kurve = 1.0; // fraction of delta_radius_curvature
   sigma_kurve = params->value(name, "sigmaKurve", sigma_kurve);

   // sigma_chord = % of PI * R, where R == radius of curvature (1/curvature)
   float sigma_chord = 0.5;
   sigma_chord = params->value(name, "sigmaChord", sigma_chord);

   float delta_theta_max = PI / 2.0;
   delta_theta_max = params->value(name, "deltaThetaMax", delta_theta_max);

   float cocirc_self = (pre != post);
   cocirc_self = params->value(name, "cocircSelf", cocirc_self);

   // from pv_common.h
   // // DK (1.0/(6*(NK-1)))   /*1/(sqrt(DX*DX+DY*DY)*(NK-1))*/         //  change in curvature
   float delta_radius_curvature = 1.0; // 1 = minimum radius of curvature
   delta_radius_curvature = params->value(name, "deltaRadiusCurvature",
         delta_radius_curvature);

   for (int patchIndex = 0; patchIndex < numPatches; patchIndex++) {
      cocircCalcWeights(patches[patchIndex], patchIndex, noPre, noPost, sigma_cocirc,
            sigma_kurve, sigma_chord, delta_theta_max, cocirc_self,
            delta_radius_curvature, numFlanks, shift, aspect, rotate, sigma, r2Max,
            strength);
   }

   return patches;
}


PVPatch ** HyPerConn::readWeights(PVPatch ** patches, int numPatches, const char * filename)
{
   double time;
   int status = PV::readWeights(patches, numPatches, filename, parent->icCommunicator(),
                                &time, pre->getLayerLoc(), true);

   if (status != 0) {
      fprintf(stderr, "PV::HyPerConn::readWeights: problem reading weight file %s, SHUTTING DOWN\n", filename);
      exit(1);
   }

   return patches;
}
#endif

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

int HyPerConn::writeWeights(PVPatch ** patches, int numPatches,
                            const char * filename, float time, bool last, int arborId)
{
   int status = 0;
   char path[PV_PATH_MAX];

   if (patches == NULL) return 0;

   const float minVal = minWeight();
   const float maxVal = maxWeight();

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
                             patches, numPatches, writeCompressedWeights);
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
#endif

   return status;
}

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
      pv_text_write_patch(fd, wPatches[arbor][k]);
      fprintf(fd, "----------------------------\n");
   }

   if (fd != stdout) {
      fclose(fd);
   }

   return 0;
}

int HyPerConn::deliver(Publisher * pub, const PVLayerCube * cube, int neighbor)
{
#ifdef DEBUG_OUTPUT
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   printf("[%d]: HyPerConn::deliver: neighbor=%d cube=%p post=%p this=%p\n", rank, neighbor, cube, post, this);
   fflush(stdout);
#endif
   for(int arborId=0;arborId<numberOfAxonalArborLists();arborId++) {
      int delay = getDelay(arborId);
      pub->readData(delay);
//      if (delay > 0) {
//         cube->data = pre->getLayerData(delay);
//      }
//      else {
//         cube->data = pre->getLayerData();
//      }
      int status = post->recvSynapticInput(this, cube, arborId);
      if (status == PV_CONTINUE) continue;
      assert(status == PV_SUCCESS);
   }
#ifdef DEBUG_OUTPUT
   printf("[%d]: HyPerConn::delivered: \n", rank);
   fflush(stdout);
#endif
   return 0;
}

int HyPerConn::insertProbe(ConnectionProbe * p)
{
   ConnectionProbe ** tmp;
   tmp = (ConnectionProbe **) malloc((numProbes + 1) * sizeof(ConnectionProbe *));
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

   for (int i = 0; i < numProbes; i++) {
      probes[i]->outputState(time, this);
   }

   if (last) {
      status = writeWeights(time, last);
      assert(status == 0);

#ifdef OBSOLETE_STDP
      if (stdpFlag) {
         convertPreSynapticWeights(time);
         status = writePostSynapticWeights(time, last);
         assert(status == 0);
      }
#endif

   }
   else if ( (time >= writeTime) && (writeStep >= 0) ) {
      writeTime += writeStep;

      status = writeWeights(time, last);
      assert(status == 0);

#ifdef OBSOLETE_STDP
      if ( stdpFlag ) {
         convertPreSynapticWeights(time);
         status = writePostSynapticWeights(time, last);
         assert(status == 0);
      }
#endif

      // append to output file after original open
      ioAppend = true;
   }

   return status;
}

int HyPerConn::updateState(float time, float dt)
{
   update_timer->start();

#ifdef OBSOLETE_STDP

   if (stdpFlag) {
      const float fac = ampLTD;
      const float decay = expf(-dt / tauLTD);

      //
      // both pDecr and activity are extended regions (plus margins)
      // to make processing them together simpler

      const int nk = pDecr->numItems;
      const float * a = post->getLayerData();
      float * m = pDecr->data; // decrement (minus) variable

      for (int k = 0; k < nk; k++) {
         m[k] = decay * m[k] - fac * a[k];
      }

      const int axonId = 0;       // assume only one for now
      updateWeights(axonId);
   }
#endif

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
   return 0;
}

//
/* M (m or pDecr->data) is an extended post-layer variable
 *
 */
int HyPerConn::updateWeights(int axonId)
{
#ifdef OBSOLETE_STDP
   // Steps:
   // 1. Update pDecr (assume already done as it should only be done once)
   // 2. update Psij (pIncr) for each synapse
   // 3. update wij

   const float dt = parent->getDeltaTime();
   const float decayLTP = expf(-dt / tauLTP);

   const int numExtended = pre->getNumExtended();
   assert(numExtended == numWeightPatches(axonId));

   const pvdata_t * preLayerData = pre->getLayerData();

   // this stride is in extended space for post-synaptic activity and
   // STDP decrement variable
   const int postStrideY = post->getLayerLoc()->nf
                         * (post->getLayerLoc()->nx + 2 * post->getLayerLoc()->nb);

   for (int kPre = 0; kPre < numExtended; kPre++) {
      PVAxonalArbor * arbor = axonalArbor(kPre, axonId);

      const float preActivity = preLayerData[kPre];

      PVPatch * pIncr   = arbor->plasticIncr;
      PVPatch * w       = arbor->weights;
      size_t postOffset = arbor->offset;

      const float * postActivity = &post->getLayerData()[postOffset];
      const float * M = &pDecr->data[postOffset];  // STDP decrement variable
      float * P = pIncr->data;                     // STDP increment variable
      float * W = w->data;

      int nk  = pIncr->nf * pIncr->nx; // one line in x at a time
      int ny  = pIncr->ny;
      int sy  = pIncr->sy;

      // TODO - unroll

      // update Psij (pIncr variable)
      // we are processing patches, one line in y at a time
      for (int y = 0; y < ny; y++) {
         pvpatch_update_plasticity_incr(nk, P + y * sy, preActivity, decayLTP, ampLTP);
      }

      if (localWmaxFlag) {
         // update weights with local post-synaptic Wmax values
         // Wmax lives in the restricted space - it is controlled
         // by average rate in the post synaptic layer
         float * Wmax_pointer = &Wmax[postOffset];
         for (int y = 0; y < ny; y++) {
            // TODO
            pvpatch_update_weights_localWMax(nk,W,M,P,preActivity,postActivity,dWMax,wMin,Wmax_pointer);
            //
            // advance pointers in y
            W += sy;
            P += sy;

            //
            // postActivity, M are extended post-layer variables
            //
            postActivity += postStrideY;
            M            += postStrideY;
            Wmax_pointer += postStrideY;
         }
      } else {
         // update weights
         for (int y = 0; y < ny; y++) {
            pvpatch_update_weights(nk, W, M, P, preActivity, postActivity, dWMax, wMin, wMax);
            //
            // advance pointers in y
            W += sy;
            P += sy;
            //
            // postActivity and M are extended layer
            postActivity += postStrideY;
            M += postStrideY;
         }

      }
   }
#endif

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

#ifdef OBSOLETE_STDP
PVPatch * HyPerConn::getPlasticityIncrement(int k, int arbor)
{
   // a separate arbor/patch of plasticity for every neuron
   if (stdpFlag) {
      return pIncr[k];
   }
   return NULL;
}
#endif

PVPatch ** HyPerConn::createWeights(PVPatch ** patches, int nPatches, int nxPatch,
      int nyPatch, int nfPatch, int axonId)
{
   // could create only a single patch with following call
   //   return createPatches(numAxonalArborLists, nxp, nyp, nfp);

   //assert(numAxonalArborLists == 1);
   assert(patches == NULL);

   patches = (PVPatch**) calloc(sizeof(PVPatch*), nPatches);
   assert(patches != NULL);

   // TODO - allocate space for them all at once (inplace)
   allocWeights(patches, nPatches, nxPatch, nyPatch, nfPatch, axonId);

   return patches;
}

/**
 * Create a separate patch of weights for every neuron
 */
PVPatch ** HyPerConn::createWeights(PVPatch ** patches, int axonId)
{
   //const int arbor = 0;
   int nPatches = numWeightPatches();
   int nxPatch = nxp;
   int nyPatch = nyp;
   int nfPatch = nfp;

   return createWeights(patches, nPatches, nxPatch, nyPatch, nfPatch, axonId);
}

int HyPerConn::deleteWeights()
{
   // to be used if createPatches is used above
   // HyPerConn::deletePatches(numAxonalArborLists, wPatches);

   for (int arbor = 0; arbor < numAxonalArborLists; arbor++) {
      int numPatches = numWeightPatches();
      if (wPatches[arbor] != NULL) {
         for (int k = 0; k < numPatches; k++) {
            pvpatch_inplace_delete(wPatches[arbor][k]);
         }
         free(wPatches[arbor]);
         wPatches[arbor] = NULL;
      }
   }

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

#ifdef OBSOLETE_STDP
   if (stdpFlag) {
      const int arbor = 0;
      int numPatches = numWeightPatches(arbor);
      for (int k = 0; k < numPatches; k++) {
         pvpatch_inplace_delete(pIncr[k]);
      }
      free(pIncr);
      pvcube_delete(pDecr);
   }
#endif

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
   PVParams * inputParams = parent->parameters();
#ifdef OBSOLETE
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

#endif
   //const int numAxons = numAxonalArborLists;

   // these strides are for post-synaptic phi variable, a non-extended layer variable
   //
   const int psf = 1;
   const int psx = nfp;
   const int psy = psx * post->getLayerLoc()->nx;

   // activity and STDP M variable are extended into margins
   //
   //for (int n = 0; n < numAxons; n++) {
   int numPatches = numWeightPatches();
   PVAxonalArbor* newArbor = (PVAxonalArbor*) calloc(numPatches, sizeof(PVAxonalArbor));
   assert(newArbor != NULL);
   setArbor(newArbor, arborId);
   assert(axonalArborList[arborId] != NULL);
   //}

   //for (int n = 0; n < numAxons; n++) {
   //int numArbors = numWeightPatches();
   PVPatch * dataPatches = (PVPatch *) calloc(numPatches, sizeof(PVPatch));
   assert(dataPatches != NULL);

   for (int kex = 0; kex < numPatches; kex++) {
      PVAxonalArbor * arbor = axonalArbor(kex, arborId);

      // kex is in extended frame, this makes transformations more difficult

#ifdef OBSOLETE
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
#else
      int kl, offset, nxPatch, nyPatch, dx, dy;
#endif
      calcPatchSize(arborId, kex, &kl, &offset, &nxPatch, &nyPatch, &dx, &dy);

      arbor->data = &dataPatches[kex];
      arbor->weights = getWeights(kex, arborId);
      arbor->plasticIncr = NULL;   // set later by initPlasticityPatches

      arbor->delay=(int) inputParams->value(name, "delay", 0);
      // initialize the receiving (of spiking data) gSyn variable
      pvdata_t * gSyn = post->getChannel(channel) + kl;
      pvpatch_init(arbor->data, nxPatch, nyPatch, nfp, psx, psy, psf, gSyn);

#ifdef OBSOLETE
      // get offset in extended frame for post-synaptic M STDP variable
      //
      kxPost += postPad;
      kyPost += postPad;

      kl = kIndex(kxPost, kyPost, kfPost, nxexPost, nyexPost, nfPost);
      assert(kl >= 0);
      assert(kl < lPost->numExtended);
#endif
      arbor->offset = offset;

      // adjust patch size (shrink) to fit within interior of post-synaptic layer
      //
      pvpatch_adjust(arbor->weights, nxPatch, nyPatch, dx, dy);

   } // loop over arbors (pre-synaptic neurons)
   //} // loop over neighbors

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
   const float powXScale = powf(2.0f, (float) xScale);
   const float powYScale = powf(2.0f, (float) yScale);

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
         wPostPatches[axonID] = createWeights(NULL, numPost, nxPostPatch, nyPostPatch, nfPostPatch, 0);
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

               const int nfp = p->nf;

               // get strides for possibly shrunken patch
               const int sxp = p->sx;
               const int syp = p->sy;
               const int sfp = p->sf;

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
   const float powXScale = powf(2, (float) xScale);
   const float powYScale = powf(2, (float) yScale);

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

int HyPerConn::writePostSynapticWeights(float time, bool last) {
   for(int axonID=0;axonID<numberOfAxonalArborLists();axonID++) {
      writePostSynapticWeights(time, last, axonID);
   }

   return 0;
}

int HyPerConn::writePostSynapticWeights(float time, bool last, int axonID)
{
   int status = 0;
   char path[PV_PATH_MAX];

   const PVLayer * lPre  = pre->getCLayer();
   const PVLayer * lPost = post->getCLayer();

   const float minVal = minWeight();
   const float maxVal = maxWeight();

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
                             wPostPatches[axonID], numPostPatches,writeCompressedWeights);
   assert(status == 0);

   return 0;
}

#ifdef OBSOLETE //The following methods have been added to the new InitWeights classes.  Please
                //use the param "weightInitType" to choose an initialization type
/**
 * generate random weights for a patch from a uniform distribution
 * NOTES:
 *    - the pointer w already points to the patch head in the data structure
 *    - it only sets the weights to "real" neurons, not to neurons in the boundary
 *    layer. For example, if x are boundary neurons and o are real neurons,
 *    x x x x
 *    x o o o
 *    x o o o
 *    x o o o
 *
 *    for a 4x4 connection it sets the weights to the o neurons only.
 *    .
 */
int HyPerConn::uniformWeights(PVPatch * wp, float minwgt, float maxwgt)
      // changed variable names to avoid confusion with data members this->wMin and this->wMax
{
   pvdata_t * w = wp->data;

   const int nxp = wp->nx;
   const int nyp = wp->ny;
   const int nfp = wp->nf;

   const int sxp = wp->sx;
   const int syp = wp->sy;
   const int sfp = wp->sf;

   double p;
   if( maxwgt <= minwgt ) {
      if( maxwgt < minwgt ) {
         fprintf(stderr, "Warning: uniformWeights maximum less than minimum.  Changing max = %f to min value of %f\n", maxwgt, minwgt);
         maxwgt = minwgt;
      }
      p = 0;
   }
   else {
       p = (maxwgt - minwgt) / pv_random_max();
   }

   // loop over all post-synaptic cells in patch
   for (int y = 0; y < nyp; y++) {
      for (int x = 0; x < nxp; x++) {
         for (int f = 0; f < nfp; f++) {
            w[x * sxp + y * syp + f * sfp] = minwgt + p * pv_random();
         }
      }
   }

   return PV_SUCCESS;
}


/**
 * generate random weights for a patch from a gaussian distribution
 * NOTES:
 *    - the pointer w already points to the patch head in the data structure
 *    - it only sets the weights to "real" neurons, not to neurons in the boundary
 *    layer. For example, if x are boundary neurons and o are real neurons,
 *    x x x x
 *    x o o o
 *    x o o o
 *    x o o o
 *
 *    for a 4x4 connection it sets the weights to the o neurons only.
 *    .
 */
// int HyPerConn::gaussianWeights(PVPatch * wp, float mean, float stdev, int * seed)
int HyPerConn::gaussianWeights(PVPatch * wp, float mean, float stdev)
{
   pvdata_t * w = wp->data;

   const int nxp = wp->nx;
   const int nyp = wp->ny;
   const int nfp = wp->nf;

   const int sxp = wp->sx;
   const int syp = wp->sy;
   const int sfp = wp->sf;

   // loop over all post-synaptic cells in patch
   for (int y = 0; y < nyp; y++) {
      for (int x = 0; x < nxp; x++) {
         for (int f = 0; f < nfp; f++) {
            w[x * sxp + y * syp + f * sfp] = box_muller(mean,stdev);
         }
      }
   }

   return 0;
}

int HyPerConn::smartWeights(PVPatch * wp, int k)
{
   pvdata_t * w = wp->data;

   const int nxp = wp->nx;
   const int nyp = wp->ny;
   const int nfp = wp->nf;

   const int sxp = wp->sx;
   const int syp = wp->sy;
   const int sfp = wp->sf;

   // loop over all post-synaptic cells in patch
   for (int y = 0; y < nyp; y++) {
      for (int x = 0; x < nxp; x++) {
         for (int f = 0; f < nfp; f++) {
            w[x * sxp + y * syp + f * sfp] = k;
         }
      }
   }

   return 0;
}

/**
 * calculate gaussian weights to segment lines
 */
int HyPerConn::gauss2DCalcWeights(PVPatch * wp, int kPre, int no, int numFlanks,
      float shift, float rotate, float aspect, float sigma, float r2Max, float strength,
      float deltaThetaMax, float thetaMax, float bowtieFlag, float bowtieAngle)
{
//   const PVLayer * lPre = pre->clayer;
//   const PVLayer * lPost = post->clayer;

   bool self = (pre != post);

   // get dimensions of (potentially shrunken patch)
   const int nxPatch = wp->nx;
   const int nyPatch = wp->ny;
   const int nfPatch = wp->nf;
   if (nxPatch * nyPatch * nfPatch == 0) {
      return 0; // reduced patch size is zero
   }

   pvdata_t * w = wp->data;

   // get strides of (potentially shrunken) patch
   const int sx = wp->sx;
   assert(sx == nfPatch);
   const int sy = wp->sy; // no assert here because patch may be shrunken
   const int sf = wp->sf;
   assert(sf == 1);

   // make full sized temporary patch, positioned around center of unit cell
   PVPatch * wp_tmp;
   wp_tmp = pvpatch_inplace_new(nxp, nyp, nfp);
   pvdata_t * w_tmp = wp_tmp->data;

   // get/check dimensions and strides of full sized temporary patch
   const int nxPatch_tmp = wp_tmp->nx;
   const int nyPatch_tmp = wp_tmp->ny;
   const int nfPatch_tmp = wp_tmp->nf;
   int kxKernelIndex;
   int kyKernelIndex;
   int kfKernelIndex;
   this->patchIndexToKernelIndex(kPre, &kxKernelIndex, &kyKernelIndex, &kfKernelIndex);

   const int kxPre_tmp = kxKernelIndex;
   const int kyPre_tmp = kyKernelIndex;
   const int kfPre_tmp = kfKernelIndex;
   const int sx_tmp = wp_tmp->sx;
   assert(sx_tmp == wp_tmp->nf);
   const int sy_tmp = wp_tmp->sy;
   assert(sy_tmp == wp_tmp->nf * wp_tmp->nx);
   const int sf_tmp = wp_tmp->sf;
   assert(sf_tmp == 1);

   // get distances to nearest neighbor in post synaptic layer (measured relative to pre-synaptic cell)
   float xDistNNPreUnits;
   float xDistNNPostUnits;
   dist2NearestCell(kxPre_tmp, pre->getXScale(), post->getXScale(),
         &xDistNNPreUnits, &xDistNNPostUnits);
   float yDistNNPreUnits;
   float yDistNNPostUnits;
   dist2NearestCell(kyPre_tmp, pre->getYScale(), post->getYScale(),
         &yDistNNPreUnits, &yDistNNPostUnits);

   // get indices of nearest neighbor
   int kxNN;
   int kyNN;
   kxNN = nearby_neighbor( kxPre_tmp, pre->getXScale(), post->getXScale());
   kyNN = nearby_neighbor( kyPre_tmp, pre->getYScale(), post->getYScale());

   // get indices of patch head
   int kxHead;
   int kyHead;
   kxHead = zPatchHead(kxPre_tmp, nxPatch_tmp, pre->getXScale(), post->getXScale());
   kyHead = zPatchHead(kyPre_tmp, nyPatch_tmp, pre->getYScale(), post->getYScale());

   // get distance to patch head (measured relative to pre-synaptic cell)
   float xDistHeadPostUnits;
   xDistHeadPostUnits = xDistNNPostUnits + (kxHead - kxNN);
   float yDistHeadPostUnits;
   yDistHeadPostUnits = yDistNNPostUnits + (kyHead - kyNN);
   float xRelativeScale = xDistNNPreUnits == xDistNNPostUnits ? 1.0f : xDistNNPreUnits
         / xDistNNPostUnits;
   float xDistHeadPreUnits;
   xDistHeadPreUnits = xDistHeadPostUnits * xRelativeScale;
   float yRelativeScale = yDistNNPreUnits == yDistNNPostUnits ? 1.0f : yDistNNPreUnits
         / yDistNNPostUnits;
   float yDistHeadPreUnits;
   yDistHeadPreUnits = yDistHeadPostUnits * yRelativeScale;


   // sigma is in units of pre-synaptic layer
   const float dxPost = xRelativeScale; //powf(2, (float) post->getXScale());
   const float dyPost = yRelativeScale; //powf(2, (float) post->getYScale());


   // TODO - the following assumes that if aspect > 1, # orientations = # features
   //   int noPost = no;
   // number of orientations only used if aspect != 1
   const int noPost = post->getLayerLoc()->nf;
   const float dthPost = PI*thetaMax / (float) noPost;
   const float th0Post = rotate * dthPost / 2.0f;
   const int noPre = pre->getLayerLoc()->nf;
   const float dthPre = PI*thetaMax / (float) noPre;
   const float th0Pre = rotate * dthPre / 2.0f;
   const int fPre = kPre % pre->getLayerLoc()->nf;
   assert(fPre == kfPre_tmp);
   const int iThPre = kPre % noPre;
   const float thPre = th0Pre + iThPre * dthPre;

   // loop over all post-synaptic cells in temporary patch
   for (int fPost = 0; fPost < nfPatch_tmp; fPost++) {
      int oPost = fPost % noPost;
      float thPost = th0Post + oPost * dthPost;
      if (noPost == 1 && noPre > 1) {
         thPost = thPre;
      }
      //TODO: add additional weight factor for difference between thPre and thPost
      if (fabs(thPre - thPost) > deltaThetaMax) {
         continue;
      }
      for (int jPost = 0; jPost < nyPatch_tmp; jPost++) {
         float yDelta = (yDistHeadPreUnits + jPost * dyPost);
         for (int iPost = 0; iPost < nxPatch_tmp; iPost++) {
            float xDelta = (xDistHeadPreUnits + iPost * dxPost);
            bool sameLoc = ((fPre == fPost) && (xDelta == 0.0f) && (yDelta == 0.0f));
            if ((sameLoc) && (!self)) {
               continue;
            }

            // rotate the reference frame by th (change sign of thPost?)
            float xp = +xDelta * cosf(thPost) + yDelta * sinf(thPost);
            float yp = -xDelta * sinf(thPost) + yDelta * cosf(thPost);
//            float xp = xDelta * cosf(thPost) - yDelta * sinf(thPost);
//            float yp = xDelta * sinf(thPost) + yDelta * cosf(thPost);

            if (bowtieFlag == 1.0f){
               float offaxis_angle = atan2(yp, xp);
               if ( ((offaxis_angle > bowtieAngle) && (offaxis_angle < (PI - bowtieAngle))) ||
                     ((offaxis_angle < -bowtieAngle) && (offaxis_angle > (-PI + bowtieAngle))) ){
                  continue;
               }
            }

            // include shift to flanks
            float d2 = xp * xp + (aspect * (yp - shift) * aspect * (yp - shift));
            w_tmp[iPost * sx_tmp + jPost * sy_tmp + fPost * sf_tmp] = 0;
            if (d2 <= r2Max) {
               w_tmp[iPost * sx_tmp + jPost * sy_tmp + fPost * sf_tmp] += expf(-d2
                     / (2.0f * sigma * sigma));
            }
            if (numFlanks > 1) {
               // shift in opposite direction
               d2 = xp * xp + (aspect * (yp + shift) * aspect * (yp + shift));
               if (d2 <= r2Max) {
                  w_tmp[iPost * sx_tmp + jPost * sy_tmp + fPost * sf_tmp] += expf(-d2
                        / (2.0f * sigma * sigma));
               }
            }
         }
      }
   }

   // copy weights from full sized temporary patch to (possibly shrunken) patch
   w = wp->data;
   pvdata_t * data_head =  (pvdata_t *) ((char*) wp + sizeof(PVPatch));
   size_t data_offset = w - data_head;
   w_tmp = &wp_tmp->data[data_offset];
   int nk = nxPatch * nfPatch;
   for (int ky = 0; ky < nyPatch; ky++) {
      for (int iWeight = 0; iWeight < nk; iWeight++) {
         w[iWeight] = w_tmp[iWeight];
      }
      w += sy;
      w_tmp += sy_tmp;
   }

   free(wp_tmp);
   return 0;
}

int HyPerConn::cocircCalcWeights(PVPatch * wp, int kPre, int noPre, int noPost,
      float sigma_cocirc, float sigma_kurve, float sigma_chord, float delta_theta_max,
      float cocirc_self, float delta_radius_curvature, int numFlanks, float shift,
      float aspect, float rotate, float sigma, float r2Max, float strength)
{
   pvdata_t * w = wp->data;

   const float min_weight = 0.0f; // read in as param
   const float sigma2 = 2 * sigma * sigma;
   const float sigma_cocirc2 = 2 * sigma_cocirc * sigma_cocirc;

   const int nxPatch = (int) wp->nx;
   const int nyPatch = (int) wp->ny;
   const int nfPatch = (int) wp->nf;
   if (nxPatch * nyPatch * nfPatch == 0) {
      return 0; // reduced patch size is zero
   }

   // get strides of (potentially shrunken) patch
   const int sx = (int) wp->sx;
   assert(sx == nfPatch);
   const int sy = (int) wp->sy; // no assert here because patch may be shrunken
   const int sf = (int) wp->sf;
   assert(sf == 1);

   // make full sized temporary patch, positioned around center of unit cell
   PVPatch * wp_tmp;
   wp_tmp = pvpatch_inplace_new(nxp, nyp, nfp);
   pvdata_t * w_tmp = wp_tmp->data;

   // get/check dimensions and strides of full sized temporary patch
   const int nxPatch_tmp = wp_tmp->nx;
   const int nyPatch_tmp = wp_tmp->ny;
   const int nfPatch_tmp = wp_tmp->nf;
   int kxKernelIndex;
   int kyKerneIndex;
   int kfKernelIndex;
   this->patchIndexToKernelIndex(kPre, &kxKernelIndex, &kyKerneIndex, &kfKernelIndex);

   const int kxPre_tmp = kxKernelIndex;
   const int kyPre_tmp = kyKerneIndex;
   //   const int kfPre_tmp = kfKernelIndex;
   const int sx_tmp = wp_tmp->sx;
   assert(sx_tmp == wp_tmp->nf);
   const int sy_tmp = wp_tmp->sy;
   assert(sy_tmp == wp_tmp->nf * wp_tmp->nx);
   const int sf_tmp = wp_tmp->sf;
   assert(sf_tmp == 1);

   // get distances to nearest neighbor in post synaptic layer
   float xDistNNPreUnits;
   float xDistNNPostUnits;
   dist2NearestCell(kxPre_tmp, pre->getXScale(), post->getXScale(), &xDistNNPreUnits,
         &xDistNNPostUnits);
   float yDistNNPreUnits;
   float yDistNNPostUnits;
   dist2NearestCell(kyPre_tmp, pre->getYScale(), post->getYScale(), &yDistNNPreUnits,
         &yDistNNPostUnits);

   // get indices of nearest neighbor
   int kxNN;
   int kyNN;
   kxNN = nearby_neighbor(kxPre_tmp, pre->getXScale(), post->getXScale());
   kyNN = nearby_neighbor(kyPre_tmp, pre->getYScale(), post->getYScale());

   // get indices of patch head
   int kxHead;
   int kyHead;
   kxHead = zPatchHead(kxPre_tmp, nxPatch_tmp, pre->getXScale(), post->getXScale());
   kyHead = zPatchHead(kyPre_tmp, nyPatch_tmp, pre->getYScale(), post->getYScale());

   // get distance to patch head
   float xDistHeadPostUnits;
   xDistHeadPostUnits = xDistNNPostUnits + (kxHead - kxNN);
   float yDistHeadPostUnits;
   yDistHeadPostUnits = yDistNNPostUnits + (kyHead - kyNN);
   float xRelativeScale = xDistNNPreUnits == xDistNNPostUnits ? 1.0f : xDistNNPreUnits
         / xDistNNPostUnits;
   float xDistHeadPreUnits;
   xDistHeadPreUnits = xDistHeadPostUnits * xRelativeScale;
   float yRelativeScale = yDistNNPreUnits == yDistNNPostUnits ? 1.0f : yDistNNPreUnits
         / yDistNNPostUnits;
   float yDistHeadPreUnits;
   yDistHeadPreUnits = yDistHeadPostUnits * yRelativeScale;

   // sigma is in units of pre-synaptic layer
   const float dxPost = powf(2, post->getXScale());
   const float dyPost = powf(2, post->getYScale());

   //const int kfPre = kPre % pre->clayer->loc.nf;
   const int kfPre = featureIndex(kPre, pre->getLayerLoc()->nx, pre->getLayerLoc()->ny,
         pre->getLayerLoc()->nf);

   bool POS_KURVE_FLAG = false; //  handle pos and neg curvature separately
   bool SADDLE_FLAG  = false; // handle saddle points separately
   const int nKurvePre = pre->getLayerLoc()->nf / noPre;
   const int nKurvePost = post->getLayerLoc()->nf / noPost;
   const float dThPre = PI / noPre;
   const float dThPost = PI / noPost;
   const float th0Pre = rotate * dThPre / 2.0;
   const float th0Post = rotate * dThPost / 2.0;
   const int iThPre = kfPre / nKurvePre;
   const float thetaPre = th0Pre + iThPre * dThPre;

   int iKvPre = kfPre % nKurvePre;
   bool iPosKurvePre = false;
   bool iSaddlePre = false;
   float radKurvPre = delta_radius_curvature + iKvPre * delta_radius_curvature;
   float kurvePre = (radKurvPre != 0.0f) ? 1 / radKurvPre : 1.0f;
   int iKvPreAdj = iKvPre;
   if (POS_KURVE_FLAG) {
      assert(nKurvePre >= 2);
      iPosKurvePre = iKvPre >= (int) (nKurvePre / 2);
      if (SADDLE_FLAG) {
         assert(nKurvePre >= 4);
         iSaddlePre = (iKvPre % 2 == 0) ? 0 : 1;
         iKvPreAdj = ((iKvPre % (nKurvePre / 2)) / 2);}
      else { // SADDLE_FLAG
         iKvPreAdj = (iKvPre % (nKurvePre/2));}
   } // POS_KURVE_FLAG
   radKurvPre = delta_radius_curvature + iKvPreAdj * delta_radius_curvature;
   kurvePre = (radKurvPre != 0.0f) ? 1 / radKurvPre : 1.0f;
   float sigma_kurve_pre = sigma_kurve * radKurvPre;
   float sigma_kurve_pre2 = 2 * sigma_kurve_pre * sigma_kurve_pre;
   sigma_chord *= PI * radKurvPre;
   float sigma_chord2 = 2.0 * sigma_chord * sigma_chord;

   // loop over all post synaptic neurons in patch
   for (int kfPost = 0; kfPost < nfPatch_tmp; kfPost++) {
      int iThPost = kfPost / nKurvePost;
      float thetaPost = th0Post + iThPost * dThPost;

      int iKvPost = kfPost % nKurvePost;
      bool iPosKurvePost = false;
      bool iSaddlePost = false;
      float radKurvPost = delta_radius_curvature + iKvPost * delta_radius_curvature;
      float kurvePost = (radKurvPost != 0.0f) ? 1 / radKurvPost : 1.0f;
      int iKvPostAdj = iKvPost;
      if (POS_KURVE_FLAG) {
         assert(nKurvePost >= 2);
         iPosKurvePost = iKvPost >= (int) (nKurvePost / 2);
         if (SADDLE_FLAG) {
            assert(nKurvePost >= 4);
            iSaddlePost = (iKvPost % 2 == 0) ? 0 : 1;
            iKvPostAdj = ((iKvPost % (nKurvePost / 2)) / 2);
         }
         else { // SADDLE_FLAG
            iKvPostAdj = (iKvPost % (nKurvePost / 2));
         }
      } // POS_KURVE_FLAG
      radKurvPost = delta_radius_curvature + iKvPostAdj * delta_radius_curvature;
      kurvePost = (radKurvPost != 0.0f) ? 1 / radKurvPost : 1.0f;
      float sigma_kurve_post = sigma_kurve * radKurvPost;
      float sigma_kurve_post2 = 2 * sigma_kurve_post * sigma_kurve_post;

      float deltaTheta = fabsf(thetaPre - thetaPost);
      deltaTheta = (deltaTheta <= PI / 2.0) ? deltaTheta : PI - deltaTheta;
      if (deltaTheta > delta_theta_max) {
         continue;
      }

      for (int jPost = 0; jPost < nyPatch_tmp; jPost++) {
         float yDelta = (yDistHeadPreUnits + jPost * dyPost);
         for (int iPost = 0; iPost < nxPatch_tmp; iPost++) {
            float xDelta = (xDistHeadPreUnits + iPost * dxPost);

            float gDist = 0.0;
            float gChord = 1.0;
            float gCocirc = 1.0;
            float gKurvePre = 1.0;
            float gKurvePost = 1.0;

            // rotate the reference frame by th
            float dxP = +xDelta * cosf(thetaPre) + yDelta * sinf(thetaPre);
            float dyP = -xDelta * sinf(thetaPre) + yDelta * cosf(thetaPre);

            // include shift to flanks
            float dyP_shift = dyP - shift;
            float dyP_shift2 = dyP + shift;
            float d2 = dxP * dxP + aspect * dyP * aspect * dyP;
            float d2_shift = dxP * dxP + (aspect * (dyP_shift) * aspect * (dyP_shift));
            float d2_shift2 = dxP * dxP + (aspect * (dyP_shift2) * aspect * (dyP_shift2));
            if (d2_shift <= r2Max) {
               gDist += expf(-d2_shift / sigma2);
            }
            if (numFlanks > 1) {
               // include shift in opposite direction
               if (d2_shift2 <= r2Max) {
                  gDist += expf(-d2_shift2 / sigma2);
               }
            }
            if (gDist == 0.0) continue;
            if (d2 == 0) {
               bool sameLoc = (kfPre == kfPost);
               if ((!sameLoc) || (cocirc_self)) {
                  gCocirc = sigma_cocirc > 0 ? expf(-deltaTheta * deltaTheta
                        / sigma_cocirc2) : expf(-deltaTheta * deltaTheta / sigma_cocirc2)
                        - 1.0;
                  if ((nKurvePre > 1) && (nKurvePost > 1)) {
                     gKurvePre = expf(-(kurvePre - kurvePost) * (kurvePre - kurvePost)
                           / 2 * (sigma_kurve_pre * sigma_kurve_pre + sigma_kurve_post
                           * sigma_kurve_post));
                  }
               }
               else { // sameLoc && !cocircSelf
                  gCocirc = 0.0;
                  continue;
               }
            }
            else { // d2 > 0

               float atanx2_shift = thetaPre + 2. * atan2f(dyP_shift, dxP); // preferred angle (rad)
               atanx2_shift += 2. * PI;
               atanx2_shift = fmodf(atanx2_shift, PI);
               atanx2_shift = fabsf(atanx2_shift - thetaPost);
               float chi_shift = atanx2_shift; //fabsf(atanx2_shift - thetaPost); // radians
               if (chi_shift >= PI / 2.0) {
                  chi_shift = PI - chi_shift;
               }
               if (noPre > 1 && noPost > 1) {
                  gCocirc = sigma_cocirc2 > 0 ? expf(-chi_shift * chi_shift
                        / sigma_cocirc2) : expf(-chi_shift * chi_shift / sigma_cocirc2)
                        - 1.0;
               }

               // compute curvature of cocircular contour
               float cocircKurve_shift = d2_shift > 0 ? fabsf(2 * dyP_shift) / d2_shift
                     : 0.0f;
               if (POS_KURVE_FLAG) {
                  if (SADDLE_FLAG) {
                     if ((iPosKurvePre) && !(iSaddlePre) && (dyP_shift < 0)) {
                        continue;
                     }
                     if (!(iPosKurvePre) && !(iSaddlePre) && (dyP_shift > 0)) {
                        continue;
                     }
                     if ((iPosKurvePre) && (iSaddlePre)
                           && (((dyP_shift > 0) && (dxP < 0)) || ((dyP_shift > 0) && (dxP
                                 < 0)))) {
                        continue;
                     }
                     if (!(iPosKurvePre) && (iSaddlePre) && (((dyP_shift > 0)
                           && (dxP > 0)) || ((dyP_shift < 0) && (dxP < 0)))) {
                        continue;
                     }
                  }
                  else { //SADDLE_FLAG
                     if ((iPosKurvePre) && (dyP_shift < 0)) {
                        continue;
                     }
                     if (!(iPosKurvePre) && (dyP_shift > 0)) {
                        continue;
                     }
                  }
               } // POS_KURVE_FLAG
               gKurvePre = (nKurvePre > 1) ? expf(-powf((cocircKurve_shift - fabsf(
                     kurvePre)), 2) / sigma_kurve_pre2) : 1.0;
               gKurvePost
                     = ((nKurvePre > 1) && (nKurvePost > 1) && (sigma_cocirc2 > 0)) ? expf(
                           -powf((cocircKurve_shift - fabsf(kurvePost)), 2)
                                 / sigma_kurve_post2)
                           : 1.0;

               // compute distance along contour
               float d_chord_shift = (cocircKurve_shift != 0.0f) ? atanx2_shift
                     / cocircKurve_shift : sqrt(d2_shift);
               gChord = (nKurvePre > 1) ? expf(-powf(d_chord_shift, 2) / sigma_chord2)
                     : 1.0;

               if (numFlanks > 1) {
                  float atanx2_shift2 = thetaPre + 2. * atan2f(dyP_shift2, dxP); // preferred angle (rad)
                  atanx2_shift2 += 2. * PI;
                  atanx2_shift2 = fmodf(atanx2_shift2, PI);
                  atanx2_shift2 = fabsf(atanx2_shift2 - thetaPost);
                  float chi_shift2 = atanx2_shift2; //fabsf(atanx2_shift2 - thetaPost); // radians
                  if (chi_shift2 >= PI / 2.0) {
                     chi_shift2 = PI - chi_shift2;
                  }
                  if (noPre > 1 && noPost > 1) {
                     gCocirc += sigma_cocirc2 > 0 ? expf(-chi_shift2 * chi_shift2
                           / sigma_cocirc2) : expf(-chi_shift2 * chi_shift2
                           / sigma_cocirc2) - 1.0;
                  }

                  float cocircKurve_shift2 = d2_shift2 > 0 ? fabsf(2 * dyP_shift2)
                        / d2_shift2 : 0.0f;
                  if (POS_KURVE_FLAG) {
                     if (SADDLE_FLAG) {
                        if ((iPosKurvePre) && !(iSaddlePre) && (dyP_shift2 < 0)) {
                           continue;
                        }
                        if (!(iPosKurvePre) && !(iSaddlePre) && (dyP_shift2 > 0)) {
                           continue;
                        }
                        if ((iPosKurvePre) && (iSaddlePre) && (((dyP_shift2 > 0) && (dxP
                              < 0)) || ((dyP_shift2 > 0) && (dxP < 0)))) {
                           continue;
                        }
                        if (!(iPosKurvePre) && (iSaddlePre) && (((dyP_shift2 > 0) && (dxP
                              > 0)) || ((dyP_shift2 < 0) && (dxP < 0)))) {
                           continue;
                        }
                     }
                     else { //SADDLE_FLAG
                        if ((iPosKurvePre) && (dyP_shift2 < 0)) {
                           continue;
                        }
                        if (!(iPosKurvePre) && (dyP_shift2 > 0)) {
                           continue;
                        }
                     } // SADDLE_FLAG
                  } // POS_KURVE_FLAG
                  gKurvePre += (nKurvePre > 1) ? expf(-powf((cocircKurve_shift2 - fabsf(
                        kurvePre)), 2) / sigma_kurve_pre2) : 1.0;
                  gKurvePost += ((nKurvePre > 1) && (nKurvePost > 1) && (sigma_cocirc2
                        > 0)) ? expf(-powf((cocircKurve_shift2 - fabsf(kurvePost)), 2)
                        / sigma_kurve_post2) : 1.0;

                  float d_chord_shift2 = cocircKurve_shift2 != 0.0f ? atanx2_shift2
                        / cocircKurve_shift2 : sqrt(d2_shift2);
                  gChord += (nKurvePre > 1) ? expf(-powf(d_chord_shift2, 2) / sigma_chord2)
                        : 1.0;

               }
            }
            float weight_tmp = gDist * gKurvePre * gKurvePost * gCocirc;
            if (weight_tmp < min_weight) continue;
            w_tmp[iPost * sx_tmp + jPost * sy_tmp + kfPost * sf_tmp] = weight_tmp;

         }
      }
   }

   // copy weights from full sized temporary patch to (possibly shrunken) patch
   w = wp->data;
   pvdata_t * data_head = (pvdata_t *) ((char*) wp + sizeof(PVPatch));
   size_t data_offset = w - data_head;
   w_tmp = &wp_tmp->data[data_offset];
   int nk = nxPatch * nfPatch;
   for (int ky = 0; ky < nyPatch; ky++) {
      for (int iWeight = 0; iWeight < nk; iWeight++) {
         w[iWeight] = w_tmp[iWeight];
      }
      w += sy;
      w_tmp += sy_tmp;
   }

   free(wp_tmp);
   return 0;

}
#endif

int HyPerConn::initNormalize() {
   PVParams * params = parent->parameters();
   normalize_flag = params->value(name, "normalize", normalize_flag);
   if( normalize_flag ) {
      normalize_strength = params->value(name, "strength", 1.0f);
      normalize_max = params->value(name, "normalize_max", 0.0f);
      normalize_zero_offset = params->value(name, "normalize_zero_offset", 0.0f);
      normalize_cutoff = params->value(name, "normalize_cutoff", 0.0f) * normalize_strength;
   }
   return PV_SUCCESS;
}

PVPatch ** HyPerConn::normalizeWeights(PVPatch ** patches, int numPatches, int arborId)
{
   this->wMax = 1.0;
   float maxVal = -FLT_MAX;
   for (int k = 0; k < numPatches; k++) {
      PVPatch * wp = patches[k];
      pvdata_t * w = wp->data;
      const int nx = wp->nx;
      const int ny = wp->ny;
      const int nf = wp->nf;
      const int sy = wp->sy;
      const int num_weights = nx * ny * nf;
      float sum = 0;
      float sum2 = 0;
      maxVal = -FLT_MAX;
      for (int ky = 0; ky < ny; ky++) {
         for(int iWeight = 0; iWeight < nf * nx; iWeight++ ){
            sum += w[iWeight];
            sum2 += w[iWeight] * w[iWeight];
            maxVal = ( maxVal > w[iWeight] ) ? maxVal : w[iWeight];
         }
         w += sy;
      }
      float sigma2 = ( sum2 / num_weights ) - ( sum / num_weights ) * ( sum / num_weights );
      float zero_offset = 0.0f;
      if (normalize_zero_offset == 1.0f){
         zero_offset = sum / num_weights;
         sum = 0.0f;
         maxVal -= zero_offset;
      }
      float scale_factor = 1.0f;
      if (normalize_max == 1.0f) {
         scale_factor = normalize_strength / ( fabs(maxVal) + (maxVal == 0.0f) );
      }
       else if (sum != 0.0f) {
         scale_factor = normalize_strength / sum;
      }
       else if (sum == 0.0f && sigma2 > 0.0f) {
         scale_factor = normalize_strength / sqrtf(sigma2);
      }
      w = wp->data;
      for (int ky = 0; ky < ny; ky++) {
         for(int iWeight = 0; iWeight < nf * nx; iWeight++ ){
            w[iWeight] = ( w[iWeight] - zero_offset ) * scale_factor;
            w[iWeight] = ( fabs(w[iWeight]) > fabs(normalize_cutoff) ) ? w[iWeight] : 0.0f;
         }
         w += sy;
      }
   }
   return patches;
}

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

#ifdef OBSOLETE // Marked obsolete Jul 25, 2011.  These variables are no longer used in this function.
   const int numAxons = numAxonalArborLists;

   // these strides are for post-synaptic phi variable, a non-extended layer variable
   //
   const int psf = 1;
   const int psx = nfp;
   const int psy = psx * nxPost;
#endif

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
      fprintf( stderr, "Params file specifies %d features for connection %s,\n", nfp, name );
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

   status = filename ? patchSizeFromFile(filename) : PV_SUCCESS;

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

   status = pvp_read_header(filename, comm, &time, &filetype, &datatype, wgtParams, &numWgtParams);
   if (status < 0) return status;

   status = checkPVPFileHeader(comm, &loc, wgtParams, numWgtParams);
   if (status < 0) return status;

   // reconcile differences with inputParams
   status = checkWeightsHeader(filename, wgtParams);
   return status;
}

int HyPerConn::checkPatchSize(int patchSize, int scalePre, int scalePost, char dim) {
   int scaleDiff = scalePre - scalePost;
   bool goodsize;

   if( scaleDiff > 0) {
      // complain if patchSize is not an odd number times 2^xScaleDiff
      int scaleFactor = (int) powf(2, (float) scaleDiff);
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
         int scaleFactor = (int) powf(2, (float) scaleDiff);
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


PVPatch ** HyPerConn::allocWeights(PVPatch ** patches, int nPatches, int nxPatch,
      int nyPatch, int nfPatch, int axonId)
{
   for (int k = 0; k < nPatches; k++) {
      patches[k] = pvpatch_inplace_new(nxPatch, nyPatch, nfPatch);
   }
   return patches;
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
   int patchIndex;
   // get size of kernel PV cube
   int nxKernel = (pre->getXScale() < post->getXScale()) ? pow(2,
         post->getXScale() - pre->getXScale()) : 1;
   int nyKernel = (pre->getYScale() < post->getYScale()) ? pow(2,
         post->getYScale() - pre->getYScale()) : 1;
   int nfKernel = pre->getLayerLoc()->nf;
   int kxPreExtended = kxPos(kernelIndex, nxKernel, nyKernel, nfKernel) + pre->getLayerLoc()->nb;
   int kyPreExtended = kyPos(kernelIndex, nxKernel, nyKernel, nfKernel) + pre->getLayerLoc()->nb;
   int kfPre = featureIndex(kernelIndex, nxKernel, nyKernel, nfKernel);
   int nxPreExtended = pre->getLayerLoc()->nx + 2*pre->getLayerLoc()->nb;
   int nyPreExtended = pre->getLayerLoc()->ny + 2*pre->getLayerLoc()->nb;
   patchIndex = kIndex(kxPreExtended, kyPreExtended, kfPre, nxPreExtended, nyPreExtended, nfKernel);
   if (kxPatchIndex != NULL){
      *kxPatchIndex = kxPreExtended;
   }
   if (kyPatchIndex != NULL){
      *kyPatchIndex = kyPreExtended;
   }
   if (kfPatchIndex != NULL){
      *kfPatchIndex = kfPre;
   }
   return patchIndex;
}

// many to one mapping from weight patches to kernels
// patchIndex always in extended space
// kernelIndex always for unit cell
int HyPerConn::patchIndexToKernelIndex(int patchIndex, int * kxKernelIndex,
      int * kyKernelIndex, int * kfKernelIndex)
{
   int kernelIndex;
   int nxPreExtended = pre->getLayerLoc()->nx + 2*pre->getLayerLoc()->nb;
   int nyPreExtended = pre->getLayerLoc()->ny + 2*pre->getLayerLoc()->nb;
   int nfPre = pre->getLayerLoc()->nf;
   int kxPreExtended = kxPos(patchIndex, nxPreExtended, nyPreExtended, nfPre);
   int kyPreExtended = kyPos(patchIndex, nxPreExtended, nyPreExtended, nfPre);

   // check that patchIndex lay within margins
   assert(kxPreExtended >= 0);
   assert(kyPreExtended >= 0);
   assert(kxPreExtended < nxPreExtended);
   assert(kyPreExtended < nyPreExtended);

   // convert from extended to restricted space (in local HyPerCol coordinates)
   int kxPreRestricted;
   kxPreRestricted = kxPreExtended - pre->getLayerLoc()->nb;
   while(kxPreRestricted < 0){
      kxPreRestricted += pre->getLayerLoc()->nx;
   }
   while(kxPreRestricted >= pre->getLayerLoc()->nx){
      kxPreRestricted -= pre->getLayerLoc()->nx;
   }

   int kyPreRestricted;
   kyPreRestricted = kyPreExtended - pre->getLayerLoc()->nb;
   while(kyPreRestricted < 0){
      kyPreRestricted += pre->getLayerLoc()->ny;
   }
   while(kyPreRestricted >= pre->getLayerLoc()->ny){
      kyPreRestricted -= pre->getLayerLoc()->ny;
   }

   int kfPre = featureIndex(patchIndex, nxPreExtended, nyPreExtended, nfPre);

   int nxKernel = (pre->getXScale() < post->getXScale()) ? pow(2,
         post->getXScale() - pre->getXScale()) : 1;
   int nyKernel = (pre->getYScale() < post->getYScale()) ? pow(2,
         post->getYScale() - pre->getYScale()) : 1;
   int kxKernel = kxPreRestricted % nxKernel;
   int kyKernel = kyPreRestricted % nyKernel;

   kernelIndex = kIndex(kxKernel, kyKernel, kfPre, nxKernel, nyKernel, nfPre);
   if (kxKernelIndex != NULL){
      *kxKernelIndex = kxKernel;
   }
   if (kyKernelIndex != NULL){
      *kyKernelIndex = kyKernel;
   }
   if (kfKernelIndex != NULL){
      *kfKernelIndex = kfPre;
   }
   return kernelIndex;
}


} // namespace PV
