/*
 * HyPerConn.cpp
 *
 *  Created on: Oct 21, 2008
 *      Author: rasmussn
 */

#include "HyPerConn.hpp"
#include "../io/ConnectionProbe.hpp"
#include "../io/io.h"
#include "../io/fileio.hpp"
#include "../utils/conversions.h"
#include "../utils/rng.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

namespace PV {

// default values

PVConnParams defaultConnParams =
{
   /*delay*/ 0, /*fixDelay*/ 0, /*varDelayMin*/ 0, /*varDelayMax*/ 0, /*numDelay*/ 1,
   /*isGraded*/ 0, /*vel*/ 45.248, /*rmin*/ 0.0, /*rmax*/ 4.0
};

HyPerConn::HyPerConn()
{
   initialize_base();
}

HyPerConn::HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, int channel)
{
   initialize_base();
   initialize(name, hc, pre, post, channel);
}

// provide filename or set to NULL
HyPerConn::HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, int channel, const char * filename)
{
   initialize_base();
   initialize(name, hc, pre, post, channel, filename);
}

HyPerConn::~HyPerConn()
{
   free(name);

   assert(params != NULL);
   free(params);

   deleteWeights();

   // free the task information

   for (int l = 0; l < numAxonalArborLists; l++) {
      // following frees all data patches for border region l because all
      // patches are allocated together, so freeing freeing first one will do
      free(this->axonalArbor(0, l)->data);
      free(this->axonalArborList[l]);
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
   this->name = "Unknown";
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

   this->probes = NULL;
   this->numProbes = 0;

   // STDP parameters for modifying weights
   this->pIncr = NULL;
   this->pDecr = NULL;
   this->stdpFlag = false;
   this->ampLTP = 1.0;
   this->ampLTD = 1.1;
   this->tauLTP = 20;
   this->tauLTD = 20;
   this->dWMax = 0.1;
   this->wMax = 1.0;

   this->wPostTime = -1.0;
   this->wPostPatches = NULL;

   for (int i = 0; i < MAX_ARBOR_LIST; i++) {
      wPatches[i] = NULL;
      axonalArborList[i] = NULL;
   }

   return 0;
}

//!
/*!
 * REMARKS:
 *      - Each neuron in the pre-synaptic layer can project "up"
 *      a number of arbors. Each arbor connects to a patch in the post-synaptic
 *      layer.
 *      - writeTime and writeStep are used to write post-synaptic pataches.These
 *      patches are written every writeStep.
 *      .
 */
int HyPerConn::initialize(const char * filename)
{
   int status = 0;
   const int arbor = 0;
   numAxonalArborLists = 1;

   assert(this->channel <= post->clayer->numPhis);

   this->connId = parent->numberOfConnections();

   PVParams * inputParams = parent->parameters();
   setParams(inputParams, &defaultConnParams);

   setPatchSize(filename);

   wPatches[arbor] = createWeights(wPatches[arbor]);

   initializeSTDP();

   // Create list of axonal arbors containing pointers to {phi,w,P,M} patches.
   //  weight patches may shrink
   // readWeights() should expect shrunken patches
   // initializeWeights() must be aware that patches may not be uniform
   createAxonalArbors();

   wPatches[arbor]
         = initializeWeights(wPatches[arbor], numWeightPatches(arbor), filename);
   assert(wPatches[arbor] != NULL);

   writeTime = parent->simulationTime();
   writeStep = inputParams->value(name, "writeStep", parent->getDeltaTime());

   parent->addConnection(this);

   return status;
}

int HyPerConn::initializeSTDP()
{
   int arbor = 0;
   if (stdpFlag) {
      int numPatches = numWeightPatches(arbor);
      pIncr = createWeights(NULL, numPatches, nxp, nyp, nfp);
      assert(pIncr != NULL);
      pDecr = pvcube_new(&post->clayer->loc, post->clayer->numExtended);
      assert(pDecr != NULL);
   }
   else {
      pIncr = NULL;
      pDecr = NULL;
   }
   return 0;
}

int HyPerConn::initialize(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, int channel, const char * filename)
{
   this->parent = hc;
   this->pre = pre;
   this->post = post;
   this->channel = channel;

   this->name = strdup(name);
   assert(this->name != NULL);

   return initialize(filename);
}

int HyPerConn::initialize(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, int channel)
{
   return initialize(name, hc, pre, post, channel, NULL);
}

int HyPerConn::setParams(PVParams * filep, PVConnParams * p)
{
   const char * name = getName();

   params = (PVConnParams *) malloc(sizeof(*p));
   assert(params != NULL);
   memcpy(params, p, sizeof(*p));

   numParams = sizeof(*p) / sizeof(float);
   assert(numParams == 9); // catch changes in structure

   params->delay    = (int) filep->value(name, "delay", params->delay);
   params->fixDelay = (int) filep->value(name, "fixDelay", params->fixDelay);

   params->vel      = filep->value(name, "vel", params->vel);
   params->rmin     = filep->value(name, "rmin", params->rmin);
   params->rmax     = filep->value(name, "rmax", params->rmax);

   params->varDelayMin = (int) filep->value(name, "varDelayMin", params->varDelayMin);
   params->varDelayMax = (int) filep->value(name, "varDelayMax", params->varDelayMax);
   params->numDelay    = (int) filep->value(name, "numDelay"   , params->numDelay);
   params->isGraded    = (int) filep->value(name, "isGraded"   , params->isGraded);

   assert(params->delay < MAX_F_DELAY);
   params->numDelay = params->varDelayMax - params->varDelayMin + 1;

   //
   // now set params that are not in the params struct (instance variables)

   wMax = filep->value(name, "strength", wMax);
   // let wMax override strength if user provides it
   wMax = filep->value(name, "wMax", wMax);

   //override dWMax if user provides it
   dWMax = filep->value(name, "dWMax", dWMax);

   stdpFlag = (bool) filep->value(name, "stdpFlag", (float) stdpFlag);

   return 0;
}

// returns handle to initialized weight patches
PVPatch ** HyPerConn::initializeWeights(PVPatch ** patches, int numPatches, const char * filename)
{
   if (filename != NULL) {
      return readWeights(patches, numPatches, filename);
      //return normalizeWeights(patches, numPatches);
   }

   PVParams * inputParams = parent->parameters();
   if (inputParams->present(getName(), "initFromLastFlag")) {
      if ((int) inputParams->value(getName(), "initFromLastFlag") == 1) {
         char name[PV_PATH_MAX];
         snprintf(name, PV_PATH_MAX-1, "%s/w%1.1d_last.pvp", OUTPUT_PATH, getConnectionId());
         return readWeights(patches, numPatches, name);
         //return normalizeWeights(patches, numPatches);
      }
   }

   int randomFlag = (int) inputParams->value(getName(), "randomFlag", 0.0f);
   int randomSeed = (int) inputParams->value(getName(), "randomSeed", 0.0f);
   int smartWeights = (int) inputParams->value(getName(), "smartWeights",0.0f);

   if (randomFlag != 0 || randomSeed != 0) {
      return initializeRandomWeights(patches, numPatches, randomSeed);
   }
   else if (smartWeights != 0) {
      return initializeSmartWeights(patches, numPatches);
   }
   else {
      initializeDefaultWeights(patches, numPatches);
      return normalizeWeights(patches, numPatches);
   }
}

int HyPerConn::checkPVPFileHeader(const PVLayerLoc * loc, int params[], int numParams)
{
   // use default header checker
   //
   return pvp_check_file_header(loc, params, numParams);
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
/*!
 * NOTES:
 *    - numPatches also counts the neurons in the boundary layer. It gives the size
 *    of the extended neuron space.
 *
 */
PVPatch ** HyPerConn::initializeRandomWeights(PVPatch ** patches, int numPatches,
      int seed)
{
   PVParams * inputParams = parent->parameters();

   float uniform_weights = inputParams->value(getName(), "uniformWeights", 0.0f);
   float gaussian_weights = inputParams->value(getName(), "gaussianWeights", 0.0f);

   if(uniform_weights && gaussian_weights){
      fprintf(stderr,"multiple random weights distributions defined: exit\n");
      exit(-1);
   }

   if (uniform_weights) {
      float wMin = inputParams->value(getName(), "wMin", 0.0f);
      float wMax = inputParams->value(getName(), "wMax", 10.0f);

      float wMinInit = inputParams->value(getName(), "wMinInit", wMin);
      float wMaxInit = inputParams->value(getName(), "wMaxInit", wMax);

      int seed = (int) inputParams->value(getName(), "randomSeed", 0);

      for (int k = 0; k < numPatches; k++) {
         uniformWeights(patches[k], wMinInit, wMaxInit, &seed); // MA
      }
   }
   else if (gaussian_weights) {
         float wGaussMean = inputParams->value(getName(), "wGaussMean", 0.5f);
         float wGaussStdev = inputParams->value(getName(), "wGaussStdev", 0.1f);
         int seed = (int) inputParams->value(getName(), "randomSeed", 0);
         for (int k = 0; k < numPatches; k++) {
            gaussianWeights(patches[k], wGaussMean, wGaussStdev, &seed); // MA
         }
      }
   else{
      fprintf(stderr,"no random weights distribution was defined: exit\n");
      exit(-1);
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
   return initializeGaussianWeights(patches, numPatches);
}

PVPatch ** HyPerConn::initializeGaussianWeights(PVPatch ** patches, int numPatches)
{
   PVParams * params = parent->parameters();

   // default values (chosen for center on cell of one pixel)
   int noPost = (int) params->value(post->getName(), "no", nfp);

   float aspect = 1.0; // circular (not line oriented)
   float sigma = 0.8;
   float rMax = 1.4;
   float strength = 1.0;

   aspect   = params->value(name, "aspect", aspect);
   sigma    = params->value(name, "sigma", sigma);
   rMax     = params->value(name, "rMax", rMax);
   strength = params->value(name, "strength", strength);

   float r2Max = rMax * rMax;

   int numFlanks = 1;
   float shift   = 0.0f;
   float rotate  = 0.0f; // rotate so that axis isn't aligned

   numFlanks = params->value(name, "numFlanks", numFlanks);
   shift = params->value(name, "flankShift", shift);
   rotate = params->value(name, "rotate", rotate);

   for (int kernelIndex = 0; kernelIndex < numPatches; kernelIndex++) {
      int patchIndex = kernelIndexToPatchIndex(kernelIndex);
      gauss2DCalcWeights(patches[kernelIndex], patchIndex, noPost, numFlanks, shift, rotate,
            aspect, sigma, r2Max, strength);
   }

   return patches;
}

PVPatch ** HyPerConn::readWeights(PVPatch ** patches, int numPatches, const char * filename)
{
   double time;
   int status = PV::readWeights(patches, numPatches, filename, parent->icCommunicator(),
                                &time, &pre->clayer->loc, true);

   if (status != 0) {
      fprintf(stderr, "SHUTTING DOWN");
      exit(1);
   }

   return patches;
}

int HyPerConn::writeWeights(float time, bool last)
{
   const int arbor = 0;
   const int numPatches = numWeightPatches(arbor);
   return writeWeights(wPatches[arbor], numPatches, NULL, time, last);
}

int HyPerConn::writeWeights(PVPatch ** patches, int numPatches,
                            const char * filename, float time, bool last)
{
   int status = 0;
   char path[PV_PATH_MAX];

   const float minVal = minWeight();
   const float maxVal = maxWeight();

   const PVLayerLoc * loc = &pre->clayer->loc;

   if (patches == NULL) return 0;

   if (filename == NULL) {
      if (last) {
         snprintf(path, PV_PATH_MAX-1, "%sw%d_last.pvp", OUTPUT_PATH, getConnectionId());
      }
      else {
         snprintf(path, PV_PATH_MAX - 1, "%sw%d.pvp", OUTPUT_PATH, getConnectionId());
      }
   }
   else {
      snprintf(path, PV_PATH_MAX-1, "%s", filename);
   }

   Communicator * comm = parent->icCommunicator();

   bool append = (last) ? false : ioAppend;

   status = PV::writeWeights(path, comm, (double) time, append,
                             loc, nxp, nyp, nfp, minVal, maxVal,
                             patches, numPatches);
   assert(status == 0);

#ifdef DEBUG_WEIGHTS
   char outfile[PV_PATH_MAX];

   // only write first weight patch

   sprintf(outfile, "%sw%d.tif", OUTPUT_PATH, getConnectionId());
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
      snprintf(outfile, PV_PATH_MAX-1, "%s%s", OUTPUT_PATH, filename);
      fd = fopen(outfile, "w");
      if (fd == NULL) {
         fprintf(stderr, "writeWeights: ERROR opening file %s\n", filename);
         return 1;
      }
   }

   fprintf(fd, "Weights for connection \"%s\", neuron %d\n", name, k);
   fprintf(fd, "   (kxPre,kyPre,kfPre)   = (%i,%i,%i)\n",
           kxPos(k,pre->clayer->loc.nx, pre->clayer->loc.ny, pre->clayer->numFeatures),
           kyPos(k,pre->clayer->loc.nx, pre->clayer->loc.ny, pre->clayer->numFeatures),
           featureIndex(k,pre->clayer->loc.nx, pre->clayer->loc.ny, pre->clayer->numFeatures) );
   fprintf(fd, "   (nxp,nyp,nfp)   = (%i,%i,%i)\n", (int) nxp, (int) nyp, (int) nfp);
   fprintf(fd, "   pre  (nx,ny,nf) = (%i,%i,%i)\n",
           pre->clayer->loc.nx, pre->clayer->loc.ny, pre->clayer->numFeatures);
   fprintf(fd, "   post (nx,ny,nf) = (%i,%i,%i)\n",
           post->clayer->loc.nx, post->clayer->loc.ny, post->clayer->numFeatures);
   fprintf(fd, "\n");
   if (stdpFlag) {
      pv_text_write_patch(fd, pIncr[k]); // write the Ps variable
   }
   int arbor = 0;
   pv_text_write_patch(fd, wPatches[arbor][k]);
   fprintf(fd, "----------------------------\n");

   if (fd != stdout) {
      fclose(fd);
   }

   return 0;
}

int HyPerConn::deliver(Publisher * pub, PVLayerCube * cube, int neighbor)
{
#ifdef DEBUG_OUTPUT
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   printf("[%d]: HyPerConn::deliver: neighbor=%d cube=%p post=%p this=%p\n", rank, neighbor, cube, post, this);
   fflush(stdout);
#endif
   post->recvSynapticInput(this, cube, neighbor);
#ifdef DEBUG_OUTPUT
   printf("[%d]: HyPerConn::delivered: \n", rank);
   fflush(stdout);
#endif
   if (stdpFlag) {
      updateWeights(cube, neighbor);
   }

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

      if (stdpFlag) {
         convertPreSynapticWeights(time);
         status = writePostSynapticWeights(time, last);
         assert(status == 0);
      }
   }
   else if (stdpFlag && time >= writeTime) {
      writeTime += writeStep;

      status = writeWeights(time, last);
      assert(status == 0);

      convertPreSynapticWeights(time);
      status = writePostSynapticWeights(time, last);
      assert(status == 0);

      // append to output file after original open
      ioAppend = true;
   }

   return status;
}

int HyPerConn::updateState(float time, float dt)
{
   if (stdpFlag) {
      const float fac = ampLTD;
      const float decay = expf(-dt / tauLTD);

      //
      // both pDecr and activity are extended regions (plus margins)
      // to make processing them together simpler

      float * a = post->clayer->activity->data;
      float * m = pDecr->data; // decrement (minus) variable
      int nk = pDecr->numItems;

      for (int k = 0; k < nk; k++) {
         m[k] = decay * m[k] - fac * a[k];
      }
   }

   return 0;
}

int HyPerConn::updateWeights(PVLayerCube * preActivityCube, int remoteNeighbor)
{
   // TODO - should no longer have remote neighbors if extended activity works out
   assert(remoteNeighbor == 0);

   const float dt = parent->getDeltaTime();
   const float decayLTP = expf(-dt / tauLTP);

   const int postStrideY = post->clayer->loc.nx + 2 * post->clayer->loc.nPad;

   // assume pDecr has been updated already, and weights have been used, so
   // 1. update Psij (pIncr) for each synapse
   // 2. update wij

   int axonId = 0;

   // TODO - what is happening here (I think this refers to remote neighbors)
   if (preActivityCube->numItems == 0) return 0;

   const int numExtended = preActivityCube->numItems;
   assert(numExtended == numWeightPatches(axonId));

   for (int kPre = 0; kPre < numExtended; kPre++) {
      PVAxonalArbor * arbor = axonalArbor(kPre, axonId);

      const float preActivity = preActivityCube->data[kPre];

      PVPatch * pIncr   = arbor->plasticIncr;
      PVPatch * w       = arbor->weights;
      size_t postOffset = arbor->offset;

      float * postActivity = &post->clayer->activity->data[postOffset];
      float * W = w->data;
      float * M = &pDecr->data[postOffset];  // STDP decrement variable
      float * P =  pIncr->data;              // STDP increment variable

      int nk  = pIncr->nf * pIncr->nx; // one line in x at a time
      int ny  = pIncr->ny;
      int sy  = pIncr->sy;

      // TODO - unroll

      // update Psij (pIncr variable)
      // we are processing patches, one line in y at a time
      for (int y = 0; y < ny; y++) {
         pvpatch_update_plasticity_incr(nk, P + y * sy, preActivity, decayLTP, ampLTP);
      }

      // update weights
      for (int y = 0; y < ny; y++) {
         pvpatch_update_weights(nk, W, M, P, preActivity, postActivity, dWMax, wMax);
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

   return 0;
}

int HyPerConn::numDataPatches(int arbor)
{
   return numWeightPatches(arbor);
}

/**
 * returns the number of weight patches for the given neighbor
 * @param neighbor the id of the neighbor (0 for interior/self)
 */
int HyPerConn::numWeightPatches(int arbor)
{
   // for now there is just one axonal arbor
   // extending to all neurons in extended layer
   return pre->clayer->numExtended;
}

PVPatch * HyPerConn::getWeights(int k, int arbor)
{
   // a separate arbor/patch of weights for every neuron
   return wPatches[arbor][k];
}

PVPatch * HyPerConn::getPlasticityIncrement(int k, int arbor)
{
   // a separate arbor/patch of plasticity for every neuron
   if (stdpFlag) {
      return pIncr[k];
   }
   return NULL;
}

PVPatch ** HyPerConn::createWeights(PVPatch ** patches, int nPatches, int nxPatch,
      int nyPatch, int nfPatch)
{
   // could create only a single patch with following call
   //   return createPatches(numAxonalArborLists, nxp, nyp, nfp);

   assert(numAxonalArborLists == 1);
   assert(patches == NULL);

   patches = (PVPatch**) calloc(sizeof(PVPatch*), nPatches);
   assert(patches != NULL);

   // TODO - allocate space for them all at once (inplace)
   allocWeights(patches, nPatches, nxPatch, nyPatch, nfPatch);

   return patches;
}

/**
 * Create a separate patch of weights for every neuron
 */
PVPatch ** HyPerConn::createWeights(PVPatch ** patches)
{
   const int arbor = 0;
   int nPatches = numWeightPatches(arbor);
   int nxPatch = nxp;
   int nyPatch = nyp;
   int nfPatch = nfp;

   return createWeights(patches, nPatches, nxPatch, nyPatch, nfPatch);
}

int HyPerConn::deleteWeights()
{
   // to be used if createPatches is used above
   // HyPerConn::deletePatches(numAxonalArborLists, wPatches);

   for (int arbor = 0; arbor < numAxonalArborLists; arbor++) {
      int numPatches = numWeightPatches(arbor);
      if (wPatches[arbor] != NULL) {
         for (int k = 0; k < numPatches; k++) {
            pvpatch_inplace_delete(wPatches[arbor][k]);
         }
         free(wPatches[arbor]);
         wPatches[arbor] = NULL;
      }
   }

   if (wPostPatches != NULL) {
      const int numPostNeurons = post->clayer->numNeurons;
      for (int k = 0; k < numPostNeurons; k++) {
         pvpatch_inplace_delete(wPostPatches[k]);
      }
      free(wPostPatches);
   }

   if (stdpFlag) {
      const int arbor = 0;
      int numPatches = numWeightPatches(arbor);
      for (int k = 0; k < numPatches; k++) {
         pvpatch_inplace_delete(pIncr[k]);
      }
      free(pIncr);
      pvcube_delete(pDecr);
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
int HyPerConn::createAxonalArbors()
{
   const PVLayer * lPre  = pre->clayer;
   const PVLayer * lPost = post->clayer;

   const int prePad  = lPre->loc.nPad;
   const int postPad = lPost->loc.nPad;

   const int nxPre  = lPre->loc.nx;
   const int nyPre  = lPre->loc.ny;
   const int kx0Pre = lPre->loc.kx0;
   const int ky0Pre = lPre->loc.ky0;
   const int nfPre  = lPre->numFeatures;

   const int nxexPre = nxPre + 2 * prePad;
   const int nyexPre = nyPre + 2 * prePad;

   const int nxPost  = lPost->loc.nx;
   const int nyPost  = lPost->loc.ny;
   const int kx0Post = lPost->loc.kx0;
   const int ky0Post = lPost->loc.ky0;
   const int nfPost  = lPost->numFeatures;

   const int nxexPost = nxPost + 2 * postPad;
   const int nyexPost = nyPost + 2 * postPad;

   const int numAxons = numAxonalArborLists;

   // these strides are for post-synaptic phi variable, a non-extended layer variable
   //
#ifndef FEATURES_LAST
   const int psf = 1;
   const int psx = nfp;
   const int psy = psx * nxPost; // (nxPost + 2*postPad); // TODO- check me///////////
#else
   const int psx = 1;
   const int psy = nxPost; // + 2*postPad;
   const int psf = psy * nyPost; // (nyPost + 2*postPad);
#endif

   // activity and STDP M variable are extended into margins
   //
   for (int n = 0; n < numAxons; n++) {
      int numArbors = numWeightPatches(n);
      axonalArborList[n] = (PVAxonalArbor*) calloc(numArbors, sizeof(PVAxonalArbor));
      assert(axonalArborList[n] != NULL);
   }

   for (int n = 0; n < numAxons; n++) {
      int numArbors = numWeightPatches(n);
      PVPatch * dataPatches = (PVPatch *) calloc(numArbors, sizeof(PVPatch));
      assert(dataPatches != NULL);

      for (int kex = 0; kex < numArbors; kex++) {
         PVAxonalArbor * arbor = axonalArbor(kex, n);

         // kex is in extended frame, this makes transformations more difficult

         // local indices in extended frame
         int kxPre = kxPos(kex, nxexPre, nyexPre, nfPre);
         int kyPre = kyPos(kex, nxexPre, nyexPre, nfPre);

         // convert to global non-extended frame
         kxPre += kx0Pre - prePad;
         kyPre += ky0Pre - prePad;

         // global non-extended post-synaptic frame
         int kxPost = zPatchHead(kxPre, nxp, lPre->xScale, lPost->xScale);
         int kyPost = zPatchHead(kyPre, nyp, lPre->yScale, lPost->yScale);

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

         arbor->data = &dataPatches[kex];
         arbor->weights = this->getWeights(kex, n);
         arbor->plasticIncr = this->getPlasticityIncrement(kex, n);

         // initialize the receiving (of spiking data) phi variable
         pvdata_t * phi = lPost->phi[channel] + kl;
         pvpatch_init(arbor->data, nxPatch, nyPatch, nfp, psx, psy, psf, phi);

         //
         // get offset in extended frame for post-synaptic M STDP variable

         kxPost += postPad;
         kyPost += postPad;

         kl = kIndex(kxPost, kyPost, kfPost, nxexPost, nyexPost, nfPost);
         assert(kl >= 0);
         assert(kl < lPost->numExtended);

         arbor->offset = kl;

         // adjust patch size (shrink) to fit within interior of post-synaptic layer

         pvpatch_adjust(arbor->weights, nxPatch, nyPatch, dx, dy);
         if (stdpFlag) {
            arbor->offset += (size_t)dx * (size_t)arbor->weights->sx +
                             (size_t)dy * (size_t)arbor->weights->sy;
            pvpatch_adjust(arbor->plasticIncr, nxPatch, nyPatch, dx, dy);
         }

      } // loop over arbors (pre-synaptic neurons)
   } // loop over neighbors

   return 0;
}

PVPatch ** HyPerConn::convertPreSynapticWeights(float time)
{
   if (time <= wPostTime) {
      return wPostPatches;
   }
   wPostTime = time;

   const PVLayer * lPre  = pre->clayer;
   const PVLayer * lPost = post->clayer;

   const int xScale = lPost->xScale - lPre->xScale;
   const int yScale = lPost->yScale - lPre->yScale;
   const float powXScale = powf(2.0f, (float) xScale);
   const float powYScale = powf(2.0f, (float) yScale);

   // TODO - fix this
   assert(xScale <= 0);
   assert(yScale <= 0);

   const int prePad = lPre->loc.nPad;

   // pre-synaptic weights are in extended layer reference frame
   const int nxPre = lPre->loc.nx + 2 * prePad;
   const int nyPre = lPre->loc.ny + 2 * prePad;
   const int nfPre = lPre->numFeatures;

   const int nxPost  = lPost->loc.nx;
   const int nyPost  = lPost->loc.ny;
   const int nfPost  = lPost->numFeatures;
   const int numPost = lPost->numNeurons;

   const int nxPrePatch = nxp;
   const int nyPrePatch = nyp;

   const int nxPostPatch = (int) (nxp * powXScale);
   const int nyPostPatch = (int) (nyp * powYScale);
   const int nfPostPatch = lPre->numFeatures;

   // the number of features is the end-point value (normally post-synaptic)
   const int numPostPatch = nxPostPatch * nyPostPatch * nfPostPatch;

   if (wPostPatches == NULL) {
      wPostPatches = createWeights(NULL, numPost, nxPostPatch, nyPostPatch, nfPostPatch);
   }

   // loop through post-synaptic neurons (non-extended indices)

   for (int kPost = 0; kPost < numPost; kPost++) {
      int kxPost = kxPos(kPost, nxPost, nyPost, nfPost);
      int kyPost = kyPos(kPost, nxPost, nyPost, nfPost);
      int kfPost = featureIndex(kPost, nxPost, nyPost, nfPost);

      int kxPreHead = zPatchHead(kxPost, nxPostPatch, lPost->xScale, lPre->xScale);
      int kyPreHead = zPatchHead(kyPost, nyPostPatch, lPost->yScale, lPre->yScale);

      // convert kxPreHead and kyPreHead to extended indices
      kxPreHead += prePad;
      kyPreHead += prePad;

      // TODO - FIXME for powXScale > 1
      int ax = (int) (1.0f / powXScale);
      int ay = (int) (1.0f / powYScale);
      int xShift = (ax - 1) - (kxPost + (int) (0.5f * ax)) % ax;
      int yShift = (ay - 1) - (kyPost + (int) (0.5f * ay)) % ay;

      for (int kp = 0; kp < numPostPatch; kp++) {
         int kxPostPatch = (int) kxPos(kp, nxPostPatch, nyPostPatch, nfPre);
         int kyPostPatch = (int) kyPos(kp, nxPostPatch, nyPostPatch, nfPre);
         int kfPostPatch = (int) featureIndex(kp, nxPostPatch, nyPostPatch, nfPre);

         int kxPre = kxPreHead + kxPostPatch;
         int kyPre = kyPreHead + kyPostPatch;
         int kfPre = kfPostPatch;
         int kPre = kIndex(kxPre, kyPre, kfPre, nxPre, nyPre, nfPre);

         // Marian, Shreyas, David changed conditions to fix boundary problems
         if (kxPre < 0 || kyPre < 0 || kxPre >= nxPre || kyPre >= nyPre) {
            assert(kxPre < 0 || kyPre < 0 || kxPre >= nxPre || kyPre >= nyPre);
            wPostPatches[kPost]->data[kp] = 0.0;
         }
         else {
            int arbor = 0;
            PVPatch * p = wPatches[arbor][kPre];

            // The patch from the pre-synaptic layer could be smaller at borders.
            // At top and left borders, calculate the offset back to the original
            // data pointer for the patch.  This make indexing uniform.
            //
            int dx = (kxPre < nxPre / 2) ? nxPrePatch - p->nx : 0;
            int dy = (kyPre < nyPre / 2) ? nyPrePatch - p->ny : 0;
            int prePatchOffset = - p->sx * dx - p->sy * dy;

            int kxPrePatch = (nxPrePatch - 1) - ax * kxPostPatch - xShift;
            int kyPrePatch = (nyPrePatch - 1) - ay * kyPostPatch - yShift;
            int kPrePatch = kIndex(kxPrePatch, kyPrePatch, kfPost, nxPrePatch, nyPrePatch, p->nf);
            wPostPatches[kPost]->data[kp] = p->data[kPrePatch + prePatchOffset];
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

   const PVLayer * lPre  = pre->clayer;
   const PVLayer * lPost = post->clayer;

   const int xScale = lPost->xScale - lPre->xScale;
   const int yScale = lPost->yScale - lPre->yScale;
   const float powXScale = powf(2, (float) xScale);
   const float powYScale = powf(2, (float) yScale);

   const int nxPostPatch = (int) (nxp * powXScale);
   const int nyPostPatch = (int) (nyp * powYScale);

   int kxPreHead = zPatchHead(kxPost, nxPostPatch, lPost->xScale, lPre->xScale);
   int kyPreHead = zPatchHead(kyPost, nyPostPatch, lPost->yScale, lPre->yScale);

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

   const PVLayer * lPre  = pre->clayer;
   const PVLayer * lPost = post->clayer;

   const int prePad  = lPre->loc.nPad;

   const int nxPre  = lPre->loc.nx;
   const int nyPre  = lPre->loc.ny;
   const int kx0Pre = lPre->loc.kx0;
   const int ky0Pre = lPre->loc.ky0;
   const int nfPre  = lPre->numFeatures;

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
   int kxPost = zPatchHead(kxPre, nxp, lPre->xScale, lPost->xScale);
   int kyPost = zPatchHead(kyPre, nyp, lPre->yScale, lPost->yScale);

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

int HyPerConn::writePostSynapticWeights(float time, bool last)
{
   int status = 0;
   char path[PV_PATH_MAX];

   const PVLayer * lPre  = pre->clayer;
   const PVLayer * lPost = post->clayer;

   const float minVal = minWeight();
   const float maxVal = maxWeight();

   const int numPostPatches = lPost->numNeurons;

   const int xScale = lPost->xScale - lPre->xScale;
   const int yScale = lPost->yScale - lPre->yScale;
   const float powXScale = powf(2, (float) xScale);
   const float powYScale = powf(2, (float) yScale);

   const int nxPostPatch = (int) (nxp * powXScale);
   const int nyPostPatch = (int) (nyp * powYScale);
   const int nfPostPatch = lPre->numFeatures;

   const char * last_str = (last) ? "_last" : "";
   snprintf(path, PV_PATH_MAX-1, "%s/w%d_post%s.pvp", OUTPUT_PATH, getConnectionId(), last_str);

   const PVLayerLoc * loc  = &post->clayer->loc;
   Communicator   * comm = parent->icCommunicator();

   bool append = (last) ? false : ioAppend;

   status = PV::writeWeights(path, comm, (double) time, append,
                             loc, nxPostPatch, nyPostPatch, nfPostPatch, minVal, maxVal,
                             wPostPatches, numPostPatches);
   assert(status == 0);

   return 0;
}

/**
 * calculate random weights for a patch given a range between wMin and wMax
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
int HyPerConn::uniformWeights(PVPatch * wp, float wMin, float wMax, int * seed)
{
   pvdata_t * w = wp->data;

   const int nxp = wp->nx;
   const int nyp = wp->ny;
   const int nfp = wp->nf;

   const int sxp = wp->sx;
   const int syp = wp->sy;
   const int sfp = wp->sf;

   double p = (wMax - wMin) / RAND_MAX;

   // loop over all post-synaptic cells in patch
   for (int y = 0; y < nyp; y++) {
      for (int x = 0; x < nxp; x++) {
         for (int f = 0; f < nfp; f++) {
            w[x * sxp + y * syp + f * sfp] = wMin + p * rand();
         }
      }
   }

   return 0;
}


/**
 * calculate random weights for a patch given a range between wMin and wMax
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
int HyPerConn::gaussianWeights(PVPatch * wp, float mean, float stdev, int * seed)
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
int HyPerConn::gauss2DCalcWeights(PVPatch * wp, int kPre, int no,
                                  int numFlanks, float shift, float rotate,
                                  float aspect, float sigma, float r2Max, float strength)
{
   const PVLayer * lPre  = pre->clayer;
   const PVLayer * lPost = post->clayer;

   pvdata_t * w = wp->data;

   const int nxPatch = wp->nx;
   const int nyPatch = wp->ny;
   const int nfPatch = wp->nf;
   if (nxPatch * nyPatch * nfPatch == 0) {
      return 0; // reduced patch size is zero
   }

   float xPreGlobal = 0.0;
   float yPreGlobal = 0.0;
   float xPatchHeadGlobal = 0.0;
   float yPatchHeadGlobal = 0.0;
   posPatchHead(kPre, lPre->xScale,
         lPre->yScale, lPre->loc, &xPreGlobal,
         &yPreGlobal, lPost->xScale, lPost->yScale,
         lPost->loc, wp, &xPatchHeadGlobal,
         &yPatchHeadGlobal);

   // ready to compute weights
   const int sx = wp->sx;
   assert(sx == nfPatch);
   const int sy = wp->sy; // no assert here because patch may be shrunken
   const int sf = wp->sf;
   assert(sf == 1);

   // sigma is in units of pre-synaptic layer
   const float dxPost = powf(2, (float) lPost->xScale);
   const float dyPost = powf(2, (float) lPost->yScale);

   const float dth = PI / (float) nfPatch;
   const float th0 = rotate * dth / 2.0f;

   // TODO - the following assumes that if aspect > 1, # orientations = # features
   //   int noPost = no;
   // number of orientations only used if aspect != 1
   const int noPost = post->clayer->numFeatures;
   const int noPre = pre->clayer->numFeatures;
   const int fPre = kPre % pre->clayer->numFeatures;
   const int iThPre = kPre % noPre;
   const float dthPre = PI / (float) noPre;
   const float thPre = th0 + iThPre * dthPre;

   // loop over all post-synaptic cells in patch
   for (int fPost = 0; fPost < nfPatch; fPost++) {
      int oPost = fPost % noPost;
      float thPost = th0 + oPost * dth;
      if (noPost == 1 && noPre > 1) {
         thPost = thPre;
      }
      for (int jPost = 0; jPost < nyPatch; jPost++) {
         float yDelta = (yPatchHeadGlobal + jPost * dyPost) - yPreGlobal;
         for (int iPost = 0; iPost < nxPatch; iPost++) {
            float xDelta = (xPatchHeadGlobal + iPost * dxPost) - xPreGlobal;

            // no self-interactions
            bool selfFlag = 0;
            if ((int) pre == (int) post) {
               if (fPre == fPost) {
                  if (fabs(xDelta) < 1.0e-5) {
                     if (fabs(yDelta) < 1.0e-5) {
                        selfFlag = 1;
                        continue;
                     }
                  }
               }
            }
            if (selfFlag){
               continue;
            }

            // rotate the reference frame by th
            float xp = +xDelta * cosf(thPost) + yDelta * sinf(thPost);
            float yp = -xDelta * sinf(thPost) + yDelta * cosf(thPost);

            // include shift to flanks
            float d2 = xp * xp + (aspect * (yp - shift) * aspect * (yp - shift));
            w[iPost * sx + jPost * sy + fPost * sf] = 0;
            if (d2 <= r2Max) {
               w[iPost * sx + jPost * sy + fPost * sf] += expf(-d2 / (2.0f * sigma
                     * sigma));
            }
            if (numFlanks > 1) {
               // shift in opposite direction
               d2 = xp * xp + (aspect * (yp + shift) * aspect * (yp + shift));
               if (d2 <= r2Max) {
                  w[iPost * sx + jPost * sy + fPost * sf] += expf(-d2 / (2.0f * sigma
                        * sigma));
               }
            }
         }
      }
   }

   return 0;

#else  // use old method
   pvdata_t * w = wp->data;

   const PVLayer * lPre = pre->clayer;
   const PVLayer * lPost = post->clayer;

   const int nxPatch = wp->nx;
   const int nyPatch = wp->ny;
   const int nfPatch = wp->nf;

   if (nxPatch * nyPatch * nfPatch == 0) {
      return 0; // reduced patch size is zero
   }

   const int sx = wp->sx; assert(sx == nfPatch);
   const int sy = wp->sy; // no assert here because patch may be shrunken
   const int sf = wp->sf; assert(sf == 1);

   // TODO - make sure this is correct
   // sigma is in units of pre-synaptic layer
   const float dxPost = powf(2, lPost->xScale);
   const float dyPost = powf(2, lPost->yScale);

   const int nxPre = lPre->loc.nx;
   const int nyPre = lPre->loc.ny;
   const int nfPre = lPre->numFeatures;

   const int kxPre = (int) kxPos(kPre, nxPre, nyPre, nfPre);
   const int kyPre = (int) kyPos(kPre, nxPre, nyPre, nfPre);

   // location of pre-synaptic neuron (relative to closest post-synaptic neuron)
   float xPre = -1.0 * deltaPosLayers(kxPre, lPre->xScale) * dxPost;
   float yPre = -1.0 * deltaPosLayers(kyPre, lPre->yScale) * dyPost;

   // closest post-synaptic neuron may not be at the center of the patch (0,0)
   // so must shift pre-synaptic location
   if (xPre < 0.0) xPre += 0.5 * dxPost;
   if (xPre > 0.0) xPre -= 0.5 * dxPost;
   if (yPre < 0.0) yPre += 0.5 * dyPost;
   if (yPre > 0.0) yPre -= 0.5 * dyPost;

   // (x0,y0) is at upper left corner of patch (i=0,j=0)
   // and shift so pre-synaptic cell is at 0
   const float xDelta0 = -(nxPatch/2.0 - 0.5) * dxPost - xPre;
   const float yDelta0 = +(nyPatch/2.0 - 0.5) * dyPost - yPre;

   const float dth = PI/nfPatch;
   const float th0 = rotate*dth/2.0;

   // loop over all post-synaptic cells in patch
   for (int fPost = 0; fPost < nfPatch; fPost++) {
      int oPost = fPost % noPost;
      float thPost = th0 + oPost * dth;
      for (int jPost = 0; jPost < nyPatch; jPost++) {
         float yDelta = yDelta0 - jPost * dyPost;
         for (int iPost = 0; iPost < nxPatch; iPost++) {
            float xDelta = xDelta0 + iPost*dxPost;

            // rotate the reference frame by th
            float xp = + xDelta * cos(thPost) + yDelta * sin(thPost);
            float yp = - xDelta * sin(thPost) + yDelta * cos(thPost);

            // include shift to flanks
            float d2 = xp * xp + (aspect*(yp-shift) * aspect*(yp-shift));

            w[iPost*sx + jPost*sy + fPost*sf] = 0;

            // TODO - figure out on/off connectivity
            // don't break it for nfPre==1 going to nfPost=numOrientations
            //if (this->channel == CHANNEL_EXC && f != fPre) continue;
            //if (this->channel == CHANNEL_INH && f == fPre) continue;

            if (d2 <= r2Max) {
               w[iPost*sx + jPost*sy + fPost*sf] = expf(-d2 / (2.0*sigma*sigma));
            }
            if (numFlanks > 1) {
               // shift in opposite direction
               d2 = xp * xp + (aspect*(yp+shift) * aspect*(yp+shift));
               if (d2 <= r2Max) {
                  w[iPost*sx + jPost*sy + fPost*sf] = expf(-d2 / (2.0*sigma*sigma));
               }
            }
            // printf("x=%f y-%f xp=%f yp=%f d2=%f w=%f\n", x, y, xp, yp, d2, w[i*sx + j*sy + f*sf]);
         }
      }
   }

   return 0;
#endif
}

PVPatch ** HyPerConn::normalizeWeights(PVPatch ** patches, int numPatches)
{
   PVParams * params = parent->parameters();
   float strength = params->value(name, "strength", 1.0);

   this->wMax = 1.0;
   float maxVal = 0;
   for (int k = 0; k < numPatches; k++) {
      PVPatch * wp = patches[k];
      pvdata_t * w = wp->data;
      const int nx = (int) wp->nx;
      const int ny = (int) wp->ny;
      const int nf = (int) wp->nf;
      float sum = 0;
      float sum2 = 0;
      for (int i = 0; i < nx * ny * nf; i++) {
         sum += w[i];
         sum2 += w[i] * w[i];
      }
      if (sum == 0.0f && sum2 > 0.0f) {
         float factor = strength / sqrtf(sum2);
         for (int i = 0; i < nx * ny * nf; i++)
            w[i] *= factor;
      }
      else if (sum != 0.0f) {
         float factor = strength / sum;
         for (int i = 0; i < nx * ny * nf; i++)
            w[i] *= factor;
      }
      for (int i = 0; i < nx * ny * nf; i++) {
         maxVal = ( w[i] > maxVal ) ? w[i] : maxVal;
      }
   }
   this->wMax = maxVal * (1.1);
   return patches;
}

int HyPerConn::setPatchSize(const char * filename)
{
   int status = 0;
   PVParams * inputParams = parent->parameters();

   nxp = (int) inputParams->value(name, "nxp", post->clayer->loc.nx);
   nyp = (int) inputParams->value(name, "nyp", post->clayer->loc.ny);
   nfp = (int) inputParams->value(name, "nfp", post->clayer->numFeatures);

   // use patch dimensions from file if (filename != NULL)
   //
   if (filename != NULL) {
      int filetype, datatype;
      double time = 0.0;
      const PVLayerLoc loc = this->pre->clayer->loc;

      int wgtParams[NUM_WGT_PARAMS];
      int numWgtParams = NUM_WGT_PARAMS;

      Communicator * comm = parent->icCommunicator();

      status = pvp_read_header(filename, comm, &time, &filetype, &datatype, wgtParams, &numWgtParams);
      if (status < 0) return status;

      status = checkPVPFileHeader(&loc, wgtParams, numWgtParams);
      if (status < 0) return status;

      // reconcile differences with inputParams
      status = checkWeightsHeader(filename, wgtParams);
   }
   return status;
}

PVPatch ** HyPerConn::allocWeights(PVPatch ** patches, int nPatches, int nxPatch,
      int nyPatch, int nfPatch)
{
   for (int k = 0; k < nPatches; k++) {
      patches[k] = pvpatch_inplace_new(nxPatch, nyPatch, nfPatch);
   }
   return patches;
}

PVPatch ** HyPerConn::allocWeights(PVPatch ** patches)
{
   int arbor = 0;
   int nPatches = numWeightPatches(arbor);
   int nxPatch = nxp;
   int nyPatch = nyp;
   int nfPatch = nfp;

   return allocWeights(patches, nPatches, nxPatch, nyPatch, nfPatch);
}

int HyPerConn::kernelIndexToPatchIndex(int kernelIndex){
   return kernelIndex;
}

// many to one mapping from weight patches to kernels
int HyPerConn::patchIndexToKernelIndex(int patchIndex){
   return patchIndex;
}

} // namespace PV
