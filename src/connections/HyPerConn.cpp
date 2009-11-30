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
      free(this->axonalArbor(0,l)->data);
      free(this->axonalArborList[l]);
   }
}

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
   // now set params that are not in the params struct (instance varibles)

   wMax = filep->value(name, "strength", wMax);
   // let wMax override strength if user provides it
   wMax = filep->value(name, "wMax", wMax);

   //override dWMax if user provides it
   dWMax = filep->value(name, "dWMax", dWMax);

   stdpFlag = (bool) filep->value(name, "stdpFlag", stdpFlag);

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

   float randomFlag = inputParams->value(getName(), "randomFlag", 0.0f);
   float randomSeed = inputParams->value(getName(), "randomSeed", 0.0f);

   if (randomFlag != 0 || randomSeed != 0) {
      return initializeRandomWeights(patches, numPatches, randomSeed);
   }
   else {
      initializeGaussianWeights(patches, numPatches);
      return normalizeWeights(patches, numPatches);
   }
}

int HyPerConn::checkWeightsHeader(const char * filename, int numParamsFile, int nxpFile,
      int nypFile, int nfpFile)
{
   const int numWeightHeaderParams = NUM_WEIGHT_PARAMS;
   if (numWeightHeaderParams != numParamsFile) {
      fprintf(
            stderr,
            "numWeightHeaderParams = %i in HyPerConn %s, using numParamsFile = %i in binary file %s\n",
            numWeightHeaderParams, name, numParamsFile, filename);
   }
   if (nxp != nxpFile) {
      fprintf(stderr,
      "ignoring nxp = %f in HyPerCol %s, using nxp = %i in binary file %s\n", nxp, name,
            nxpFile, filename);
      nxp = nxpFile;
   }
   if ((nyp > 0) && (nyp != nypFile)) {
      fprintf(stderr,
      "ignoring nyp = %f in HyPerCol %s, using nyp = %i in binary file %s\n", nyp, name,
            nypFile, filename);
      nyp = nypFile;
   }
   if ((nfp > 0) && (nfp != nfpFile)) {
      fprintf(stderr,
      "ignoring nfp = %f in HyPerCol %s, using nfp = %i in binary file %s\n", nfp, name,
            nfpFile, filename);
      nfp = nfpFile;
   }
   return 0;
}

PVPatch ** HyPerConn::initializeRandomWeights(PVPatch ** patches, int numPatches,
      int seed)
{
   PVParams * params = parent->parameters();

   float wMin = params->value(name, "wMin", 0.0f);

   for (int k = 0; k < numPatches; k++) {
      randomWeights(patches[k], wMin, wMax, seed);
   }
   return patches;
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
   shift     = params->value(name, "flankShift", shift);
   rotate    = params->value(name, "rotate", rotate);

   const int xScale = post->clayer->xScale - pre->clayer->xScale;
   const int yScale = post->clayer->yScale - pre->clayer->yScale;
   for (int k = 0; k < numPatches; k++) {
      gauss2DCalcWeights(patches[k], k, noPost, xScale, yScale, numFlanks, shift, rotate,
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

#ifdef DONT_COMPILE
   FILE * fp;
   int numParamsFile;
   int fileType, nxpFile, nypFile, nfpFile;

   fp = pv_open_binary(filename, &numParamsFile, &fileType, &nxpFile, &nypFile, &nfpFile);
   if (fp == NULL) {
      return NULL;
   }
   checkWeightsHeader(filename, numParamsFile, nxpFile, nypFile, nfpFile);

   //   int append = 0; // only write one time step
   int status = 0;

   // header information
   const int numWeightHeaderParams = NUM_WEIGHT_PARAMS;
   int params[numWeightHeaderParams];
   float minVal, maxVal;
   int numPatchesFile;

   pv_read_binary_params(fp, numParamsFile, params);

   if (numParamsFile >= (MIN_BIN_PARAMS + 1)) {
      minVal = params[MIN_BIN_PARAMS + 0];
   }
   else {
      minVal = 0;
   }

   if (numParamsFile >= (MIN_BIN_PARAMS + 2)) {
      maxVal = params[MIN_BIN_PARAMS + 1];
   }
   else {
      maxVal = 1.0;
   }

   // numPatches should equal numDataPatches should equal numPatchesFile (if present)
   if (numParamsFile >= (MIN_BIN_PARAMS + 3)) {
      numPatchesFile = params[MIN_BIN_PARAMS + 2];
   }
   else {
      const int arbor = 0;
      numPatchesFile = numDataPatches(arbor);
   }
   if (numPatchesFile != numPatches) {
      fprintf(
            stderr,
            "ignoring numPatches = %i in HyPerCol %s, using numPatchesFile = %i in binary file %s\n",
            numPatches, name, numPatchesFile, filename);
      nxp = nxpFile;
   }

   status = pv_read_patches(fp, nfp, minVal, maxVal, numPatchesFile, patches);
   pv_close_binary(fp);
#endif

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

   const LayerLoc * loc = &pre->clayer->loc;

   if (patches == NULL) return 0;

   if (filename == NULL) {
      if (last) {
         snprintf(path, PV_PATH_MAX-1, "%sw%d_last.pvp", OUTPUT_PATH, getConnectionId());
      }
      else {
         snprintf(path, PV_PATH_MAX-1, "%sw%d.pvp", OUTPUT_PATH, getConnectionId());
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
   fprintf(fd, "   (nxp,nyp,nfp)   = (%d,%d,%d)\n", (int)nxp, (int)nyp, (int)nfp);
   fprintf(fd, "   pre  (nx,ny,nf) = (%d,%d,%d)\n",
           (int)pre->clayer->loc.nx, (int)pre->clayer->loc.ny, (int)pre->clayer->numFeatures);
   fprintf(fd, "   post (nx,ny,nf) = (%d,%d,%d)\n",
           (int)post->clayer->loc.nx, (int)post->clayer->loc.ny, (int)post->clayer->numFeatures);
   fprintf(fd, "\n");
   if (stdpFlag) {
      pv_text_write_patch(fd, pIncr[k]);  // write the Ps variable
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
      const float decay = expf(-dt/tauLTD);

      //
      // both pDecr and activity are extended regions (plus margins)
      // to make processing them together simpler

      float * a = post->clayer->activity->data;
      float * m = pDecr->data;            // decrement (minus) variable
      int nk = pDecr->numItems;

      for (int k = 0; k < nk; k++) {
         m[k] = decay * m[k] - fac * a[k];
//         if (a[k] > 0) {
//            fprintf(stderr, "k=%d, m=%f addr(m)=%p\n", k, m[k], &m[k]);
//         }
      }
   }

   return 0;
}

int HyPerConn::updateWeights(PVLayerCube * preActivityCube, int remoteNeighbor)
{
   // TODO - should no longer have remote neighbors if extended activity works out
   assert(remoteNeighbor == 0);

   const float dt = parent->getDeltaTime();
   const float decayLTP = expf(-dt/tauLTP);

   const int postStrideY = post->clayer->loc.nx + 2*post->clayer->loc.nPad;

   // assume pDecr has been updated already, and weights have been used, so
   // 1. update Psij (pIncr) for each synapse
   // 2. update wij

   int axonId = 0;

   // TODO - what is happening here (I think this refers to remote neighbors)
   if (preActivityCube->numItems == 0) return 0;

   const int numExtended = preActivityCube->numItems;
   assert(numExtended == numWeightPatches(axonId));

   for (int kPre = 0; kPre < numExtended; kPre++) {
      PVAxonalArbor * arbor  = axonalArbor(kPre, axonId);

      const float preActivity = preActivityCube->data[kPre];

      PVPatch * pIncr   = arbor->plasticIncr;
      PVPatch * w       = arbor->weights;
      size_t postOffset = arbor->offset;

      float * postActivity = &post->clayer->activity->data[postOffset];
      float * W = w->data;
      float * M = &pDecr->data[postOffset];  // STDP decrement variable
      float * P =  pIncr->data;              // STDP increment variable

      int nk  = (int) pIncr->nf * (int) pIncr->nx; // one line in x at a time
      int ny  = (int) pIncr->ny;
      int sy  = (int) pIncr->sy;

      // TODO - unroll

      // update Psij (pIncr variable)
      // we are processing patches, one line in y at a time
      for (int y = 0; y < ny; y++) {
         pvpatch_update_plasticity_incr(nk, P + y*sy, preActivity, decayLTP, ampLTP);
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

int HyPerConn::createAxonalArbors()
{
   // TODO - these needs to be an input parameter obtained from the connection
   const float kxPost0Left = 0.0f;
   const float kyPost0Left = 0.0f;

   const float prePad  = pre->clayer->loc.nPad;
   const float postPad = post->clayer->loc.nPad;

   const float nxPre  = pre->clayer->loc.nx;
   const float nyPre  = pre->clayer->loc.ny;
   const float kx0Pre = pre->clayer->loc.kx0;
   const float ky0Pre = pre->clayer->loc.ky0;
   const float nfPre  = pre->clayer->numFeatures;

   const float nxexPre = nxPre + 2.0f*prePad;
   const float nyexPre = nyPre + 2.0f*prePad;

   const float nxPost  = post->clayer->loc.nx;
   const float nyPost  = post->clayer->loc.ny;
   const float kx0Post = post->clayer->loc.kx0;
   const float ky0Post = post->clayer->loc.ky0;
   const float nfPost  = post->clayer->numFeatures;

   const float nxexPost = nxPost + 2.0f*postPad;
   const float nyexPost = nyPost + 2.0f*postPad;

   const int xScale = post->clayer->xScale - pre->clayer->xScale;
   const int yScale = post->clayer->yScale - pre->clayer->yScale;

   const int numAxons = numAxonalArborLists;

//
// these strides are for post-synaptic phi variable, a non-extended layer variable
#ifndef FEATURES_LAST
   const float psf = 1;
   const float psx = nfp;
   const float psy = psx * nxPost; // (nxPost + 2.0f*postPad); // TODO- check me///////////
#else
   const float psx = 1;
   const float psy = nxPost; // + 2.0f*postPad;
   const float psf = psy * nyPost; // (nyPost + 2.0f*postPad);
#endif

   //
   // activity and STDP M variable are extended into margins

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
         float kxPre = kxPos(kex, nxexPre, nyexPre, nfPre);
         float kyPre = kyPos(kex, nxexPre, nyexPre, nfPre);

         // convert to global non-extended frame
         kxPre += kx0Pre - prePad;
         kyPre += ky0Pre - prePad;

         // global non-extended post-synaptic frame
         float kxPost = pvlayer_patchHead(kxPre, kxPost0Left, xScale, nxp);
         float kyPost = pvlayer_patchHead(kyPre, kyPost0Left, yScale, nyp);

         // TODO - can get nf from weight patch but what about kf0?
         // weight patch is actually a pencil and so kfPost is always 0?
         float kfPost = 0.0f;

         // convert to local non-extended post-synaptic frame
         kxPost = kxPost - kx0Post;
         kyPost = kyPost - ky0Post;

         // adjust location so patch is in bounds
         float dx = 0;
         float dy = 0;
         float nxPatch = nxp;
         float nyPatch = nyp;

         if (kxPost < 0.0) {
            nxPatch -= -kxPost;
            kxPost = 0.0;
            if (nxPatch < 0.0) nxPatch = 0.0;
            dx = nxp - nxPatch;
         }
         else if (kxPost + nxp > nxPost) {
            nxPatch -= kxPost + nxp - nxPost;
            if (nxPatch <= 0.0) {
               nxPatch = 0.0;
               kxPost  = nxPost - 1.0;
            }
         }

         if (kyPost < 0.0) {
            nyPatch -= -kyPost;
            kyPost = 0.0;
            if (nyPatch < 0.0) nyPatch = 0.0;
            dy = nyp - nyPatch;
         }
         else if (kyPost + nyp > nyPost) {
            nyPatch -= kyPost + nyp - nyPost;
            if (nyPatch <= 0.0) {
               nyPatch = 0.0;
               kyPost  = nyPost - 1.0;
            }
         }

         // if out of bounds in x (y), also out in y (x)
         if (nxPatch == 0.0 || nyPatch == 0.0) {
            dx = 0.0;
            dy = 0.0;
            nxPatch = 0.0;
            nyPatch = 0.0;
         }

         // local non-extended index but shifted to be in bounds
         int kl = kIndex(kxPost, kyPost, kfPost, nxPost, nyPost, nfPost);
         assert(kl >= 0);
         assert(kl < post->clayer->numNeurons);

         arbor->data = &dataPatches[kex];
         arbor->weights = this->getWeights(kex, n);
         arbor->plasticIncr = this->getPlasticityIncrement(kex, n);

         // initialize the receiving (of spiking data) phi variable
         pvdata_t * phi = post->clayer->phi[channel] + kl;
         pvpatch_init(arbor->data, nxPatch, nyPatch, nfp, psx, psy, psf, phi);

         //
         // get offset in extended frame for post-synaptic M STDP variable

         kxPost += postPad;
         kyPost += postPad;

         kl = kIndex(kxPost, kyPost, kfPost, nxexPost, nyexPost, nfPost);
         assert(kl >= 0);
         assert(kl < post->clayer->numExtended);

         arbor->offset = kl;

         // adjust patch size (shrink) to fit within interior of post-synaptic layer

         pvpatch_adjust(arbor->weights, (int)nxPatch, (int)nyPatch, (int)dx, (int)dy);
         if (stdpFlag) {
            arbor->offset += (size_t)dx * (size_t)arbor->weights->sx +
                             (size_t)dy * (size_t)arbor->weights->sy;
            pvpatch_adjust(arbor->plasticIncr, (int)nxPatch, (int)nyPatch, (int)dx, (int)dy);
         }

      } // loop over arbors (pre-synaptic neurons)
   } // loop over neighbors

   return 0;
}

//////////////
// This code should no longer be needed as the shrinking of weight patches is no
// done in createAxonalArbors
int HyPerConn::adjustAxonalArborWeights()
{
   // TODO - these needs to be an input parameter obtained from the connection
   const float kxPost0Left = 0.0f;
   const float kyPost0Left = 0.0f;

   const float nxBorderPre  = pre->clayer->loc.nPad;
   const float nyBorderPre  = pre->clayer->loc.nPad;

   // use the extended reference frame (with margins) for pre-synaptic layer

   const float nxPre  = pre->clayer->loc.nx + 2.0f * nxBorderPre;
   const float nyPre  = pre->clayer->loc.ny + 2.0f * nyBorderPre;
   const float kx0Pre = pre->clayer->loc.kx0 - nxBorderPre;
   const float ky0Pre = pre->clayer->loc.ky0 - nyBorderPre;
   const float nfPre  = pre->clayer->numFeatures;

   // use the non-extended reference frame for post-synaptic layer
   // because phi data structure is not extended

   const float nxPost  = post->clayer->loc.nx;
   const float nyPost  = post->clayer->loc.ny;
   const float kx0Post = post->clayer->loc.kx0;
   const float ky0Post = post->clayer->loc.ky0;

   const int xScale = post->clayer->xScale - pre->clayer->xScale;
   const int yScale = post->clayer->yScale - pre->clayer->yScale;

   const int numNeighbors = numAxonalArborLists;

   for (int n = 0; n < numNeighbors; n++) {
      int numArbors = numWeightPatches(n);

      for (int k = 0; k < numArbors; k++) {
         PVAxonalArbor * arbor = axonalArbor(k, n);
         int kPre = kIndexFromNeighbor(k, n);

         // global indices
         float kxPre = kx0Pre + kxPos(kPre, nxPre, nyPre, nfPre);
         float kyPre = ky0Pre + kyPos(kPre, nxPre, nyPre, nfPre);

         // global indices
         float kxPost = pvlayer_patchHead(kxPre, kxPost0Left, xScale, nxp);
         float kyPost = pvlayer_patchHead(kyPre, kyPost0Left, yScale, nyp);

         // convert to local frame
         kxPost = kxPost - kx0Post;
         kyPost = kyPost - ky0Post;

         // adjust location so patch is in bounds
         float dx = 0;
         float dy = 0;
         float nxPatch = nxp;
         float nyPatch = nyp;

         if (kxPost < 0.0) {
            nxPatch -= -kxPost;
            kxPost = 0.0;
            if (nxPatch < 0.0) nxPatch = 0.0;
            dx = nxp - nxPatch;
         }
         else if (kxPost + nxp > nxPost) {
            nxPatch -= kxPost + nxp - nxPost;
            if (nxPatch < 0.0) {
               nxPatch = 0.0;
               kxPost  = nxPost - 1.0;
            }
         }

         if (kyPost < 0.0) {
            nyPatch -= -kyPost;
            kyPost = 0.0;
            if (nyPatch < 0.0) nyPatch = 0.0;
            dy = nyp - nyPatch;
         }
         else if (kyPost + nyp > nyPost) {
            nyPatch -= kyPost + nyp - nyPost;
            if (nyPatch < 0.0) {
               nyPatch = 0.0;
               kyPost  = nyPost - 1.0;
            }
         }

         // if out of bounds in x (y), also out in y (x)
         if (nxPatch == 0.0 || nyPatch == 0.0) {
            dx = 0.0;
            dy = 0.0;
            nxPatch = 0.0;
            nyPatch = 0.0;
         }

         pvpatch_adjust(arbor->weights, (int)nxPatch, (int)nyPatch, (int)dx, (int)dy);
         if (stdpFlag) {
            arbor->offset += (size_t)dx * (size_t)arbor->weights->sx +
                             (size_t)dy * (size_t)arbor->weights->sy;
            pvpatch_adjust(arbor->plasticIncr, (int)nxPatch, (int)nyPatch, (int)dx, (int)dy);
         }
      }
   }

   return 0;
}

#ifdef USE_OLD_CODE
// this version has borders in it, need to keep until sure that image doesn't needs to
// have border
int HyPerConn::createAxonalArbors(int numArbors)
{
   // TODO - these needs to be an input parameter obtained from the connection
   const float kxPost0Left = 0.0f;
   const float kyPost0Left = 0.0f;

   const float nxPre  = pre->clayer->loc.nx;
   const float nyPre  = pre->clayer->loc.ny;
   const float kx0Pre = pre->clayer->loc.kx0;
   const float ky0Pre = pre->clayer->loc.ky0;
   const float nfPre  = pre->clayer->numFeatures;

   const float nxPost  = post->clayer->loc.nx;
   const float nyPost  = post->clayer->loc.ny;
   const float kx0Post = post->clayer->loc.kx0;
   const float ky0Post = post->clayer->loc.ky0;
   const float nfPost  = post->clayer->numFeatures;

   const int xScale = post->clayer->xScale - pre->clayer->xScale;
   const int yScale = post->clayer->yScale - pre->clayer->yScale;

   const float numBorder = post->clayer->numBorder;
   assert(numBorder = 0);

#ifndef FEATURES_LAST
   const float psf = 1;
   const float psx = nfp;
   const float psy = psx * (nxPost + 2.0f*numBorder);
#else
   const float psx = 1;
   const float psy = nxPost + 2.0f*numBorder;
   const float psf = psy * (nyPost + 2.0f*numBorder);
#endif

   int nNeurons = pre->clayer->numNeurons;
   int nTotalTasks = nNeurons * numArbors;

   PVAxonalArborList ** allBPtrs   = (PVAxonalArborList**) malloc(nNeurons*sizeof(PVAxonalArborList*));
   PVAxonalArborList  * allBundles = (PVAxonalArborList *) malloc(nNeurons*sizeof(PVAxonalArborList));
   PVAxonalArbor** allTPtrs = (PVAxonalArbor**) malloc(nTotalTasks*sizeof(PVAxonalArbor*));
   PVAxonalArbor * allTasks = (PVAxonalArbor*)  malloc(nTotalTasks*sizeof(PVAxonalArbor));
   PVPatch       * allData  = (PVPatch*)        malloc(nTotalTasks*sizeof(PVPatch));

   assert(allBPtrs != NULL);
   assert(allBundles != NULL);
   assert(allTPtrs != NULL);
   assert(allTasks != NULL);
   assert(allData != NULL);

   bundles = allBPtrs;

   for (int kPre = 0; kPre < nNeurons; kPre++) {
      int offset = kPre*sizeof(PVAxonalArborList);
      bundles[kPre] = (PVAxonalArborList*) ((char*) allBundles + offset);
      bundles[kPre]->numArbors = numArbors;
      bundles[kPre]->axonalArbor = allTPtrs + kPre*numArbors;

      PVAxonalArborList * list = bundles[kPre];
      for (int i = 0; i < numArbors; i++) {
         offset = (i + kPre*numArbors) * sizeof(PVAxonalArbor);
         list->axonalArbor[i] = (PVAxonalArbor*) ((char*) allTasks + offset);
      }

      for (int i = 0; i < numArbors; i++) {
         PVAxonalArbor * task = list->axonalArbor[i];

         // get weights from virtual method
         task->weights = this->getWeights(kPre, i);

         // global indices
         float kxPre = kx0Pre + kxPos(kPre, nxPre, nyPre, nfPre);
         float kyPre = ky0Pre + kyPos(kPre, nxPre, nyPre, nfPre);

         // global indices
         float kxPost = pvlayer_patchHead(kxPre, kxPost0Left, xScale, nxp);
         float kyPost = pvlayer_patchHead(kyPre, kyPost0Left, yScale, nyp);

         // TODO - can get nf from weight patch but what about kf0?
         float kfPost = 0.0f;

         // convert to local in extended frame
         kxPost = kxPost - (kx0Post - numBorder);
         kyPost = kyPost - (ky0Post - numBorder);

         // phi is in extended frame
         int kl = kIndex(kxPost, kyPost, kfPost, nxPost+2.0f*numBorder, nyPost+2.0f*numBorder, nfPost);
         // TODO get the correct phi index
         pvdata_t * phi = post->clayer->phi[channel] + kl;

         offset = (i + kPre*numArbors) * sizeof(PVPatch);
         task->data = (PVPatch*) ((char*) allData + offset);
         pvpatch_init(task->data, nxp, nyp, nfp, psx, psy, psf, phi);
      }
   }

   return 0;
}
#endif

int HyPerConn::kIndexFromNeighbor(int kNeighbor, int neighbor)
{
   int kl;

   switch (neighbor) {
   case 0:
      kl = kNeighbor;  break;
   case NORTHWEST:
      kl = kNeighbor % pre->numberOfNeurons(neighbor);  break;
   case NORTH:
      kl = kNeighbor % pre->numberOfNeurons(neighbor);  break;
   case NORTHEAST:
      kl = kNeighbor % pre->numberOfNeurons(neighbor);  break;
   case WEST:
      kl = kNeighbor % pre->numberOfNeurons(neighbor);  break;
   case EAST:
      kl = kNeighbor % pre->numberOfNeurons(neighbor);  break;
   case SOUTHWEST:
      kl = kNeighbor % pre->numberOfNeurons(neighbor);  break;
   case SOUTH:
      kl = kNeighbor % pre->numberOfNeurons(neighbor);  break;
   case SOUTHEAST:
      kl = kNeighbor % pre->numberOfNeurons(neighbor);  break;
   default:
      fprintf(stderr, "ERROR:HyPerConn:kIndexFromNeighbor: bad border index %d\n", neighbor);
   }

   return kl;
}

PVPatch ** HyPerConn::convertPreSynapticWeights(float time)
{
   if (time <= wPostTime) {
      return wPostPatches;
   }
   wPostTime = time;

   const int xScale = post->clayer->xScale - pre->clayer->xScale;
   const int yScale = post->clayer->yScale - pre->clayer->yScale;
   const float powXScale = powf(2.0, (float) xScale);
   const float powYScale = powf(2.0, (float) yScale);

   // TODO - fix this
   assert(xScale < 0);
   assert(yScale < 0);

   const int prePad = pre->clayer->loc.nPad;

   // pre-synaptic weights are in extended layer reference frame
   const int nxPre  = (int) pre->clayer->loc.nx + 2 * prePad;
   const int nyPre  = (int) pre->clayer->loc.ny + 2 * prePad;
   const int nfPre  = pre->clayer->numFeatures;

   const int nxPost  = (int) post->clayer->loc.nx;
   const int nyPost  = (int) post->clayer->loc.ny;
   const int nfPost  = post->clayer->numFeatures;
   const int numPost = post->clayer->numNeurons;

   const int nxPrePatch = (int) nxp;
   const int nyPrePatch = (int) nyp;

   const int nxPostPatch = (int) (nxp * powXScale);
   const int nyPostPatch = (int) (nyp * powYScale);
   const int nfPostPatch = pre->clayer->numFeatures;

   // the number of features is the end-point value (normally post-synaptic)
   const int numPostPatch = nxPostPatch * nyPostPatch * nfPostPatch;

   if (wPostPatches == NULL) {
      wPostPatches = createWeights(NULL, numPost, nxPostPatch, nyPostPatch, nfPostPatch);
   }

   // loop through post-synaptic neurons

   for (int kPost = 0; kPost < numPost; kPost++) {
      int kxPost = (int) kxPos(kPost, nxPost, nyPost, nfPost);
      int kyPost = (int) kyPos(kPost, nxPost, nyPost, nfPost);
      int kfPost = (int) featureIndex(kPost, nxPost, nyPost, nfPost);

      // TODO - does patchHead work in general for post to pre mapping and -scale?
      int kxPreHead = (int) pvlayer_patchHead((float) kxPost, 0.0, -xScale, (float) nxPostPatch);
      int kyPreHead = (int) pvlayer_patchHead((float) kyPost, 0.0, -yScale, (float) nyPostPatch);

      // convert kxPreHead and kyPreHead to extended indices
      kxPreHead += prePad;
      kyPreHead += prePad;

      // TODO - FIXME for powXScale > 1
      int ax = 1.0f/powXScale;
      int ay = 1.0f/powYScale;
      int xShift = ax - (kxPost + (int)(0.5f * ax)) % ax;
      int yShift = ay - (kyPost + (int)(0.5f * ay)) % ay;

      for (int kp = 0; kp < numPostPatch; kp++) {
         int kxPostPatch = (int) kxPos(kp, nxPostPatch, nyPostPatch, nfPre);
         int kyPostPatch = (int) kyPos(kp, nxPostPatch, nyPostPatch, nfPre);
         int kfPostPatch = (int) featureIndex(kp, nxPostPatch, nyPostPatch, nfPre);

         int kxPre = kxPreHead + kxPostPatch;
         int kyPre = kyPreHead + kyPostPatch;
         int kfPre = kfPostPatch;
         int kPre = kIndex(kxPre, kyPre, kfPre, nxPre, nyPre, nfPre);

         // Marian, Shreyas, David changed conditions to fix boundary problems
         if (kxPre<0 || kyPre < 0 || kxPre >= nxPre|| kyPre >= nyPre) {
            assert(kxPre < 0 || kyPre < 0 || kxPre >= nxPre || kyPre >= nyPre);
            wPostPatches[kPost]->data[kp] = 0.0;
         }
         else {
            int arbor = 0;
            PVPatch * p = wPatches[arbor][kPre];
            int nxp = (kxPre < nxPrePatch/2) ? p->nx : nxPrePatch;
            int nyp = (kyPre < nyPrePatch/2) ? p->ny : nyPrePatch;
            int kxPrePatch = nxp - (1 + kxPostPatch / powXScale);
            int kyPrePatch = nyp - (1 + kyPostPatch / powYScale);
            kxPrePatch = nxp - ax * kxPostPatch - xShift;
            kyPrePatch = nyp - ay * kyPostPatch - yShift;
            int kPrePatch = kIndex(kxPrePatch, kyPrePatch, kfPost, p->nx, p->ny, p->nf);
            wPostPatches[kPost]->data[kp] = p->data[kPrePatch];
         }
      }
   }

   return wPostPatches;
}

void HyPerConn::preSynapticPatchHead(int kxPost, int kyPost, int kfPost, int * kxPre, int * kyPre)
{
   const int xScale = post->clayer->xScale - pre->clayer->xScale;
   const int yScale = post->clayer->yScale - pre->clayer->yScale;
   const float powXScale = powf(2, (float) xScale);
   const float powYScale = powf(2, (float) yScale);

   const int nxPostPatch = (int) (nxp * powXScale);
   const int nyPostPatch = (int) (nyp * powYScale);

   // TODO - does patchHead work in general for post to pre mapping and -scale?
   //      - it seems to work
   int kxPreHead = (int) pvlayer_patchHead((float) kxPost, 0.0, -xScale, (float) nxPostPatch);
   int kyPreHead = (int) pvlayer_patchHead((float) kyPost, 0.0, -yScale, (float) nyPostPatch);

   *kxPre = kxPreHead;
   *kyPre = kyPreHead;
}

int HyPerConn::writePostSynapticWeights(float time, bool last)
{
   int status = 0;
   char path[PV_PATH_MAX];

   const float minVal = minWeight();
   const float maxVal = maxWeight();

   const int numPostPatches = post->clayer->numNeurons;

   const int xScale = post->clayer->xScale - pre->clayer->xScale;
   const int yScale = post->clayer->yScale - pre->clayer->yScale;
   const float powXScale = powf(2, (float) xScale);
   const float powYScale = powf(2, (float) yScale);

   const int nxPostPatch = nxp * powXScale;
   const int nyPostPatch = nyp * powYScale;
   const int nfPostPatch = pre->clayer->numFeatures;

   const char * last_str = (last) ? "_last" : "";
   snprintf(path, PV_PATH_MAX-1, "%s/w%d_post%s.pvp", OUTPUT_PATH, getConnectionId(), last_str);

   const LayerLoc * loc  = &post->clayer->loc;
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
 */
int HyPerConn::randomWeights(PVPatch * wp, float wMin, float wMax, int seed)
{
   pvdata_t * w = wp->data;

   const int nx = (int) wp->nx;
   const int ny = (int) wp->ny;
   const int nf = (int) wp->nf;
   const int nk = nx * ny * nf;

   double p = (wMax - wMin) / RAND_MAX;

   // loop over all post-synaptic cells in patch
   for (int k = 0; k < nk; k++) {
      w[k] = wMin + p * rand();
   }

   return 0;
}

/**
 * calculate gaussian weights to segment lines
 */
int HyPerConn::gauss2DCalcWeights(PVPatch * wp, int kPre, int no, int xScale, int yScale,
                                  int numFlanks, float shift, float rotate,
                                  float aspect, float sigma, float r2Max, float strength)
{
   pvdata_t * w = wp->data;

   const int nx = (int) wp->nx;
   const int ny = (int) wp->ny;
   const int nf = (int) wp->nf;

   if (nx * ny * nf == 0) {
      return 0;  // reduced patch size is zero
   }

   const int sx = (int) wp->sx;  assert(sx == nf);
   const int sy = (int) wp->sy;  // no assert here because patch may be shrunken
   const int sf = (int) wp->sf;  assert(sf == 1);

   //   const float dx = powf(2, xScale);
   //   const float dy = powf(2, yScale);

   // TODO - make sure this is correct
   // sigma is in units of pre-synaptic layer
   const float dx = 1.0;
   const float dy = 1.0;

   const float nxPre = pre->clayer->loc.nx;
   const float nyPre = pre->clayer->loc.ny;
   const float nfPre = pre->clayer->numFeatures;

   const int kxPre = (int) kxPos(kPre, nxPre, nyPre, nfPre);
   const int kyPre = (int) kyPos(kPre, nxPre, nyPre, nfPre);

   // location of pre-synaptic neuron (relative to closest post-synaptic neuron)
   float xPre = -1.0 * deltaPosLayers(kxPre, xScale) * dx;
   float yPre = -1.0 * deltaPosLayers(kyPre, yScale) * dy;

   // closest post-synaptic neuron may not be at the center of the patch (0,0)
   // so must shift pre-synaptic location
   if (xPre < 0.0) xPre += 0.5 * dx;
   if (xPre > 0.0) xPre -= 0.5 * dx;
   if (yPre < 0.0) yPre += 0.5 * dy;
   if (yPre > 0.0) yPre -= 0.5 * dy;

   // (x0,y0) is at upper left corner of patch (i=0,j=0)
   // and shift so pre-synaptic cell is at 0
   const float x0 = -(nx/2.0 - 0.5) * dx - xPre;
   const float y0 = +(ny/2.0 - 0.5) * dy - yPre;

   const float dth = PI/nf;
   const float th0 = rotate*dth/2.0;

   // loop over all post-synaptic cells in patch
   for (int f = 0; f < nf; f++) {
      int o = f % no;
      float th = th0 + o * dth;
      for (int j = 0; j < ny; j++) {
         float y = y0 - j * dy;
         for (int i = 0; i < nx; i++) {
            float x  = x0 + i*dx;

            // rotate the reference frame by th
            float xp = + x * cos(th) + y * sin(th);
            float yp = - x * sin(th) + y * cos(th);

            // include shift to flanks
            float d2 = xp * xp + (aspect*(yp-shift) * aspect*(yp-shift));

            w[i*sx + j*sy + f*sf] = 0;

            // TODO - figure out on/off connectivity
            // don't break it for nfPre==1 going to nfPost=numOrientations
            //if (this->channel == CHANNEL_EXC && f != fPre) continue;
            //if (this->channel == CHANNEL_INH && f == fPre) continue;

            if (d2 <= r2Max) {
               w[i*sx + j*sy + f*sf] = expf(-d2 / (2.0*sigma*sigma));
            }
            if (numFlanks > 1) {
               // shift in opposite direction
               d2 = xp * xp + (aspect*(yp+shift) * aspect*(yp+shift));
               if (d2 <= r2Max) {
                  w[i*sx + j*sy + f*sf] = expf(-d2 / (2.0*sigma*sigma));
               }
            }
            // printf("x=%f y-%f xp=%f yp=%f d2=%f w=%f\n", x, y, xp, yp, d2, w[i*sx + j*sy + f*sf]);
         }
      }
   }

   return 0;
}

PVPatch ** HyPerConn::normalizeWeights(PVPatch ** patches, int numPatches)
{
   PVParams * params = parent->parameters();
   float strength = params->value(name, "strength", 1.0);

   for (int k = 0; k < numPatches; k++) {
      PVPatch * wp = patches[k];
      pvdata_t * w = wp->data;
      const int nx = (int) wp->nx;
      const int ny = (int) wp->ny;
      const int nf = (int) wp->nf;
      for (int f = 0; f < nf; f++) {
         float sum = 0;
         float sum2 = 0;
         for (int i = 0; i < nx * ny; i++) {
            sum += w[f + i * nf];
            sum2 += w[f + i * nf] * w[f + i * nf];
         }
         if (sum == 0.0 && sum2 > 0.0) {
            float factor = strength / sqrt(sum2);
            for (int i = 0; i < nx * ny; i++)
               w[f + i * nf] *= factor;
         }
         else {
            float factor = strength / sum;
            for (int i = 0; i < nx * ny; i++)
               w[f + i * nf] *= factor;
         }
      }
   }
   return patches;
}

int HyPerConn::setPatchSize(const char * filename)
{
   PVParams * inputParams = parent->parameters();

   nxp = inputParams->value(name, "nxp", post->clayer->loc.nx);
   nyp = inputParams->value(name, "nyp", post->clayer->loc.ny);
   nfp = inputParams->value(name, "nfp", post->clayer->numFeatures);

   // use patch dimensions from file if (filename != NULL)
   if (filename != NULL) {
      FILE * fp;
      int numParamsFile;
      int nxpFile, nypFile, nfpFile;
      int fileType;

      fp = pv_open_binary(filename, &numParamsFile, &fileType, &nxpFile, &nypFile, &nfpFile);
      checkWeightsHeader(filename, numParamsFile, nxpFile, nypFile, nfpFile);
      pv_close_binary(fp);
   }
   return 0;
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

} // namespace PV
