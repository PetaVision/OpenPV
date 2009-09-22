/*
 * HyPerConnection.cpp
 *
 *  Created on: Oct 21, 2008
 *      Author: rasmussn
 */

#include "HyPerConn.hpp"
#include "../io/ConnectionProbe.hpp"
#include "../io/io.h"
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

HyPerConn::HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
                     int channel)
         : pre(pre), post(post), parent(hc), channel(channel),
           connId(hc->numberOfConnections()), name(strdup(name))
{
   // Every presynaptic neuron has a weight patch connecting to postsynaptic layer
   // and all postsynaptic border regions.
   // TODO - fix the number of lists
//   this->numAxonalArborLists = 1 + parent->numberOfBorderRegions();
   this->numAxonalArborLists = 1;

   // this is only called from protected constructor by derived classes
   initialize_base();

   // this is virtual but will be HyPerConn version when called from constructor
   initialize();

   hc->addConnection(this);
}

///////
// This constructor is protected so that only derived classes can call it.
// It should be called as the normal method of object construction by
// derived classes.  It should NOT call any virtual methods
//
HyPerConn::HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
                     int channel, int magic_number)
         : pre(pre), post(post), parent(hc), channel(channel),
           connId(hc->numberOfConnections()), name(strdup(name))
{
   assert(magic_number == PROTECTED_NUMBER);
   initialize_base();
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

// base (and private) initializer
int HyPerConn::initialize_base()
{
   this->probes = NULL;
   this->ioAppend = 0;

   this->numProbes = 0;

   // STDP parameters for modifying weights
   this->stdpFlag = 0;
   this->ampLTP = 1.0;
   this->ampLTD = 1.1;
   this->tauLTP = 20;
   this->tauLTD = 20;
   this->dWMax  = 0.1;
   this->wMax   = 1.0;

   this->wPostTime    = -1.0;
   this->wPostPatches = NULL;

   assert(this->channel <= post->clayer->numPhis);

   PVParams * inputParams = parent->parameters();
   nxp = inputParams->value(name, "nxp");
   nyp = inputParams->value(name, "nyp");

   nfp = post->clayer->numFeatures;

   if (inputParams->present(name, "strength")) {
      this->wMax = inputParams->value(name, "strength");
   }
   // let wMax override strength if user provides it
   if (inputParams->present(name, "wMax")) {
      this->wMax = inputParams->value(name, "wMax");
   }

   if (inputParams->present(name, "stdpFlag")) {
      stdpFlag = (int) inputParams->value(name, "stdpFlag");
   }

   return 0;
}

// virtual initializer
int HyPerConn::initialize()
{
   int status = 0;

   PVParams * inputParams = parent->parameters();
   setParams(inputParams, &defaultConnParams);
   assert(params->delay < MAX_F_DELAY);

   params->numDelay = params->varDelayMax - params->varDelayMin + 1;

   createWeights(nxp, nyp, nfp);

   // TODO - create STDP variables for border regions
   if (stdpFlag) {
      int numPatches = numberOfWeightPatches(0);
      pIncr = createWeights(nxp, nyp, nfp, numPatches);
      pDecr = pvcube_new(&post->clayer->loc, post->clayer->numNeurons);
   }
   else {
      pIncr = NULL;
      pDecr = NULL;
   }

   // create list of axonal arbors containing pointers to {phi,w,P,M} patches
   createAxonalArbors();

   if (stdpFlag) {
      for (int k = 0; k < numAxonalArborLists; k++) {
         pvdata_t * p = pIncr[k]->data;
         for (int i = 0; i < nxp*nyp*nfp; i++) {
            p[i] = 0;
         }
      }
      for (int k = 0; k < pDecr->numItems; k++) {
         pDecr->data[k] = 0.0;
      }
   }

   const char * filename = NULL;
   status = -1;

   if (filename != NULL) {
      status = readWeights(filename);
   }

   if (status != 0) {
      char name[PV_PATH_MAX];
      snprintf(name, PV_PATH_MAX-1, "%s/w%1.1d_last.bin",
                     OUTPUT_PATH, getConnectionId());
      status = this->readWeights(name);
   }

   if (status != 0) {
      PVParams * params = parent->parameters();

      float randomFlag = 0;
      if (params->present(getName(), "randomFlag")) {
         randomFlag = params->value(getName(), "randomFlag");
      }
      if (randomFlag > 0) {
         float seed = 0;
         if (params->present(getName(), "randomSeed")) {
            seed = params->value(getName(), "randomFlag");
         }
         status = initializeRandomWeights(seed);
      }
      else {
         status = initializeGaussianWeights();
      }
   }

   status = adjustAxonalArborWeights();

   return status;
}

int HyPerConn::setParams(PVParams * filep, PVConnParams * p)
{
   const char * name = getName();

   params = (PVConnParams *) malloc(sizeof(*p));
   memcpy(params, p, sizeof(*p));

   numParams = sizeof(*p) / sizeof(float);
   assert(numParams == 9);        // catch changes in structure

   if (filep->present(name, "delay"))    params->delay    = (int) filep->value(name, "delay");
   if (filep->present(name, "fixDelay")) params->fixDelay = (int) filep->value(name, "fixDelay");
   if (filep->present(name, "vel"))      params->vel      = filep->value(name, "vel");
   if (filep->present(name, "rmin"))     params->rmin     = filep->value(name, "rmin");
   if (filep->present(name, "rmax"))     params->rmax     = filep->value(name, "rmax");

   if (filep->present(name, "varDelayMin")) params->varDelayMin = (int) filep->value(name, "varDelayMin");
   if (filep->present(name, "varDelayMax")) params->varDelayMax = (int) filep->value(name, "varDelayMax");
   if (filep->present(name, "numDelay"))    params->numDelay    = (int) filep->value(name, "numDelay");
   if (filep->present(name, "isGraded"))    params->isGraded    = (int) filep->value(name, "isGraded");

   return 0;
}

int HyPerConn::initializeRandomWeights(int seed)
{
   PVParams * params = parent->parameters();

   float wMin = 0.0;
   if (params->present(name, "wMin")) wMin = params->value(name, "wMin");

   for (int n = 0; n < 1 + parent->numberOfBorderRegions(); n++) {
      int numPatches = numberOfWeightPatches(n);
      for (int k = 0; k < numPatches; k++) {
         randomWeights(wPatches[n][k], wMin, wMax, seed);
      }
   }

   return 0;
}

int HyPerConn::initializeGaussianWeights()
{
   PVParams * params = parent->parameters();

   // default values (chosen for center on cell of one pixel)

   int noPost = 1;
   if (params->present(post->getName(), "no")) {
      noPost = (int) params->value(post->getName(), "no");
   }

   float aspect   = 1.0;  // circular (not line oriented)
   float sigma    = 0.8;
   float rMax     = 1.4;
   float strength = 1.0;

   aspect = params->value(name, "aspect");
   sigma  = params->value(name, "sigma");
   rMax   = params->value(name, "rMax");
   if (params->present(name, "strength")) {
      strength = params->value(name, "strength");
   }

   float r2Max = rMax * rMax;

   int numFlanks = 1;
   float shift   = 0.0;
   float rotate  = 1.0;  // rotate so that axis isn't aligned

   if (params->present(name, "numFlanks"))  numFlanks = (int) params->value(name, "numFlanks");
   if (params->present(name, "flankShift")) shift     = params->value(name, "flankShift");
   if (params->present(name, "rotate"))     rotate    = params->value(name, "rotate");

   const int xScale = post->clayer->xScale - pre->clayer->xScale;
   const int yScale = post->clayer->yScale - pre->clayer->yScale;
   for (int n = 0; n < numAxonalArborLists; n++) {
      int numPatches = numberOfWeightPatches(n);
      for (int k = 0; k < numPatches; k++) {
         int kPre = kIndexFromNeighbor(k, n);
         gauss2DCalcWeights(wPatches[n][k], kPre, noPost, xScale, yScale,
                            numFlanks, shift, rotate, aspect, sigma, r2Max, strength);
      }
   }

   return 0;
}

int HyPerConn::readWeights(const char * filename)
{
   FILE * fp;
   int numParams, nxIn, nyIn, nfIn;
   int params[MAX_BIN_PARAMS];
   int status = 0;

   fp = pv_open_binary(filename, &numParams, &nxIn, &nyIn, &nfIn);

   if (fp == NULL) {
      return -1;
   }

   assert(nxIn == nxp);
   assert(nyIn == nyp);
   assert(nfIn == nfp);

   pv_read_binary_params(fp, numParams, params);
   int wMaxIn = params[numParams-2];
   int numPatchesIn = params[numParams-1];

   int arbor = 0;
   int numPatches = this->numberOfWeightPatches(arbor);
   assert(numPatches == numPatchesIn);
   this->wMax = wMaxIn;

   status = pv_read_patches(fp, nfp, 0.0, wMax, numPatches, wPatches[arbor]);
   assert(status == 0);

   status = pv_close_binary(fp);

   return status;
}

int HyPerConn::writeWeights(float time)
{
   int status = 0;
   char name[PV_PATH_MAX];

   if (time == FINAL_TIME) {
      snprintf(name, PV_PATH_MAX-1, "w%d_last", getConnectionId());
   }
   else {
      snprintf(name, PV_PATH_MAX-1, "w%d", getConnectionId());
   }

   int arbor = 0;
   int numPatches = numberOfWeightPatches(arbor);
   status = pv_write_patches(name, false, (int) nxp, (int) nyp, (int) nfp,
                             0.0, wMax, numPatches, wPatches[arbor]);
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
   pv_tiff_write_patch(fd, wPatches[arbor][0]);
   fclose(fd);

   sprintf(outfile, "w%d.txt", getConnectionId());
   err = writeWeights(outfile, 0);

#endif

   return status;
}

int HyPerConn::writeWeights(int k)
{
   return writeWeights(NULL, k);
}

int HyPerConn::writeWeights(const char * filename, int k)
{
   FILE * fd;
   char outfile[PV_PATH_MAX];

   if (filename != NULL) {
      sprintf(outfile, "%s%s", OUTPUT_PATH, filename);
      fd = fopen(outfile, "w");
      if (fd == NULL) {
         fprintf(stderr, "writeWeights: ERROR opening file %s\n", filename);
         return 1;
      }
   }
   else {
      fd = stdout;
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

int HyPerConn::deliver(PVLayerCube * cube, int neighbor)
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

   for (int i = 0; i < numProbes; i++) {
      tmp[i] = probes[i];
   }
   delete probes;

   probes = tmp;
   probes[numProbes] = p;

   return ++numProbes;
}

int HyPerConn::outputState(float time)
{
   int status = 0;

   for (int i = 0; i < numProbes; i++) {
      probes[i]->outputState(time, this);
   }

   if (time == FINAL_TIME) {
      status = writeWeights(time);
      assert(status == 0);
   }
   else if (stdpFlag)
   {
      status = writeWeights(time);
      assert(status == 0);

      convertPreSynapticWeights(time);
      status = writePostSynapticWeights(ioAppend);
      assert(status == 0);
   }

   // append to output file after original open
   ioAppend = 1;

   return status;
}

int HyPerConn::updateState(float time, float dt)
{
   if (stdpFlag) {
      const float fac = ampLTD;
      const float decay = expf(-dt/tauLTD);

      float * a = post->clayer->activity->data;
      float * m = pDecr->data;            // decrement (minus) variable
      int nk = pDecr->numItems;

      for (int k = 0; k < nk; k++) {
         m[k] = decay * m[k] - fac * a[k];
         if (a[k] > 0) {
//            fprintf(stderr, "k=%d, m=%f addr(m)=%p\n", k, m[k], &m[k]);
         }
      }
   }

   return 0;
}

int HyPerConn::updateWeights(PVLayerCube * preActivityCube, int neighbor)
{
   const float dt = parent->getDeltaTime();
   const float decayLTP = expf(-dt/tauLTP);

   // assume pDecr has been updated already, and weights have been used, so
   // 1. update Psij (pIncr) for each synapse
   // 2. update wij

   // TODO - handle neighbors
   if (neighbor != 0) {
      return 0;
   }

   // TODO - what is happening here
   if (preActivityCube->numItems == 0) return 0;

   const int numNeurons = preActivityCube->numItems;
   assert(numNeurons == pre->clayer->numNeurons);

   for (int kPre = 0; kPre < numNeurons; kPre++) {
      PVAxonalArbor * arbor  = axonalArbor(kPre, neighbor);

      const float preActivity = preActivityCube->data[kPre];

      PVPatch * pIncr = arbor->plasticIncr;
      PVPatch * w     = arbor->weights;
      size_t offset   = arbor->offset;

      float * postActivity = &post->clayer->activity->data[offset];
      float * M = &pDecr->data[offset];  // STDP decrement variable
      float * P =  pIncr->data;          // STDP increment variable

      int nk  = (int) pIncr->nf * (int) pIncr->nx;
      int ny  = (int) pIncr->ny;
      int sy  = (int) pIncr->sy;

      // TODO - unroll

      // update Psij (pIncr variable)
      for (int y = 0; y < ny; y++) {
         pvpatch_update_plasticity_incr(nk, pIncr->data + y*sy, preActivity, decayLTP, ampLTP);
      }

      // update weights
      for (int y = 0; y < ny; y++) {
         int yOff = y*sy;
         pvpatch_update_weights(nk, w->data + yOff, M + yOff, P + yOff,
                                preActivity, postActivity + yOff, dWMax, wMax);
      }
   }

   return 0;
}

/**
 * returns the number of weight patches for the given neighbor
 * @param neighbor the id of the neighbor (0 for interior/self)
 */
int HyPerConn::numberOfWeightPatches(int arbor)
{
   int neighbor = arbor; // for now there is one axonal arbor per neighbor
   return pre->numberOfNeurons(neighbor);
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

int HyPerConn::createWeights(int nxPatch, int nyPatch, int nfPatch)
{
   for (int n = 0; n < numAxonalArborLists; n++) {
      // for STDP, an axonal arbor (patch) is required for every presynaptic neuron
      int numPatches = numberOfWeightPatches(n);
      wPatches[n] = createWeights(nxp, nyp, nfp, numPatches);
   }
   return 0;
}

/**
 * Create a separate patch of weights for every neuron
 */
PVPatch ** HyPerConn::createWeights(int nxPatch, int nyPatch, int nfPatch, int numPatches)
{
   // could create only a single patch with following call
   //   return createPatches(numBundles, nxp, nyp, nfp);

   PVPatch ** patches = (PVPatch**) calloc(numPatches*sizeof(PVPatch*), sizeof(char));

   // TODO - allocate space for them all at once
   for (int k = 0; k < numPatches; k++) {
      patches[k] = pvpatch_inplace_new(nxPatch, nyPatch, nfPatch);
   }

   return patches;
}

int HyPerConn::deleteWeights()
{
   // to be used if createPatches is used above
   // HyPerConn::deletePatches(numBundles, wPatches);

   for (int arbor = 0; arbor < numAxonalArborLists; arbor++) {
      int numPatches = numberOfWeightPatches(arbor);
      for (int k = 0; k < numPatches; k++) {
         pvpatch_delete(wPatches[arbor][k]);
      }
      free(wPatches[arbor]);
   }

   if (wPostPatches != NULL) {
      const int numPostNeurons = post->clayer->numNeurons;
      for (int k = 0; k < numPostNeurons; k++) {
         pvpatch_inplace_delete(wPostPatches[k]);
      }
      free(wPostPatches);
   }

   if (stdpFlag) {
      int numPatches = numberOfWeightPatches(0);
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

   const float nxBorderPost = post->clayer->loc.nPad;

   const int xScale = post->clayer->xScale - pre->clayer->xScale;
   const int yScale = post->clayer->yScale - pre->clayer->yScale;

   const int numNeighbors = numAxonalArborLists;

#ifndef FEATURES_LAST
   const float psf = 1;
   const float psx = nfp;
   const float psy = psx * (nxPost + 2.0f*nxBorderPost);
#else
   const float psx = 1;
   const float psy = nxPost + 2.0f*nxBorderPost;
   const float psf = psy * (nyPost + 2.0f*nxBorderPost);
#endif

   for (int n = 0; n < numNeighbors; n++) {
      int numArbors = pre->numberOfNeurons(n);
      axonalArborList[n] = (PVAxonalArbor*) calloc(numArbors, sizeof(PVAxonalArbor));
   }

   // there is an arbor list for every neighbor
   for (int n = 0; n < numNeighbors; n++) {
      int numArbors = pre->numberOfNeurons(n);
      PVPatch * dataPatches = (PVPatch *) calloc(numArbors, sizeof(PVPatch));

      for (int k = 0; k < numArbors; k++) {
         PVAxonalArbor * arbor = axonalArbor(k, n);
         int kPre = kIndexFromNeighbor(k, n);

         // global indices
         float kxPre = kx0Pre + kxPos(kPre, nxPre, nyPre, nfPre);
         float kyPre = ky0Pre + kyPos(kPre, nxPre, nyPre, nfPre);

         // global indices
         float kxPost = pvlayer_patchHead(kxPre, kxPost0Left, xScale, nxp);
         float kyPost = pvlayer_patchHead(kyPre, kyPost0Left, yScale, nyp);

         // TODO - can get nf from weight patch but what about kf0?
         // weight patch is actually a pencil and so kfPost is always 0?
         float kfPost = 0.0f;

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

         // local index
         int kl = kIndex(kxPost, kyPost, kfPost, nxPost, nyPost, nfPost);
         assert(kl >= 0);
         assert(kl < post->clayer->numNeurons);

         arbor->data = &dataPatches[k];  // patches allocated above to fit border region
         arbor->weights = this->getWeights(kPre, n);
         arbor->plasticIncr = this->getPlasticityIncrement(kPre, n);
         arbor->offset = kl;

         // initialize
         pvdata_t * phi = post->clayer->phi[channel] + kl;
         pvpatch_init(arbor->data, nxPatch, nyPatch, nfp, psx, psy, psf, phi);

      } // loop over arbors (pre-synaptic neurons in neighbor region)
   } // loop over neighbors

   return 0;
}

int HyPerConn::adjustAxonalArborWeights()
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

   const int xScale = post->clayer->xScale - pre->clayer->xScale;
   const int yScale = post->clayer->yScale - pre->clayer->yScale;

   const int numNeighbors = numAxonalArborLists;

   for (int n = 0; n < numNeighbors; n++) {
      int numArbors = pre->numberOfNeurons(n);

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
   const float powXScale = powf(2, (float) xScale);
   const float powYScale = powf(2, (float) yScale);

   const int nxPre  = (int) pre->clayer->loc.nx;
   const int nyPre  = (int) pre->clayer->loc.ny;
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
      wPostPatches = createWeights(nxPostPatch, nyPostPatch, nfPostPatch, numPost);
   }

   // loop through post-synaptic neurons

   for (int kPost = 0; kPost < numPost; kPost++) {
      int kxPost = (int) kxPos(kPost, nxPost, nyPost, nfPost);
      int kyPost = (int) kyPos(kPost, nxPost, nyPost, nfPost);
      int kfPost = (int) featureIndex(kPost, nxPost, nyPost, nfPost);

      // TODO - does patchHead work in general for post to pre mapping and -scale?
      int kxPreHead = (int) pvlayer_patchHead((float) kxPost, 0.0, -xScale, (float) nxPostPatch);
      int kyPreHead = (int) pvlayer_patchHead((float) kyPost, 0.0, -yScale, (float) nyPostPatch);

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
            int kPrePatch = kIndex(kxPrePatch, kyPrePatch, kfPost, p->nx, p->ny, p->nf);
            wPostPatches[kPost]->data[kp] = p->data[kPrePatch];
         }
      }
   }

   return wPostPatches;
}

int HyPerConn::writePostSynapticWeights(int ioAppend)
{
   char txtfile[PV_PATH_MAX];
   char poststr[PV_PATH_MAX];

   int status = 0;

   FILE * fp = NULL;

   const int numPost   = post->clayer->numNeurons;
   const int nxPost  = post->clayer->loc.nx;
   const int nyPost  = post->clayer->loc.ny;
   const int nfPost  = post->clayer->numFeatures;

   const int xScale = post->clayer->xScale - pre->clayer->xScale;
   const int yScale = post->clayer->yScale - pre->clayer->yScale;
   const float powXScale = powf(2, (float) xScale);
   const float powYScale = powf(2, (float) yScale);

   const int nxPostPatch = nxp * powXScale;
   const int nyPostPatch = nyp * powYScale;
   const int nfPostPatch = pre->clayer->numFeatures;

   // the number of features is the end-point value (normally post-synaptic)
   const int numPostPatch = nxPostPatch * nyPostPatch * nfPostPatch;

   snprintf(txtfile, PV_PATH_MAX-1, "%sw%d_post.txt", OUTPUT_PATH, getConnectionId());
   snprintf(poststr, PV_PATH_MAX-1, "w%d_post", getConnectionId());

   status = pv_write_patches(poststr, ioAppend, nxPostPatch, nyPostPatch, nfPostPatch,
                             0.0, wMax, numPost, wPostPatches);

   if (!ioAppend) {
      fp = fopen(txtfile, "w");

      if (fp != NULL) {
         fprintf(fp, "nxPost==%d nyPost==%d nfPost==%d numPostPatch==%d\n",
                     nxPost, nyPost, nfPost, numPostPatch);
      }
   }
   else {
      fp = fopen(txtfile, "a");
   }

   if (fp != NULL) {
      for (int i = 0; i < numPost; i++) {
         if (wPostPatches[i] == NULL) {
            fprintf(stderr, "Post-patch weights are NULL\n");
            fclose(fp);
            return -1;
         }
         for (int j = 0; j < numPostPatch; j++) {
            fprintf(fp, "%f ", wPostPatches[i]->data[j]);
         }
         fprintf(fp, "\n");
      }
      fclose(fp);
   }
   else {
      fprintf(stderr, "Error opening file: Cannot write post-patch weights.\n");
      return -1;
   }

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

   const int sx = (int) wp->sx;  assert(sx == nf);
   const int sy = (int) wp->sy;  assert(sy == nf*nx);
   const int sf = (int) wp->sf;  assert(sf == 1);

   //   const float dx = powf(2, xScale);
   //   const float dy = powf(2, yScale);

   // TODO - make sure this is correct
   // sigma is in units of pre-synaptic layer
   const float dx = 1.0;
   const float dy = 1.0;

   const float nxPre = pre->clayer->loc.nx;
   const float nyPre = pre->clayer->loc.nx;
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

   // normalize
   for (int f = 0; f < nf; f++) {
      float sum = 0;
      for (int i = 0; i < nx*ny; i++) sum += w[f + i*nf];

      if (sum == 0.0) continue;  // all weights == zero is ok //should this be here?

      float factor = strength/sum;
      for (int i = 0; i < nx*ny; i++) w[f + i*nf] *= factor;
      //for (int i = 0; i < nx*ny; i++) w[f + i*nf] = kPre; //used for testing, bug-checking
   }

   return 0;
}

}
