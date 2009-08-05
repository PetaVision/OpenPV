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

#include <iostream>
#include <fstream>

using namespace std;

namespace PV {

// default values

PVConnParams defaultConnParams =
{
   /*delay*/ 0, /*fixDelay*/ 0, /*varDelayMin*/ 0, /*varDelayMax*/ 0, /*numDelay*/ 1,
   /*isGraded*/ 0, /*vel*/ 45.248, /*rmin*/ 0.0, /*rmax*/ 4.0
};

HyPerConn::HyPerConn()
{
   this->connId = 0;
   this->pre    = NULL;
   this->post   = NULL;
   this->probes = NULL;
   this->pvconn = NULL;
   this->name   = strdup("Unknown");
   this->channel   = 0;
   this->stdpFlag  = 0;
   this->numProbes = 0;
   this->ioAppend = 0;
}

HyPerConn::HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post)
{
   this->connId = hc->numberOfConnections();
   this->name   = strdup(name);
   this->parent = hc;

   // default for this connection is 1 weight patch
   this->numBundles = 1;

   initialize(NULL, pre, post, CHANNEL_EXC);

   hc->addConnection(this);
}

HyPerConn::HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
                     int channel)
{
   this->connId = hc->numberOfConnections();
   this->name   = strdup(name);
   this->parent = hc;

   // default for this connection is 1 weight patch
   this->numBundles = 1;

   initialize(NULL, pre, post, channel);

   hc->addConnection(this);
}

HyPerConn::HyPerConn(const char * name, int argc, char ** argv, HyPerCol * hc,
                     HyPerLayer * pre, HyPerLayer * post)
{
   char * filename = NULL;

   this->connId = hc->numberOfConnections();
   this->name   = strdup(name);
   this->parent = hc;

   for (int i = 1; i < argc; i++) {
      char * arg = argv[i];
      if (strcmp(arg, "-f") == 0) {
         if (++i < argc) {
            filename = argv[i++];
         }
      }

      if (filename != NULL) {
         // TODO - strip off args as used
         break;
      }
   }

   // default for this connection is 1 weight patch
   this->numBundles = 1;

   initialize(filename, pre, post, CHANNEL_EXC);

   hc->addConnection(this);
}


HyPerConn::HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
                     const char * filename)
{
   this->connId = hc->numberOfConnections();
   this->name   = strdup(name);
   this->parent = hc;

   // default for this connection is 1 weight patch
   this->numBundles = 1;

   initialize(filename, pre, post, CHANNEL_EXC);

   hc->addConnection(this);
}

HyPerConn::~HyPerConn()
{
   assert(params != NULL);
   free(params);

   if (pvconn != NULL) {
      pvConnFinalize(pvconn);
      delete pvconn;
   }

   if (name != NULL) free(name);

   deleteWeights();

   // free the task information

   free(bundles[0]->tasks[0]->data);   // allData
   free(bundles[0]->tasks[0]);         // allTasks
   free(bundles[0]->tasks);            // allTPtrs
   free(bundles[0]);                   // allLists
   free(bundles);                      // allLPtrs
}

int HyPerConn::initialize(const char * filename, HyPerLayer * pre, HyPerLayer * post,
                          int channel)
{
   this->pre = pre;
   this->post = post;
   this->probes = NULL;
   this->channel = channel;
   this->ioAppend = 0;
   this->pvconn = new PVConnection;

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

   setParams(inputParams, &defaultConnParams);
   assert(params->delay < MAX_F_DELAY);

   params->numDelay = params->varDelayMax - params->varDelayMin + 1;

   if (inputParams->present(name, "strength")) {
      this->wMax = inputParams->value(name, "strength");
   }
   // let wMax override strength if user provides it
   if (inputParams->present(name, "wMax")) {
      this->wMax = inputParams->value(name, "wMax");
   }

   pvConnInit(pvconn, pre->clayer, post->clayer, params, channel);

   if (inputParams->present(name, "stdpFlag")) {
      stdpFlag = inputParams->value(name, "stdpFlag");
   }


   const int numPatches = numberOfWeightPatches();

   wPatches = createWeights(nxp, nyp, nfp, numPatches);
   if (stdpFlag) {
      pIncr = createWeights(nxp, nyp, nfp, numPatches);
      pDecr = pvcube_new(&post->clayer->loc, post->clayer->numNeurons);
   }
   else {
      pIncr = NULL;
      pDecr = NULL;
   }

   // create synapse bundles
   createSynapseBundles(numBundles);

   // temporarily set weights to 0 as default

   for (int k = 0; k < numBundles; k++) {
      pvdata_t * w = wPatches[k]->data;
      for (int i = 0; i < nxp*nyp*nfp; i++) {
         w[i] = 0;
      }
   }
   if (stdpFlag) {
      for (int k = 0; k < numBundles; k++) {
         pvdata_t * p = pIncr[k]->data;
         for (int i = 0; i < nxp*nyp*nfp; i++) {
            p[i] = 0;
         }
      }
      for (int k = 0; k < pDecr->numItems; k++) {
         pDecr->data[k] = 0.0;
      }
   }

   initializeWeights(filename);
   writeWeights();
   adjustWeightBundles(numBundles);

   return 0;
}

int HyPerConn::setParams(PVParams * filep, PVConnParams * p)
{
   const char * name = getName();

   params = (PVConnParams *) malloc(sizeof(*p));
   memcpy(params, p, sizeof(*p));

   numParams = sizeof(*p) / sizeof(float);
   assert(numParams == 9);        // catch changes in structure

   if (filep->present(name, "delay"))    params->delay    = filep->value(name, "delay");
   if (filep->present(name, "fixDelay")) params->fixDelay = filep->value(name, "fixDelay");
   if (filep->present(name, "vel"))      params->vel      = filep->value(name, "vel");
   if (filep->present(name, "rmin"))     params->rmin     = filep->value(name, "rmin");
   if (filep->present(name, "rmax"))     params->rmax     = filep->value(name, "rmax");

   if (filep->present(name, "varDelayMin")) params->varDelayMin = filep->value(name, "varDelayMin");
   if (filep->present(name, "varDelayMax")) params->varDelayMax = filep->value(name, "varDelayMax");
   if (filep->present(name, "numDelay"))    params->numDelay    = filep->value(name, "numDelay");
   if (filep->present(name, "isGraded"))    params->isGraded    = filep->value(name, "isGraded");

   return 0;
}

int HyPerConn::initializeRandomWeights(int seed)
{
   PVParams * params = parent->parameters();

   float wMin = 0.0;
   if (params->present(name, "wMin")) wMin = params->value(name, "wMin");

   const int numPatches = numberOfWeightPatches();
   for (int k = 0; k < numPatches; k++) {
      randomWeights(wPatches[k], wMin, wMax, seed);
   }

   return 0;
}

int HyPerConn::initializeWeights(const char * filename)
{
   if (filename == NULL) {
      PVParams * params = parent->parameters();

      float randomFlag = 0;
      if (params->present(getName(), "randomFlag")) {
         randomFlag = params->value(getName(), "randomFlag");
      }
      if (randomFlag > 0) {
         return initializeRandomWeights(0);
      }

      // default values (chosen for center on cell of one pixel)

      int noPost = 1;
      if (params->present(post->getName(), "no")) {
         noPost = params->value(post->getName(), "no");
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

      if (params->present(name, "numFlanks"))  numFlanks = params->value(name, "numFlanks");
      if (params->present(name, "flankShift")) shift     = params->value(name, "flankShift");
      if (params->present(name, "rotate"))     rotate    = params->value(name, "rotate");

      const int numPatches = numberOfWeightPatches();
      const int xScale = post->clayer->xScale - pre->clayer->xScale;
      const int yScale = post->clayer->yScale - pre->clayer->yScale;
      for (int k = 0; k < numPatches; k++) {
         gauss2DCalcWeights(wPatches[k], k, noPost, xScale, yScale,
                            numFlanks, shift, rotate, aspect, sigma, r2Max, strength);
      }
   }
   else {
      int err = 0;
      size_t size, count, dim[3];

       // TODO rank 0 should read and distribute file

      FILE * fd = fopen(filename, "rb");

      if (fd == NULL) {
         fprintf(stderr,
                 "FileConn:: couldn't open file %s, using 8x8 weights = 1\n",
                  filename);
         return -1;
      }

      if ( fread(dim,    sizeof(size_t), 3, fd) != 3 ) err = 1;
      if ( fread(&size,  sizeof(size_t), 1, fd) != 1 ) err = -1;
      if ( fread(&count, sizeof(size_t), 1, fd) != 1 ) err = -1;

      // check for consistency

      if (dim[0] != (size_t) nfp) err = -1;
      if (dim[1] != (size_t) nxp) err = -1;
      if (dim[2] != (size_t) nyp) err = -1;
      if ((int) count != numBundles) err = -1;
      if (size  != sizeof(PVPatch) + nxp*nyp*nfp*sizeof(float) ) err = -1;

      if (err) {
         fprintf(stderr, "FileConn:: ERROR: difference in dim, size or count of patches\n");
         return err;
      }

      for (unsigned int i = 0; i < count; i++) {
         PVPatch* patch = wPatches[i];
         if ( fread(patch, size, 1, fd) != 1) {
            fprintf(stderr, "FileConn:: ERROR reading patch %d\n", i);
            return -1;
         }
         // TODO fix address with a function
         patch->data = (pvdata_t*) ((char*) patch + sizeof(PVPatch));
      }

      // TODO - adjust strides sy, sf in weight patches
   } // end if for filename

   return 0;
}

int HyPerConn::writeWeights()
{
   int err = 0;

#ifdef DEBUG_WEIGHTS
   char outfile[128];

   // only write first weight patch

   sprintf(outfile, "%sw%d.tif", OUTPUT_PATH, getConnectionId());
   FILE * fd = fopen(outfile, "wb");
   if (fd == NULL) {
      fprintf(stderr, "writeWeights: ERROR opening file %s\n", outfile);
      return 1;
   }
   pv_tiff_write_patch(fd, wPatches[0]);
   fclose(fd);

   sprintf(outfile, "w%d.txt", getConnectionId());
   err = writeWeights(outfile, 0);

#endif

   return err;
}

int HyPerConn::writeWeights(int k)
{
   return writeWeights(NULL, k);
}

int HyPerConn::writeWeights(const char * filename, int k)
{
   FILE * fd;
   char outfile[128];

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
   pv_text_write_patch(fd, wPatches[k]);
   fprintf(fd, "----------------------------\n");

   if (fd != stdout) {
      fclose(fd);
   }

   return 0;
}

int HyPerConn::deliver(PVLayerCube * cube, int neighbor)
{
   post->recvSynapticInput(this, cube, neighbor);
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
   int err = 0;

   for (int i = 0; i < numProbes; i++) {
      probes[i]->outputState(time, this);
   }

   if (stdpFlag) {
      // Output weights
      char str[32];
      sprintf(str, "w%1.1d", getConnectionId());
      err = pv_write_patches(str, ioAppend, (int) nxp, (int) nyp, (int) nfp,
                             0.0, wMax, numberOfWeightPatches(), wPatches);
      assert(err == 0);

      err = writePostPatchWeights(ioAppend);
      assert(err == 0);
   }

   // append to dump file after original open
   ioAppend = 1;

   return err;
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
      PVSynapseBundle * tasks = this->tasks(kPre, neighbor);

      const float preActivity = preActivityCube->data[kPre];

      for (unsigned int i = 0; i < tasks->numTasks; i++) {
         PVPatch * pIncr = tasks->tasks[i]->plasticIncr;
         PVPatch * w     = tasks->tasks[i]->weights;
         size_t offset   = tasks->tasks[i]->offset;

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
   }

   return 0;
}

int HyPerConn::numberOfWeightPatches()
{
   // TODO - fix for bundles
   assert(numBundles == 1);
   return pre->clayer->numNeurons;
}

PVPatch * HyPerConn::getWeights(int k, int bundle)
{
   // TODO - make work with bundles as well
   assert(numBundles == 1);
   // a separate patch of weights for every neuron
   return wPatches[k];
}

PVPatch * HyPerConn::getPlasticityIncrement(int k, int bundle)
{
   // TODO - make work with bundles as well
   assert(numBundles == 1);
   // a separate patch of plasticity for every neuron
   if (stdpFlag) {
      return pIncr[k];
   }
   return NULL;
}

/**
 * Create a separate patch of weights for every neuron
 */
PVPatch ** HyPerConn::createWeights(int nxPatch, int nyPatch, int nfPatch, int numPatches)
{
   // could create only a single patch with following call
   //   return createPatches(numBundles, nxp, nyp, nfp);

   assert(numBundles == 1);

   PVPatch ** patches = (PVPatch**) malloc(numPatches*sizeof(PVPatch*));

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

   assert(numBundles == 1);

   const int numPreNeurons = pre->clayer->numNeurons;

   for (int k = 0; k < numPreNeurons; k++) {
      pvpatch_inplace_delete(wPatches[k]);
   }
   free(wPatches);

   if (wPostPatches != NULL) {
      const int numPostNeurons = post->clayer->numNeurons;
      for (int k = 0; k < numPostNeurons; k++) {
         pvpatch_inplace_delete(wPostPatches[k]);
      }
      free(wPostPatches);
   }

   if (stdpFlag) {
      for (int k = 0; k < numPreNeurons; k++) {
         pvpatch_inplace_delete(pIncr[k]);
      }
      free(pIncr);
      pvcube_delete(pDecr);
   }

   return 0;
}

int HyPerConn::createSynapseBundles(int numTasks)
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

   const float xScale = post->clayer->xScale - pre->clayer->xScale;
   const float yScale = post->clayer->yScale - pre->clayer->yScale;

   const float numBorder = post->clayer->numBorder;
   assert(numBorder == 0);

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
   int nTotalTasks = nNeurons * numTasks;

   PVSynapseBundle ** allBPtrs   = (PVSynapseBundle**) malloc(nNeurons*sizeof(PVSynapseBundle*));
   PVSynapseBundle  * allBundles = (PVSynapseBundle *) malloc(nNeurons*sizeof(PVSynapseBundle));
   PVSynapseTask** allTPtrs = (PVSynapseTask**) malloc(nTotalTasks*sizeof(PVSynapseTask*));
   PVSynapseTask * allTasks = (PVSynapseTask*)  malloc(nTotalTasks*sizeof(PVSynapseTask));
   PVPatch       * allData  = (PVPatch*)        malloc(nTotalTasks*sizeof(PVPatch));

   bundles = allBPtrs;

   for (int kPre = 0; kPre < nNeurons; kPre++) {
      int offset = kPre*sizeof(PVSynapseBundle);
      bundles[kPre] = (PVSynapseBundle*) ((char*) allBundles + offset);
      bundles[kPre]->numTasks = numTasks;
      bundles[kPre]->tasks = allTPtrs + kPre*numTasks;

      PVSynapseBundle * list = bundles[kPre];
      for (int i = 0; i < numTasks; i++) {
         offset = (i + kPre*numTasks) * sizeof(PVSynapseTask);
         list->tasks[i] = (PVSynapseTask*) ((char*) allTasks + offset);
      }

      for (int i = 0; i < numTasks; i++) {
         PVSynapseTask * task = list->tasks[i];

         task->weights = this->getWeights(kPre, i);
         task->plasticIncr = this->getPlasticityIncrement(kPre, i);

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

         pvdata_t * phi = post->clayer->phi[channel] + kl;
         task->offset = kl;

         // TODO - shouldn't kPre vary first?
//         offset = (i + kPre*numTasks) * sizeof(PVPatch);
         offset = (kPre + i*numTasks) * sizeof(PVPatch);
         task->data = (PVPatch*) ((char*) allData + offset);

         pvpatch_init(task->data, nxPatch, nyPatch, nfp, psx, psy, psf, phi);
      }
   }

   return 0;
}

int HyPerConn::adjustWeightBundles(int numTasks)
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

   const float xScale = post->clayer->xScale - pre->clayer->xScale;
   const float yScale = post->clayer->yScale - pre->clayer->yScale;

   const float numBorder = post->clayer->numBorder;
   assert(numBorder == 0);

   int nNeurons = pre->clayer->numNeurons;

   for (int kPre = 0; kPre < nNeurons; kPre++) {
      for (int i = 0; i < numTasks; i++) {
         PVSynapseTask * task = bundles[kPre]->tasks[i];

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

         pvpatch_adjust(task->weights, nxPatch, nyPatch, dx, dy);
         if (stdpFlag) {
            task->offset += dx * task->weights->sx + dy * task->weights->sy;
            pvpatch_adjust(task->plasticIncr, nxPatch, nyPatch, dx, dy);
         }
      }
   }

   return 0;
}

#ifdef USE_OLD_CODE
// this version has borders in it, need to keep until sure that image doesn't needs to
// have border
int HyPerConn::createSynapseBundles(int numTasks)
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

   const float xScale = post->clayer->xScale - pre->clayer->xScale;
   const float yScale = post->clayer->yScale - pre->clayer->yScale;

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
   int nTotalTasks = nNeurons * numTasks;

   PVSynapseBundle ** allBPtrs   = (PVSynapseBundle**) malloc(nNeurons*sizeof(PVSynapseBundle*));
   PVSynapseBundle  * allBundles = (PVSynapseBundle *) malloc(nNeurons*sizeof(PVSynapseBundle));
   PVSynapseTask** allTPtrs = (PVSynapseTask**) malloc(nTotalTasks*sizeof(PVSynapseTask*));
   PVSynapseTask * allTasks = (PVSynapseTask*)  malloc(nTotalTasks*sizeof(PVSynapseTask));
   PVPatch       * allData  = (PVPatch*)        malloc(nTotalTasks*sizeof(PVPatch));

   bundles = allBPtrs;

   for (int kPre = 0; kPre < nNeurons; kPre++) {
      int offset = kPre*sizeof(PVSynapseBundle);
      bundles[kPre] = (PVSynapseBundle*) ((char*) allBundles + offset);
      bundles[kPre]->numTasks = numTasks;
      bundles[kPre]->tasks = allTPtrs + kPre*numTasks;

      PVSynapseBundle * list = bundles[kPre];
      for (int i = 0; i < numTasks; i++) {
         offset = (i + kPre*numTasks) * sizeof(PVSynapseTask);
         list->tasks[i] = (PVSynapseTask*) ((char*) allTasks + offset);
      }

      for (int i = 0; i < numTasks; i++) {
         PVSynapseTask * task = list->tasks[i];

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

         offset = (i + kPre*numTasks) * sizeof(PVPatch);
         task->data = (PVPatch*) ((char*) allData + offset);
         pvpatch_init(task->data, nxp, nyp, nfp, psx, psy, psf, phi);
      }
   }

   return 0;
}
#endif

int HyPerConn::createNorthernSynapseBundles(int numTasks)
{
   // TODO - these needs to be an input parameter obtained from the connection
   const float kxPost0Left = 0.0f;
   const float kyPost0Left = 0.0f;

   const float nxPre  = pre->clayer->loc.nx;
   const float nyPre  = nyp;
   const float kx0Pre = pre->clayer->loc.kx0;
   const float ky0Pre = pre->clayer->loc.ky0 - nyp;
   const float nfPre  = pre->clayer->numFeatures;

   const float nxPost  = post->clayer->loc.nx;
   const float nyPost  = post->clayer->loc.ny;
   const float kx0Post = post->clayer->loc.kx0;
   const float ky0Post = post->clayer->loc.ky0;
   const float nfPost  = post->clayer->numFeatures;

   const float xScale = post->clayer->xScale - pre->clayer->xScale;
   const float yScale = post->clayer->yScale - pre->clayer->yScale;

   const float numBorder = post->clayer->numBorder;
   assert(numBorder == 0);

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
   int nTotalTasks = nNeurons * numTasks;

   PVSynapseBundle ** allBPtrs   = (PVSynapseBundle**) malloc(nNeurons*sizeof(PVSynapseBundle*));
   PVSynapseBundle  * allBundles = (PVSynapseBundle *) malloc(nNeurons*sizeof(PVSynapseBundle));
   PVSynapseTask** allTPtrs = (PVSynapseTask**) malloc(nTotalTasks*sizeof(PVSynapseTask*));
   PVSynapseTask * allTasks = (PVSynapseTask*)  malloc(nTotalTasks*sizeof(PVSynapseTask));
   PVPatch       * allData  = (PVPatch*)        malloc(nTotalTasks*sizeof(PVPatch));

   bundles = allBPtrs;

   for (int kPre = 0; kPre < nNeurons; kPre++) {
      int offset = kPre*sizeof(PVSynapseBundle);
      bundles[kPre] = (PVSynapseBundle*) ((char*) allBundles + offset);
      bundles[kPre]->numTasks = numTasks;
      bundles[kPre]->tasks = allTPtrs + kPre*numTasks;

      PVSynapseBundle * list = bundles[kPre];
      for (int i = 0; i < numTasks; i++) {
         offset = (i + kPre*numTasks) * sizeof(PVSynapseTask);
         list->tasks[i] = (PVSynapseTask*) ((char*) allTasks + offset);
      }

      for (int i = 0; i < numTasks; i++) {
         PVSynapseTask * task = list->tasks[i];

         // use the translation invariant set of weights (default)
         task->weights = this->wPatches[i];

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
         pvdata_t * phi = post->clayer->phi[0] + kl;

         offset = (i + kPre*numTasks) * sizeof(PVPatch);
         task->data = (PVPatch*) ((char*) allData + offset);
         pvpatch_init(task->data, nxp, nyp, nfp, psx, psy, psf, phi);
      }
   }

   return 0;
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

   const int nxPre  = pre->clayer->loc.nx;
   const int nyPre  = pre->clayer->loc.ny;
   const int nfPre  = pre->clayer->numFeatures;

   const int nxPost  = post->clayer->loc.nx;
   const int nyPost  = post->clayer->loc.ny;
   const int nfPost  = post->clayer->numFeatures;
   const int numPost = post->clayer->numNeurons;

   const int nxPrePatch = nxp;
   const int nyPrePatch = nyp;

   const int nxPostPatch = nxp * powXScale;
   const int nyPostPatch = nyp * powYScale;
   const int nfPostPatch = pre->clayer->numFeatures;

   // the number of features is the end-point value (normally post-synaptic)
   const int numPostPatch = nxPostPatch * nyPostPatch * nfPostPatch;


   if (wPostPatches == NULL) {
      cout << " \nWeights created\n\n" << endl;
      wPostPatches = createWeights(nxPostPatch, nyPostPatch, nfPostPatch, numPost);
   }

   // loop through post-synaptic neurons

   for (int kPost = 0; kPost < numPost; kPost++) {
//      if(kPost == 63*4)
  //          cout << "nxPostPatch = " << nxPostPatch << " nyPostPatch = " << nyPostPatch
    //             << " nfPostPatch = " << nfPostPatch << endl;
//      if(kPost == 63*4)
  //       cout << "kPost = " << kPost;
      int kxPost = kxPos(kPost, nxPost, nyPost, nfPost);
      int kyPost = kyPos(kPost, nxPost, nyPost, nfPost);
      int kfPost = featureIndex(kPost, nxPost, nyPost, nfPost);
    //  if(kPost == 63*4)
      //   cout << " kxPost = " << kxPost << " kyPost = " << kyPost << " kfPost = " << kfPost << endl;

//      int kxPreHead = nearby_neighbor(kxPost, -xScale) - nxPostPatch/2;
//      int kyPreHead = nearby_neighbor(kyPost, -yScale) - nyPostPatch/2;
      // TODO - does patchHead work in general for post to pre mapping and -scale?
      int kxPreHead = pvlayer_patchHead((float) kxPost, 0.0, -xScale, (float) nxPostPatch);
      int kyPreHead = pvlayer_patchHead((float) kyPost, 0.0, -yScale, (float) nyPostPatch);
//      if(kPost == 63*4)
  //       cout << " kxPreHead = " << kxPreHead << " kyPreHead = " << kyPreHead << endl;


      for (int kp = 0; kp < numPostPatch; kp++) {
         int kxPostPatch = kxPos(kp, nxPostPatch, nyPostPatch, nfPre);
         int kyPostPatch = kyPos(kp, nxPostPatch, nyPostPatch, nfPre);
         int kfPostPatch = featureIndex(kp, nxPostPatch, nyPostPatch, nfPre);
         //if(kPost == 63*4)
          //  cout << " kxPostPatch = " << kxPostPatch << " kyPostPatch = " << kyPostPatch <<
           //           " kfPostPatch = " << kfPostPatch << endl;

         int kxPre = kxPreHead + kxPostPatch;
         int kyPre = kyPreHead + kyPostPatch;
         int kfPre = kfPostPatch;
         int kPre = kIndex(kxPre, kyPre, kfPre, nxPre, nyPre, nfPre);

         //if(kPost == 63*4)
           //          cout << " kxPre = " << kxPre << " kyPre = " << kyPre <<
             //                  " kfPre= " << kfPre <<  " kPre = " << kPre << endl;

         if (kxPre<0 || kyPre < 0 || kxPre >= nxPre|| kyPre >= nyPre) { //Marian, Shreyas, David changed conditions to fix boundary problems
            assert(kxPre < 0 || kyPre < 0 || kxPre >= nxPre || kyPre >= nyPre);
            wPostPatches[kPost]->data[kp] = 0.0;
            //printf("kxPost=%2d kxPre=%2d kPre=%2d kp=%2d kxPrePatch=?  kxPostPatch=%2d kPrePatch=?  w=%f\n", kxPost, kxPre, kPre, kp, kxPostPatch, wPostPatches[kPost]->data[kp]);
         }
         else {
            PVPatch * p = wPatches[kPre];
            int nxp = (kxPre < nxPrePatch/2) ? p->nx : nxPrePatch;
            int nyp = (kyPre < nyPrePatch/2) ? p->ny : nyPrePatch;
            int kxPrePatch = nxp - (1 + kxPostPatch / powXScale);
            int kyPrePatch = nyp - (1 + kyPostPatch / powYScale);
            int kPrePatch = kIndex(kxPrePatch, kyPrePatch, kfPost, p->nx, p->ny, p->nf);
            //cout << " kPrePatch = " << kPrePatch << " " << p->data[kPrePatch] ;
            wPostPatches[kPost]->data[kp] = p->data[kPrePatch];
            //printf("kxPost=%2d kxPre=%2d kPre=%2d kp=%2d kxPrePatch=%2d kxPostPatch=%2d kPrePatch=%2d w=%f\n", kxPost, kxPre, kPre, kp, kxPrePatch, kxPostPatch, kPrePatch, wPostPatches[kPost]->data[kp]);
         }
         //cout << wPostPatches[kPost]->data[kp] << " "
      }
      if(kPost == 63*4)
         getchar();

   }
   return wPostPatches;
}

int HyPerConn::writePostPatchWeights(int ioAppend) {

   char txtoutfile[128];
   char binoutfile[128];

   const int numPost   = post->clayer->numNeurons;
   const int numParams = post->clayer->numParams;
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

   FILE *txtoutput = NULL, *binoutput = NULL;

   // the number of features is the end-point value (normally post-synaptic)
   const int numPostPatch = nxPostPatch * nyPostPatch * nfPostPatch;

   //TODO: take out magic number
   snprintf(txtoutfile, 127, "%sw%d_Post.txt", OUTPUT_PATH, getConnectionId());
   snprintf(txtoutfile, 127, "%sw%d_Post.bin", OUTPUT_PATH, getConnectionId());
   printf("txt: %s, bin: %s\n", txtoutfile, binoutfile);

   if (!ioAppend) {
      txtoutput = fopen(txtoutfile, "w");
      binoutput = fopen(binoutfile, "wb");

      if ((NULL!= txtoutput) && (NULL != binoutput)) {
         fprintf(txtoutput, "numParams: %d,\nnxPost: %d,\nnyPost: %d, ", \
                            numParams, nxPost, nyPost);
         fprintf(txtoutput, "nfPost: %d,\nnumPostPatch: %d, ", \
                            nfPost, numPostPatch);

         fprintf(binoutput, "%d%d%d%d%d", numParams, nxPost, nyPost, 
                                          nfPost, numPostPatch);
      }
   }
   else {
      txtoutput = fopen(txtoutfile, "a");
      binoutput = fopen(binoutfile, "ab");
   }

   if ((NULL != txtoutput) && (NULL != binoutput)) {
      for (int i = 0; i < numPost; i++) {
         for (int j = 0; j < numPostPatch; j++) {
            fprintf(txtoutput, "%f ", wPostPatches[i]->data[j]);
            fprintf(binoutput, "%f", wPostPatches[i]->data[j]);
            }
         fprintf(txtoutput, "\n");
         fprintf(binoutput, "\n");
         }
      fclose(txtoutput);
      fclose(binoutput);
      }
   else {
      fprintf(stderr, "Error opening file: Cannot write post-patch weights. \n");
      return 1;
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
   const int fPre  = (int) featureIndex(kPre, nxPre, nyPre, nfPre);

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
