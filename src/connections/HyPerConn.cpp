/*
 * HyPerConnection.cpp
 *
 *  Created on: Oct 21, 2008
 *      Author: rasmussn
 */

#include "HyPerConn.hpp"
#include "WeightCache.h"
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

HyPerConn::HyPerConn()
{
   this->connId = 0;
   this->pre    = NULL;
   this->post   = NULL;
   this->pvconn = NULL;
   this->name   = strdup("Unknown");
   this->channel  = 0;
   this->stdpFlag = 0;
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

// TODO - while ((c = getopt(*argc, *argv, "f:")) != -1) {
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
   this->channel = channel;
   this->pvconn = new PVConnection;

   this->stdpFlag = 0;

   assert(this->channel <= post->clayer->numPhis);

   PVParams * inputParams = parent->parameters();
   nxp = inputParams->value(name, "nxp");
   nyp = inputParams->value(name, "nyp");

   nfp = post->clayer->numFeatures;

   setParams(inputParams, &defaultConnParams);
   assert(params->delay < MAX_F_DELAY);

   params->numDelay = params->varDelayMax - params->varDelayMin + 1;

   pvConnInit(pvconn, pre->clayer, post->clayer, params, channel);

   if (inputParams->present(name, "stdpFlag")) {
      stdpFlag = inputParams->value(name, "stdpFlag");
   }

   wPatches = createWeights();
   if (stdpFlag == 1) {
      pIncr = createWeights();
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
   if (stdpFlag == 1) {
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

int HyPerConn::initializeWeights(const char * filename)
{
   if (filename == NULL) {

      PVParams * params = parent->parameters();

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
      const int yScale = post->clayer->xScale - pre->clayer->yScale;
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

   fprintf(fd, "Connection weights for connection %d, neuron %d\n", getConnectionId(), k);
   fprintf(fd, "   (nxp,nyp,nfp)   = (%d,%d,%d)\n", (int)nxp, (int)nyp, (int)nfp);
   fprintf(fd, "   pre  (nx,ny,nf) = (%d,%d,%d)\n",
           (int)pre->clayer->loc.nx, (int)pre->clayer->loc.ny, (int)pre->clayer->numFeatures);
   fprintf(fd, "   post (nx,ny,nf) = (%d,%d,%d)\n",
           (int)post->clayer->loc.nx, (int)post->clayer->loc.ny, (int)post->clayer->numFeatures);
   fprintf(fd, "\n");
   pv_text_write_patch(fd, wPatches[k]);

   if (fd != stdout) {
      fclose(fd);
   }

   return 0;
}

int HyPerConn::deliver(PVLayerCube * cube, int neighbor)
{
   post->recvSynapticInput(this, cube, neighbor);

   return 0;
}

int HyPerConn::updateWeights(PVLayerCube * preActivity, int neighbor)
{
   // TODO - initialize these parameters
   float decayIncr;
   float decayDecr;
   float facIncr;
   float facDecr;
   float dWmax;
   float decay;

   // assume pDecr has been updated already, and weights have been used, so
   // 1. update Psij (pIncr) for each synapse
   // 2. update wij

   const int numActive = preActivity->numItems;

   // TODO - handle neighbors
   if (neighbor != 0) {
      return 0;
   }

#ifdef MULTITHREADED
   pv_signal_threads_recv(activity, (unsigned char) neighbor);
   pv_signal_threads_recv(conn->weights(), 0);
   pv_signal_threads_recv(conn->cliques(), 0);
#endif

   for (int kPre = 0; kPre < numActive; kPre++) {
      PVSynapseBundle * tasks = this->tasks(kPre, neighbor);

      float aj = preActivity->data[kPre];

      for (unsigned int i = 0; i < tasks->numTasks; i++) {
         PVPatch * pIncr = tasks->tasks[i]->plasticIncr;
         PVPatch * w     = tasks->tasks[i]->weights;
         float   * ai    = tasks->tasks[i]->activity;

         int nk  = (int) pIncr->nf * (int) pIncr->nx;
         int ny  = (int) pIncr->ny;
         int sy  = (int) pIncr->sy;

         // TODO - unroll

         // update Psij (pIncr variable)
         for (int y = 0; y < ny; y++) {
            pvpatch_update_plasticity_incr(nk, pIncr->data + y*sy, aj, decayIncr, facIncr);
         }

         // update weights
         for (int y = 0; y < ny; y++) {
            int yOff = y*sy;
            pvpatch_update_weights(nk, w->data + yOff, pDecr->data + yOff, pIncr->data + yOff,
                                   ai + yOff, aj, decay, dWmax,
                                   decayIncr, facIncr,
                                   decayDecr, facDecr);
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
   if (stdpFlag == 1) {
      return pIncr[k];
   }
   return NULL;
}

/**
 * Create a separate patch of weights for every neuron
 */
PVPatch ** HyPerConn::createWeights()
{
   // could create only a single patch with following call
   //   return createPatches(numBundles, nxp, nyp, nfp);

   assert(numBundles == 1);

   const int numPatches = numberOfWeightPatches();

   PVPatch ** patches = (PVPatch**) malloc(numPatches*sizeof(PVPatch*));

   // TODO - allocate space for them all at once
   for (int k = 0; k < numPatches; k++) {
      patches[k] = pvpatch_new(nxp, nyp, nfp);
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
      pvpatch_delete(wPatches[k]);
   }
   free(wPatches);

   if (stdpFlag == 1) {
      for (int k = 0; k < numPreNeurons; k++) {
         pvpatch_delete(pIncr[k]);
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
         // TODO make sure this is correct.........
         task->activity = post->clayer->activity->data + kl;

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

            // figure out on/off connectivity
            //if (this->channel == CHANNEL_EXC && f != fPre) continue;
            //if (this->channel == CHANNEL_INH && f == fPre) continue;
            if (f != fPre) continue;

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

      if (sum == 0.0) continue;  // all weights == zero is ok

      float factor = strength/sum;
      for (int i = 0; i < nx*ny; i++) w[f + i*nf] *= factor;
   }

   return 0;
}

}
