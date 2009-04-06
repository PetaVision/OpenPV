/*
 * PVConnection.c
 *
 *  Created on: Jul 29, 2008
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h> // for memcpy
#include "../include/pv_common.h"
#include "PVConnection.h"
#include "WeightCache.h"

#ifdef __cplusplus
extern "C"
{
#endif

// TODO - use HyPerConn class to do much of this?
int pvConnInit(PVConnection * pvConn, PVLayer * pre, PVLayer * post, PVConnParams * p, int channel)
{
   pvConn->pre  = pre;
   pvConn->post = post;

   pvConn->whichPhi = channel;

   // if this is not null, there is a free error (not used anyway for now)
   pvConn->params   = NULL;

   pvConn->delay       = p->delay;
   pvConn->fixDelay    = p->fixDelay;
   pvConn->varDelayMin = p->varDelayMin;
   pvConn->varDelayMax = p->varDelayMax;
   pvConn->numDelay    = p->numDelay;
   pvConn->isGraded    = p->isGraded;
   pvConn->vel         = p->vel;
   pvConn->rmin        = p->rmin;
   pvConn->rmax        = p->rmax;

   pvConn->preNormF  = NULL;
   pvConn->postNormF = NULL;

   // make the cutoff boundary big for now
   pvConn->r2 = post->loc.nx * post->loc.ny + post->loc.ny * post->loc.ny;

   PV_weightCache_init(pvConn);

   // TODO - use numDelayLevels rather than MAX_F_DELAY
   // Init the read ptr such that it follows the writeIdx (which
   // start at 0) by the correct amount.
   pvConn->readIdx = (MAX_F_DELAY - pvConn->delay - 1);

   return 0;
}

int pvConnFinalize(PVConnection * pvConn)
{
   if (pvConn->params)    free(pvConn->params);
   if (pvConn->preNormF)  free(pvConn->preNormF);
   if (pvConn->postNormF) free(pvConn->postNormF);
   PV_weightCache_finalize(pvConn);

   return 0;
}

// Calc responses to uniform stimuli to normalize the total input into each postsynaptic neuron
// to account for pixel aliasing.
int PVConnection_default_normalize(PVConnection* con, PVWeightFunction calcWeight,
      float scale)
{
   PVLayer* pre  = con->pre;
   PVLayer* post = con->post;
   void *params  = con->params;
   float prePos[MAX_DIMS], postPos[MAX_DIMS], prePosTranslated[MAX_DIMS];
   float *tempPhi;
   int synapse;
   float weight;
   int postNdx, preZ, preKernelIndex;

   if (con->pre->columnId == 0) {
#ifdef DEBUG_OUTPUT
      char msg[128];
      sprintf(msg, "Normalizing %s -> %s...\n", con->pre->name, con->post->name);
      pv_log(stderr, msg);
#endif
   }

   tempPhi = (float*) calloc(con->numKernels, sizeof(float));
   // first normalize total output of each presynaptic feature
   // For each feature at one presynaptic pixel (i.e. preX,preY = 0,0)
   for (preZ = 0; preZ < con->numKernels; preZ++) {
      //pvlayer_getPos(pre, preZ, &prePos[DIMX], &prePos[DIMY],
      //&prePos[DIMO]);
      //preKernelIndex = PV_weightCache_getKernelIndex( con, prePos);
      preKernelIndex = preZ;
      PV_weightCache_getKernelPos(con, preKernelIndex, prePos);

      memcpy(prePosTranslated, prePos, sizeof(prePos));
      PV_weightCache_getPreKernelIndex(con, prePosTranslated, 1);

      // For each neuron in the postsynaptic patch
      for (postNdx = 0; postNdx < post->numNeurons; postNdx++) {
         // Determine postsynaptic feature vector
         pvlayer_getPos(post, postNdx, &postPos[DIMX], &postPos[DIMY], &postPos[DIMO]);

         synapse = 1; // weightcache may change it
         if (!PV_weightCache_get(con, prePos, preKernelIndex, postPos, &weight)) {
            calcWeight(pre, post, params, prePos, postPos, &weight);
            synapse = PV_weightCache_set(con, prePosTranslated, preKernelIndex, postPos,
                  weight);
         }

         if (synapse) {
            tempPhi[preZ] += weight;
         } // for each neuron in the postsynaptic patch
      } // for each input neuron
   }

   // normalize total output of each presynaptic feature to 1.0
   for (preZ = 0; preZ < con->numKernels; preZ++) {
      con->preNormF[preZ] = 1.0 / tempPhi[preZ];
   }
   free(tempPhi);

   // TODO - take into account extended border
   tempPhi = (float*) calloc(PV_weightCache_getNumPostKernels(con), sizeof(float));
   int preNdx, postZ;
   // For each neuron in the presynaptic patch
   for (preNdx = 0; preNdx < pre->numNeurons; preNdx++) {
      pvlayer_getPos(pre, preNdx, &prePos[DIMX], &prePos[DIMY], &prePos[DIMO]);

      preKernelIndex = PV_weightCache_getPreKernelIndex(con, prePos, 0);
      memcpy(prePosTranslated, prePos, sizeof(prePos));
      PV_weightCache_getPreKernelIndex(con, prePosTranslated, 1);

      // For each feature at one postsynaptic pixel (i.e. postX, postY = 0,0)
      for (postZ = 0; postZ < PV_weightCache_getNumPostKernels(con); postZ++) {
         // Get a good representative neuron for normalization
         PV_weightCache_getPostKernelPos(con, postZ, postPos);

         synapse = 1; // calcWeight may change it
         if (!PV_weightCache_get(con, prePos, preKernelIndex, postPos, &weight)) {
            calcWeight(pre, post, params, prePos, postPos, &weight);

            synapse = PV_weightCache_set(con, prePosTranslated, preKernelIndex, postPos,
                  weight);
         }

         if (synapse) {
            tempPhi[postZ] += con->preNormF[preKernelIndex] * weight;
         } // for each neuron in the postsynaptic patch
      } // for each input neuron
   }

   for (postZ = 0; postZ < PV_weightCache_getNumPostKernels(con); postZ++) {
      con->postNormF[postZ] = scale / tempPhi[postZ];
   }
   free(tempPhi);

   return 0;
}//normalize

// --------------------------------------------------------------------------
// Default rcvSynapticInput() implementation:
//
// Input: non-sparse activity input.
// Output: For each postsynaptic, neuron, sum of weights based on input activity.
//
// Algorithm: Finds each active presynaptic neurons and calculates weight for
// each post-synaptic neuron, summing weights to get a single value for each
// post-synaptic neuron.
// --------------------------------------------------------------------------
int PVConnection_default_rcv(PVConnection* con, PVLayer *post, int nActivity,
      float *fActivity)
{
   PVLayer* pre = con->pre;
   // TODO - take into account extended border
   float *phi = con->post->phi[con->whichPhi];

   int postNdx, preNdx;
   float prePos[MAX_DIMS], postPos[MAX_DIMS];
   float weight;
   int preKernelIndex, postKernelIndex, numSynapses, postCounter;
   int postIdxOffset = (post->yOrigin * post->loc.nx + post->xOrigin) * post->numFeatures;

   // For each neuron in the presynaptic patch
   for (preNdx = 0; preNdx < pre->numNeurons; preNdx++) {
      if (fActivity[preNdx] == 0.0) continue; // optimization: skip 0 inputs (spiking or continuous)

      pvlayer_getPos(pre, preNdx, &prePos[DIMX], &prePos[DIMY], &prePos[DIMO]);
      preKernelIndex = PV_weightCache_getPreKernelIndex(con, prePos, 1);
      PV_weightCache_getPostNeurons(con, prePos, preKernelIndex, &numSynapses);

      for (postCounter = 0; postCounter < numSynapses; postCounter++) {
         PV_weightCache_getPostByIndex(con, prePos, preKernelIndex, postCounter, &weight,
               postPos, &postKernelIndex);
         if (pvlayer_getIndex(post, postPos, &postNdx) < 0) continue;

         // TODO - take into account extended border
         if (weight) phi[postNdx + postIdxOffset] += weight
               * con->preNormF[preKernelIndex] * con->postNormF[postKernelIndex];
      } // for each input neuron
   }
   return 0;
}

#ifdef __cplusplus
}
#endif
