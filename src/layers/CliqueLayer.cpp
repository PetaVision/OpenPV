/*
 * CliqueLayer.cpp
 *
 *  Created on: Sep 3, 2011
 *      Author: gkenyon
 */

#include "CliqueLayer.hpp"
#include "../utils/conversions.h"
#include <assert.h>

namespace PV {

CliqueLayer::CliqueLayer(const char * name, HyPerCol * hc, int numChannels) : ANNLayer(name, hc, numChannels) {
   CliqueLayer::initialize();
}

CliqueLayer::CliqueLayer(const char * name, HyPerCol * hc) : ANNLayer(name, hc, MAX_CHANNELS) {
   CliqueLayer::initialize();
}

// parent class initialize already called in constructor
int CliqueLayer::initialize() {
   PVParams * params = parent->parameters();
   Voffset = params->value(name, "Voffset", 0.0f, true);
   Vgain = params->value(name, "Vgain", 2.0f, true);
   return PV_SUCCESS; //ANNLayer::initialize();
}

int CliqueLayer::recvSynapticInput(HyPerConn * conn, PVLayerCube * activity,
      int axonId) {
   recvsyn_timer->start();

   assert(axonId == 0); // assume called only once
   //const int numExtended = activity->numItems;

#ifdef DEBUG_OUTPUT
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   printf("[%d]: HyPerLayr::recvSyn: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, axonId, numExtended, activity, this, conn);
   fflush(stdout);
#endif

   // get margin indices of pre layer
   // (whoops, not needed since we probably have to recompute all active indices in extended layer on the fly due to variable delays)
   //const int * marginIndices = conn->getPre()->getMarginIndices();
   //int numMargin = conn->getPre()->getNumMargin();

   const PVLayerLoc * preLoc = conn->getPre()->getLayerLoc();
   const int nfPre = preLoc->nf;
   //const int nxPre = preLoc->nx;
   //const int nyPre = preLoc->ny;
   const int nxPreExt = preLoc->nx + 2*preLoc->nb;
   const int nyPreExt = preLoc->ny + 2*preLoc->nb;

   // gather active indices in extended layer
   // hard to pre-compute at HyPerLayer level because of variable delays
   int numActiveExt = 0;
   unsigned int * activeExt = (unsigned int *) calloc(conn->getPre()->getNumExtended(), sizeof(int));
   assert(activeExt != NULL);
   float * aPre = activity->data;
   for (int kPreExt = 0; kPreExt < conn->getPre()->getNumExtended(); kPreExt++) {
      if (aPre[kPreExt] == 0)
         continue;
      activeExt[numActiveExt++] = kPreExt;
   }

   // calc dimensions of wPostPatch
   // TODO: following is copied from calculation of wPostPatches and should be pre-calculated and stored there in HyPerConn::wPostPatches
   // TODO: following should be implemented as HyPerConn::calcPostPatchSize
   //const PVLayer * lPre = clayer;
   const float xScaleDiff = conn->getPost()->getXScale() - getXScale();
   const float yScaleDiff = conn->getPost()->getYScale() - getYScale();
   const float powXScale = powf(2.0f, (float) xScaleDiff);
   const float powYScale = powf(2.0f, (float) yScaleDiff);
   //const int prePad = lPre->loc.nb;
   const int nxPostPatch = (int) (conn->xPatchSize() * powXScale); // TODO: store in HyPerConn::wPostPatches
   const int nyPostPatch = (int) (conn->yPatchSize() * powYScale);// TODO: store in HyPerConn::wPostPatches
   //const int nfPostPatch = nf;

   // clique dimensions
   // a new set of cliques is centered on each pre-synaptic cell with radius nzPostPatch/2
   // TODO: precompute clique dimensions during CliqueConn::initialize
   int nyCliqueRadius = (int) (nyPostPatch/2);
   int nxCliqueRadius = (int) (nxPostPatch/2);
   int cliquePatchSize = (2*nxCliqueRadius + 1) * (2*nyCliqueRadius + 1) * nfPre;
   int cliqueSize = 1;// number of presynaptic cells in clique (traditional ANN uses 1)
   //int numKernels = conn->numDataPatches();  // per arbor?
   int numCliques = pow(cliquePatchSize, cliqueSize-1);
   assert(numCliques == conn->numberOfAxonalArborLists());

   // loop over all products of cliqueSize active presynaptic cells
   // outer loop is over presynaptic cells, each of which defines the center of a cliquePatch
   // inner loop is over all combinations of clique cells within cliquePatch boundaries, which may be shrunken
   // TODO: pre-allocate cliqueActiveIndices as CliqueConn::cliquePatchSize member variable
   int * cliqueActiveIndices = (int *) calloc(cliquePatchSize, sizeof(int));
   assert(cliqueActiveIndices != NULL);
   for (int kPreActive = 0; kPreActive < numActiveExt; kPreActive++) {
      int kPreExt = activeExt[kPreActive];

      // get indices of active elements in clique radius
      int numActiveElements = 0;
      int kxPreExt = kxPos(kPreExt, nxPreExt, nyPreExt, nfPre);
      int kyPreExt = kyPos(kPreExt, nxPreExt, nyPreExt, nfPre);
      for(int kyCliqueExt = kyPreExt - nyCliqueRadius; kyCliqueExt < kyPreExt + nyCliqueRadius; kyCliqueExt++) {
         for(int kxCliqueExt = kxPreExt - nxCliqueRadius; kxCliqueExt < kxPreExt + nxCliqueRadius; kxCliqueExt++) {
            for(int kfCliqueExt = 0; kfCliqueExt < nfPre; kfCliqueExt++) {
               int kCliqueExt = kIndex(kxCliqueExt, kyCliqueExt, kfCliqueExt, nxPreExt, nyPreExt, nfPre);
               if (aPre[kCliqueExt] == 0) continue;
               cliqueActiveIndices[numActiveElements++] = kCliqueExt;
            }
         }
      }

      // loop over all active elements in clique radius
      int numActiveCliques = pow(numActiveElements, cliqueSize-1);
      for(int kClique = 0; kClique < numActiveCliques; kClique++) {

         // decompose kClique to compute product of active clique elements
         int arborNdx = 0;
         pvdata_t cliqueProd = aPre[kPreExt];
         int kResidue = kClique;
         for(int iProd = 0; iProd < cliqueSize-1; iProd++) {
            int kPatchActive = (unsigned int) (kResidue / numActiveElements);
            int kCliqueExt = cliqueActiveIndices[kPatchActive];
            cliqueProd *= aPre[kCliqueExt];
            kResidue = kResidue - kPatchActive * numActiveElements;

            // compute arborIndex for this clique element
            int kxCliqueExt = kxPos(kCliqueExt, nxPreExt, nyPreExt, nfPre);
            int kyCliqueExt = kyPos(kCliqueExt, nxPreExt, nyPreExt, nfPre);
            int kfClique = featureIndex(kCliqueExt, nxPreExt, nyPreExt, nfPre);
            int kxPatch = kxCliqueExt - kxPreExt + nxCliqueRadius;
            int kyPatch = kyCliqueExt - kyPreExt + nyCliqueRadius;
            unsigned int kArbor = kIndex(kxPatch, kyPatch, kfClique, (2*nxCliqueRadius + 1), (2*nyCliqueRadius + 1), nfPre);
            arborNdx += kArbor * pow(cliquePatchSize,iProd);
         }

         // receive weights input from clique (mostly copied from superclass method)
         PVAxonalArbor * arbor = conn->axonalArbor(kPreExt, arborNdx);
         PVPatch * GSyn = arbor->data;
         PVPatch * weights = arbor->weights;

         // WARNING - assumes weight and GSyn patches from task same size
         //         - assumes patch stride sf is 1

         int nkPost = GSyn->nf * GSyn->nx;
         int nyPost = GSyn->ny;
         int syPost = GSyn->sy;// stride in layer
         int sywPatch = weights->sy;// stride in patch

         // TODO - unroll
         for (int y = 0; y < nyPost; y++) {
            pvpatch_accumulate(nkPost, GSyn->data + y*syPost, cliqueProd, weights->data + y*sywPatch);
         }

      } // kClique
   } // kPreActive
   free(activeExt);
   free(cliqueActiveIndices);
   recvsyn_timer->stop();
   return PV_BREAK;
}

// TODO: direct clique input to separate GSyn: CHANNEL_CLIQUE
// the following is copied directly from ODDLayer::updateState()
int CliqueLayer::updateState(float time, float dt) {

   pv_debug_info("[%d]: CliqueLayer::updateState:", clayer->columnId);

   pvdata_t * V = clayer->V;
   pvdata_t * gSynExc = getChannel(CHANNEL_EXC);
   pvdata_t * gSynInh = getChannel(CHANNEL_INH);
   pvdata_t * gSynInhB = getChannel(CHANNEL_INHB);
<<<<<<< .mine
   float offset = 0.0f; //VThresh;
   float gain = 1.75f;  // 1 -> log base 2, 2 -> log base sqrt(2)
=======
//   float offset = 0.0f; //VThresh;
//   float gain = 2.0f;  // 1 -> log base 2, 2 -> log base sqrt(2)
//   assert(this->Vgain == 16.0f);
//   assert(this->Voffset == 0.0f);
>>>>>>> .r4350

   // assume bottomUp input to GSynExc, target lateral input to gSynInh, distractor lateral input to gSynInhB
   for (int k = 0; k < clayer->numNeurons; k++) {
      V[k] = 0.0f;
      pvdata_t bottomUp_input = gSynExc[k];
      if (bottomUp_input <= 0.0f) {
         continue;
      }
      pvdata_t target_input = gSynInh[k];
      pvdata_t distractor_input = gSynInhB[k];
      if (distractor_input > 0.0f){
         if (target_input > 0.0f){
            V[k] = bottomUp_input * (this->Voffset + this->Vgain * ((target_input - distractor_input) / distractor_input));
         }
         else{
            V[k] = 0.0f;
         }
      }
      else{  // distractor_input <= 0
         if (target_input > 0.0f){
            V[k] = 1.0f;
         }
         else{
            V[k] = 0.0f; //bottomUp_input;  // not sure what to do here, no support + or -
         }
      }
   } // k

   resetGSynBuffers();
   applyVMax();
   applyVThresh();
   setActivity();
   updateActiveIndices();

   return 0;
}

int CliqueLayer::updateActiveIndices(){
   int numActive = 0;
   PVLayerLoc & loc = clayer->loc;
   pvdata_t * activity = clayer->activity->data;

   for (int k = 0; k < getNumNeurons(); k++) {
      const int kex = kIndexExtended(k, loc.nx, loc.ny, loc.nf, loc.nb);
      if (activity[kex] > 0.0) {
         clayer->activeIndices[numActive++] = k; //globalIndexFromLocal(k, loc);
      }
   }
   clayer->numActive = numActive;
   return PV_SUCCESS;
}

} /* namespace PV */


