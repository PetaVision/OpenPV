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

CliqueLayer::CliqueLayer()
{
   initialize_base();
}

CliqueLayer::CliqueLayer(const char * name, HyPerCol * hc, int numChannels)
{
   initialize_base();
   initialize(name, hc, numChannels);
}

CliqueLayer::CliqueLayer(const char * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc, MAX_CHANNELS);
}


CliqueLayer::~CliqueLayer()
{
}

int CliqueLayer::initialize_base()
{
   return PV_SUCCESS;
}

int CliqueLayer::initialize(const char * name, HyPerCol * hc, int numChannels)
{
   ANNLayer::initialize(name, hc, numChannels);
   PVParams * params = parent->parameters();
   Voffset = params->value(name, "Voffset", 0.0f, true);
   Vgain = params->value(name, "Vgain", 2.0f, true);
   //cliqueSize = params->value(name, "cliqueSize", 1, true);
   return PV_SUCCESS;
}

int CliqueLayer::recvSynapticInput(HyPerConn * conn, const PVLayerCube * activity, int axonId)
{
   recvsyn_timer->start();
   enum ChannelType channel_type = conn->getChannel();
   if (channel_type == CHANNEL_EXC){
      return HyPerLayer::recvSynapticInput(conn, activity, axonId);
   }
   // number of axons = patch size ^ (clique size - 1)
   int numCliques = conn->numberOfAxonalArborLists();
   int cliqueSize = 1 + (int) rint(log2(numCliques)/ log2(conn->xPatchSize()*conn->yPatchSize()*conn->fPatchSize()));
   if (cliqueSize == 1){
      return HyPerLayer::recvSynapticInput(conn, activity, axonId);
   }

   assert(axonId == 0);
   // assume called only once
   //const int numExtended = activity->numItems;

#ifdef DEBUG_OUTPUT
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   printf("[%d]: HyPerLayer::recvSynapticInput: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, axonId, getNumExtended(), activity, this, conn);
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
   const int nxPreExt = preLoc->nx + 2 * preLoc->nb;
   const int nyPreExt = preLoc->ny + 2 * preLoc->nb;
   int syPost = conn->getPostNonextStrides()->sy; // stride in layer

   // if pre and post are the same layers, make a clone of size PVPatch to hold temporary activity values
   // in order to eliminate generalize self-interactions
   // note that during learning, per and post may be separate instantiations
   bool self_flag = conn->getSelfFlag();
   pvdata_t * a_post_mask = NULL;
   const int a_post_size = conn->fPatchSize() * conn->xPatchSize() * conn->yPatchSize();
   a_post_mask = (pvdata_t *) calloc(a_post_size, sizeof(pvdata_t));
   assert(a_post_mask != NULL);
   // get linear index of cell at center of patch for self_flag == true
   const int k_post_self = (int) (a_post_size / 2);  // if self_flag == true, a_post_size should be odd
   for (int k_post = 0; k_post < a_post_size; k_post++) {
      a_post_mask[k_post] = 1;
   }

   // gather active indices in extended layer
   // hard to pre-compute at HyPerLayer level because of variable delays
   int numActiveExt = 0;
   unsigned int * activeExt = (unsigned int *) calloc(conn->getPre()->getNumExtended(),
         sizeof(int));
   assert(activeExt != NULL);
   float * aPre = activity->data;
   for (int kPreExt = 0; kPreExt < conn->getPre()->getNumExtended(); kPreExt++) {
      if (aPre[kPreExt] == 0) continue;
      activeExt[numActiveExt++] = kPreExt;
   }

   // calc dimensions of wPostPatch
   // TODO: following is copied from calculation of wPostPatches and should be pre-calculated and stored there in HyPerConn::wPostPatches
   // TODO: following should be implemented as HyPerConn::calcPostPatchSize
   //const PVLayer * lPre = clayer;
   const float xScaleDiff = conn->getPost()->getXScale() - getXScale();
   const float yScaleDiff = conn->getPost()->getYScale() - getYScale();
   const float powXScale = pow(2.0f, (float) xScaleDiff);
   const float powYScale = pow(2.0f, (float) yScaleDiff);
   //const int prePad = lPre->loc.nb;
   const int nxPostPatch = (int) (conn->xPatchSize() * powXScale); // TODO: store in HyPerConn::wPostPatches
   const int nyPostPatch = (int) (conn->yPatchSize() * powYScale); // TODO: store in HyPerConn::wPostPatches
   //const int nfPostPatch = nf;

   // clique dimensions
   // a new set of cliques is centered on each pre-synaptic cell with radius nzPostPatch/2
   // TODO: precompute clique dimensions during CliqueConn::initialize
   int nyCliqueRadius = (int) (nyPostPatch / 2);
   int nxCliqueRadius = (int) (nxPostPatch / 2);
   int cliquePatchSize = (2 * nxCliqueRadius + 1) * (2 * nyCliqueRadius + 1) * nfPre;
   //int numKernels = conn->numDataPatches();  // per arbor?
   //int numCliques = pow(cliquePatchSize, cliqueSize - 1);
   //assert(numCliques == conn->numberOfAxonalArborLists());

   // loop over all products of cliqueSize active presynaptic cells
   // outer loop is over presynaptic cells, each of which defines the center of a cliquePatch
   // inner loop is over all combinations of clique cells within cliquePatch boundaries, which may be shrunken
   // TODO: pre-allocate cliqueActiveIndices as CliqueConn::cliquePatchSize member variable
   int * cliqueActiveIndices = (int *) calloc(cliquePatchSize, sizeof(int));
   assert(cliqueActiveIndices != NULL);
   for (int kPreActive = 0; kPreActive < numActiveExt; kPreActive++) {
      int kPreExt = activeExt[kPreActive];

      // get indices of active elements in clique radius
      // watch out for shrunken patches!
      int numActiveElements = 0;
      int kxPreExt = kxPos(kPreExt, nxPreExt, nyPreExt, nfPre);
      int kyPreExt = kyPos(kPreExt, nxPreExt, nyPreExt, nfPre);
      if (cliqueSize > 1) {
         for (int kyCliqueExt = ((kyPreExt - nyCliqueRadius) > 0 ? (kyPreExt - nyCliqueRadius) : 0);
               kyCliqueExt < ((kyPreExt + nyCliqueRadius) <= nyPreExt ? (kyPreExt + nyCliqueRadius) : nyPreExt); kyCliqueExt++) {
            //if (kyCliqueExt < 0 || kyCliqueExt > nyPreExt) continue;
            for (int kxCliqueExt = ((kxPreExt - nxCliqueRadius) > 0 ? (kxPreExt - nxCliqueRadius) : 0);
                  kxCliqueExt < ((kxPreExt + nxCliqueRadius) <= nxPreExt ? (kxPreExt + nxCliqueRadius) : nxPreExt); kxCliqueExt++) {
               //if (kyCliqueExt < 0 || kyCliqueExt > nxPreExt) continue;
               for (int kfCliqueExt = 0; kfCliqueExt < nfPre; kfCliqueExt++) {
                  int kCliqueExt = kIndex(kxCliqueExt, kyCliqueExt, kfCliqueExt, nxPreExt,
                        nyPreExt, nfPre);
                  if ((aPre[kCliqueExt] == 0) || (kCliqueExt == kPreExt)) continue;
                  cliqueActiveIndices[numActiveElements++] = kCliqueExt;
               }
            }
         }
      } // cliqueSize > 1
      else {
         cliqueActiveIndices[numActiveElements++] = kPreExt; // each cell is its own clique if cliqueSize == 1
      }
      if (numActiveElements < (cliqueSize-1)) continue;

      // loop over all active combinations of size=cliqueSize-1 in clique radius
      int numActiveCliques = (int)pow(numActiveElements, cliqueSize - 1);
      for (int kClique = 0; kClique < numActiveCliques; kClique++) {

         //initialize a_post_tmp
         if (self_flag) {  // otherwise, a_post_mask is not modified and thus doesn't have to be updated
            for (int k_post = 0; k_post < a_post_size; k_post++) {
               a_post_mask[k_post] = 1;
            }
            a_post_mask[k_post_self] = 0;
         }

         // decompose kClique to compute product of active clique elements
         int arborNdx = 0;
         pvdata_t cliqueProd = aPre[kPreExt];
         int kResidue = kClique;
         int maxIndex = -1;
         for (int iProd = 0; iProd < cliqueSize - 1; iProd++) {
            int kPatchActive = (unsigned int) (kResidue
                  / pow(numActiveElements, cliqueSize - 1 - iProd - 1));

            // only apply each permutation of clique elements once, no element can contribute more than once
            if (kPatchActive <= maxIndex) {
               break;
            }
            else {
               maxIndex = kPatchActive;
            }
            int kCliqueExt = cliqueActiveIndices[kPatchActive];
            cliqueProd *= aPre[kCliqueExt];
            kResidue = kResidue
                  - kPatchActive * (int)pow(numActiveElements, cliqueSize - 1 - iProd - 1);

            // compute arborIndex for this clique element
            int kxCliqueExt = kxPos(kCliqueExt, nxPreExt, nyPreExt, nfPre);
            int kyCliqueExt = kyPos(kCliqueExt, nxPreExt, nyPreExt, nfPre);
            int kfClique = featureIndex(kCliqueExt, nxPreExt, nyPreExt, nfPre);
            int kxPatch = kxCliqueExt - kxPreExt + nxCliqueRadius;
            int kyPatch = kyCliqueExt - kyPreExt + nyCliqueRadius;
            unsigned int kArbor = kIndex(kxPatch, kyPatch, kfClique,
                  (2 * nxCliqueRadius + 1), (2*nyCliqueRadius + 1), nfPre);
            arborNdx += kArbor * (int)pow(cliquePatchSize, cliqueSize - 1 - iProd - 1);
            if ((arborNdx < 0) || (arborNdx >= numCliques)){
                  assert((arborNdx >= 0) && (arborNdx < numCliques));
            }
            // remove self-interactions if pre == post
            if (self_flag){
               a_post_mask[kArbor] = 0;
            }
         } // iProd

         PVPatch * w_patch = conn->getWeights(kPreExt, arborNdx);

         const pvdata_t * w_start = conn->get_wDataStart(arborNdx);
         int kernelIndex = conn->patchToDataLUT(kPreExt);
         const pvdata_t * w_head = &(w_start[a_post_size*kernelIndex]);
         size_t w_offset = w_patch->offset; // w_patch->data - w_head;

         // WARNING - assumes weight and GSyn patches from task same size
         //         - assumes patch stride sf is 1

         int nkPost = conn->getPost()->getLayerLoc()->nf * w_patch->nx;
         int nyPost = w_patch->ny;
         int sywPatch = conn->yPatchStride(); // stride in patch

         // TODO - unroll
         for (int y = 0; y < nyPost; y++) {
            pvpatch_accumulate2(nkPost,
                  (float *) (conn->getGSynPatchStart(kPreExt, arborNdx) + y * syPost),
                  cliqueProd,
                  (float *) (w_head + w_offset + y * sywPatch),// (w_patch->data + y * sywPatch),
                  (float *) (a_post_mask + w_offset + y * sywPatch));
         }

      } // kClique
   } // kPreActive
   free(activeExt);
   free(cliqueActiveIndices);
   free(a_post_mask);
   recvsyn_timer->stop();
   return PV_BREAK;
}

// TODO: direct clique input to separate GSyn: CHANNEL_CLIQUE
// the following is copied directly from ODDLayer::updateState()
int CliqueLayer::updateState(double timef, double dt)
{
   return updateState(timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(), getNumChannels(), GSyn[0], this->Voffset, this->Vgain, this->VMax, this->VMin, this->VThresh, clayer->columnId);
}

int CliqueLayer::updateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead, pvdata_t Voffset, pvdata_t Vgain, pvdata_t VMax, pvdata_t VMin, pvdata_t VThresh, int columnID) {
   pv_debug_info("[%d]: CliqueLayer::updateState:", columnID);

   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;

   // Assumes that channels are contiguous in memory, i.e. GSyn[ch] = GSyn[0]+num_neurons*ch.  See allocateBuffers().
   pvdata_t * gSynExc = getChannelStart(gSynHead, CHANNEL_EXC, num_neurons);
   pvdata_t * gSynInh = getChannelStart(gSynHead, CHANNEL_INH, num_neurons);
   //pvdata_t * gSynInhB = getChannelStart(gSynHead, CHANNEL_INHB, num_neurons);
// assume bottomUp input to gSynExc, target lateral input to gSynInh, distractor lateral input to gSynInhB
   for (int k = 0; k < num_neurons; k++) {
      V[k] = 0.0f;
      pvdata_t bottomUp_input = gSynExc[k];
      if (bottomUp_input <= 0.0f) {
         continue;
      }
      pvdata_t lateral_exc = gSynInh[k];
      //pvdata_t lateral_inh = gSynInhB[k];
      //pvdata_t lateral_denom = ((lateral_exc + fabs(lateral_inh)) > 0.0f) ? (lateral_exc + fabs(lateral_inh)) : 1.0f;

      //V[k] = bottomUp_input * (this->Voffset + this->Vgain * (lateral_exc - lateral_inh));
      V[k] = bottomUp_input * (Voffset + Vgain * (lateral_exc)); // - fabs(lateral_inh))); // / lateral_denom);
   } // k

   resetGSynBuffers_HyPerLayer(num_neurons, getNumChannels(), gSynHead);
   // resetGSynBuffers();
   applyVMax_ANNLayer(num_neurons, V, VMax); // applyVMax();
   applyVThresh_ANNLayer(num_neurons, V, VMin, VThresh); // applyVThresh();
   setActivity_HyPerLayer(num_neurons, A, V, nx, ny, nf, loc->nb); // setActivity();
   updateActiveIndices();

   return 0;
}

int CliqueLayer::updateActiveIndices()
{
//   int numActive = 0;
//   PVLayerLoc & loc = clayer->loc;
//   pvdata_t * activity = clayer->activity->data;
//
//   for (int k = 0; k < getNumNeurons(); k++) {
//      const int kex = kIndexExtended(k, loc.nx, loc.ny, loc.nf, loc.nb);
//      if (activity[kex] > 0.0) {
//         clayer->activeIndices[numActive++] = globalIndexFromLocal(k, loc);
//      }
//   }
//   clayer->numActive = numActive;
//   return PV_SUCCESS;
   return calcActiveIndices();
}

} /* namespace PV */

