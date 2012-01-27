/*
 * CliqueConn.cpp
 *
 *  Created on: Sep 8, 2011
 *      Author: gkenyon
 */

#include "CliqueConn.hpp"
int pvpatch_update_clique(int nk, float* RESTRICT v, float a, float* RESTRICT w);

namespace PV {

CliqueConn::CliqueConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, ChannelType channel, const char * filename,
      InitWeights *weightInit)
{
   KernelConn::initialize_base();
   KernelConn::initialize(name, hc, pre, post, channel, filename, weightInit);
};

int CliqueConn::updateState(float time, float dt)
{
   int status = KernelConn::updateState(time, dt);
   assert(status == PV_SUCCESS);
   return PV_SUCCESS;
};

int CliqueConn::update_dW(int arborId)
{
   const PVLayerLoc * preLoc = this->getPre()->getLayerLoc();
   const int nfPre = preLoc->nf;
   //const int nxPre = preLoc->nx;
   //const int nyPre = preLoc->ny;
   const int nxPreExt = preLoc->nx + 2 * preLoc->nb;
   const int nyPreExt = preLoc->ny + 2 * preLoc->nb;

   int delay = getDelay(arborId);
   //     // assume each synaptic connection with the same arborId has the same delay
   //     int delay = this->axonalArbor(0, arborId)->delay;

   // gather active indices in extended layer
   // hard to pre-compute at HyPerLayer level because of variable delays
   int numActiveExt = 0;
   unsigned int * activeExt = (unsigned int *) calloc(this->getPre()->getNumExtended(),
         sizeof(int));
   assert(activeExt != NULL);
   const pvdata_t * aPre = this->getPre()->getLayerData(delay);
   for (int kPreExt = 0; kPreExt < this->getPre()->getNumExtended(); kPreExt++) {
      if (aPre[kPreExt] == 0) continue;
      activeExt[numActiveExt++] = kPreExt;
   }

   // calc dimensions of wPostPatch
   // TODO: following is copied from calculation of wPostPatches and should be pre-calculated and stored there in HyPerConn::wPostPatches
   // TODO: following should be implemented as HyPerConn::calcPostPatchSize
   //const PVLayer * lPre = clayer;
   const float xScaleDiff = this->getPost()->getXScale() - this->getPre()->getXScale();
   const float yScaleDiff = this->getPost()->getYScale() - this->getPre()->getYScale();
   const float powXScale = pow(2.0f, (float) xScaleDiff);
   const float powYScale = pow(2.0f, (float) yScaleDiff);
   //const int prePad = lPre->loc.nb;
   const int nxPostPatch = (int) (this->xPatchSize() * powXScale); // TODO: store in HyPerConn::wPostPatches
   const int nyPostPatch = (int) (this->yPatchSize() * powYScale); // TODO: store in HyPerConn::wPostPatches
   //const int nfPostPatch = nf;

   // clique dimensions
   // a new set of cliques is centered on each pre-synaptic cell with radius nzPostPatch/2
   // TODO: precompute clique dimensions during CliqueConn::initialize
   int nyCliqueRadius = (int) (nyPostPatch / 2);
   int nxCliqueRadius = (int) (nxPostPatch / 2);
   int cliquePatchSize = (2 * nxCliqueRadius + 1) * (2 * nyCliqueRadius + 1) * nfPre;
   int cliqueSize = 1; // number of presynaptic cells in clique (traditional ANN uses 1)
   //int numKernels = conn->numDataPatches();  // per arbor?
   int numCliques = pow(cliquePatchSize, cliqueSize - 1);
   assert(numCliques == this->numberOfAxonalArborLists());

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
      if (cliqueSize > 1) {
         for (int kyCliqueExt = kyPreExt - nyCliqueRadius;
               kyCliqueExt < kyPreExt + nyCliqueRadius; kyCliqueExt++) {
            for (int kxCliqueExt = kxPreExt - nxCliqueRadius;
                  kxCliqueExt < kxPreExt + nxCliqueRadius; kxCliqueExt++) {
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
         cliqueActiveIndices[numActiveElements++] = kPreExt; // ensure at least 1 active index
      }

      // loop over all active combinations of size=cliqueSize-1 in clique radius
      int numActiveCliques = pow(numActiveElements, cliqueSize - 1);
      for (int kClique = 0; kClique < numActiveCliques; kClique++) {

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
                  - kPatchActive * pow(numActiveElements, cliqueSize - 1 - iProd - 1);

            // compute arborIndex for this clique element
            int kxCliqueExt = kxPos(kCliqueExt, nxPreExt, nyPreExt, nfPre);
            int kyCliqueExt = kyPos(kCliqueExt, nxPreExt, nyPreExt, nfPre);
            int kfClique = featureIndex(kCliqueExt, nxPreExt, nyPreExt, nfPre);
            int kxPatch = kxCliqueExt - kxPreExt + nxCliqueRadius;
            int kyPatch = kyCliqueExt - kyPreExt + nyCliqueRadius;
            unsigned int kArbor = kIndex(kxPatch, kyPatch, kfClique,
                  (2 * nxCliqueRadius + 1), (2*nyCliqueRadius + 1), nfPre);
            arborNdx += kArbor * pow(cliquePatchSize, cliqueSize - 1 - iProd - 1);
         }

         // receive weights input from clique (mostly copied from superclass method)
         // PVAxonalArbor * arbor = this->axonalArbor(kPreExt, arborNdx);
         PVPatch * dWPatch = pIncr[arborNdx][kPreExt]; // arbor->plasticIncr;
         size_t postOffset = getAPostOffset(kPreExt, arborNdx);
         const float * aPost = &post->getLayerData()[postOffset];
         int syPostExt = post->getLayerLoc()->nf
               * (post->getLayerLoc()->nx + 2 * post->getLayerLoc()->nb);

         // WARNING - assumes weight and GSyn patches from task same size
         //         - assumes patch stride sf is 1

         int nkPatch = dWPatch->nf * dWPatch->nx;
         int nyPatch = dWPatch->ny;
         int syPatch = dWPatch->sy;

         // TODO - unroll
         for (int y = 0; y < nyPatch; y++) {
            pvpatch_update_clique(
                  nkPatch,
                  (float *) (dWPatch->data + y * syPatch), cliqueProd, (float *) (aPost + y*syPostExt));
               }

            } // kClique
               } // kPreActive
   free(activeExt);
   free(cliqueActiveIndices);
   return PV_BREAK;

}
;
// calc_dW

int CliqueConn::updateWeights(int arborId)
{
   int status = KernelConn::updateWeights(arborId);
   assert((status == PV_SUCCESS) || (status == PV_BREAK));
   return PV_BREAK;

}
;
// updateWeights

/*
 int CliqueConn::normalizeWeights(PVPatch ** patches, int numPatches, int arborId){
 return PV_CONTINUE;};
 */

}//  namespace PV

int pvpatch_update_clique(int nk, float* RESTRICT dW, float aPre, float* RESTRICT aPost)
{
   int k;
   int err = 0;
   for (k = 0; k < nk; k++) {
      dW[k] += aPre * aPost[k];
   }
   return err;
}

