/*
 * CliqueConn.cpp
 *
 *  Created on: Sep 8, 2011
 *      Author: gkenyon
 */

#include "CliqueConn.hpp"
#include "../normalizers/NormalizeBase.hpp"
int pvpatch_update_clique(int nk, float* RESTRICT v, float a, float* RESTRICT w);
int pvpatch_update_clique2(int nk, float* RESTRICT v, float a, float* RESTRICT w, float* RESTRICT m);

namespace PV {

CliqueConn::CliqueConn(const char * name, HyPerCol * hc) {
   CliqueConn::initialize_base();
   CliqueConn::initialize(name, hc);
}

int CliqueConn::initialize_base(){
   cliqueSize = 1;
   HyPerConn::initialize_base();
   return PV_SUCCESS;
}

int CliqueConn::initialize(const char * name, HyPerCol * hc) {
   HyPerConn::initialize(name, hc);
   return PV_SUCCESS;
}

// TODO: make sure code works in non-shared weight case
int CliqueConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   return status;
}

void CliqueConn::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   sharedWeights = true;
   if (ioFlag == PARAMS_IO_READ) {
      fileType = PVP_KERNEL_FILE_TYPE;
      parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights", true/*correctValue*/);
   }
}

void CliqueConn::ioParam_cliqueSize(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "cliqueSize", &cliqueSize, 1, true);
}

int CliqueConn::updateState(double time, double dt)
{
   update_timer->start();
   int status = PV_SUCCESS;
   if( !plasticityFlag ) {
      return status;
   }
   for(int axonID=0;axonID<numberOfAxonalArborLists();axonID++) {
      status = update_dW(axonID);  // don't clear dW, just accumulate changes
      if (status == PV_BREAK) {break;}
      assert(status == PV_SUCCESS);
   }

#ifdef PV_USE_MPI
   if (keepKernelsSynchronized_flag
         || parent->simulationTime() >= parent->getStopTime()-parent->getDeltaTime()) {
      for (int axonID = 0; axonID < numberOfAxonalArborLists(); axonID++) {
         status = reduceKernels(axonID); // combine partial changes in each column
         if (status == PV_BREAK) {
            break;
         }
         assert(status == PV_SUCCESS);
      }
   }
#endif // PV_USE_MPI

   // CliqueConn doesn't need to normalize.  Taking this out makes it easier to have normalizations over groups
   // // dW and W are the same so don't copy
   // if (parent->simulationTime() >= parent->getStopTime() - parent->getDeltaTime()) {
   //    if (normalizer) {
   //       normalizer->normalizeWeights(this);
   //    }
   // } //


   update_timer->stop();
   return PV_SUCCESS;
}

int CliqueConn::update_dW(int arborId)
{
   const PVLayerLoc * preLoc = this->getPre()->getLayerLoc();
   const int nfPre = preLoc->nf;

   const int nxPreExt = preLoc->nx + preLoc->halo.lt + preLoc->halo.rt;
   const int nyPreExt = preLoc->ny + preLoc->halo.dn + preLoc->halo.up;

   int syPostExt = post->getLayerLoc()->nf
         * (post->getLayerLoc()->nx + post->getLayerLoc()->halo.lt + post->getLayerLoc()->halo.rt); // compute just once

   // if pre and post denote the same layers, make a clone of size PVPatch to hold temporary activity values
   // in order to eliminate generalized self-interactions
   // note that during learning, pre and post may be separate instantiations
   bool self_flag = this->getSelfFlag(); //this->getPre() == this->getPost();
   pvdata_t * a_post_mask = NULL;
   const int a_post_size = nfp * nxp * nyp;
   a_post_mask = (pvdata_t *) calloc(a_post_size, sizeof(pvdata_t));
   assert(a_post_mask != NULL);
   // get linear index of cell at center of patch for self_flag == true
   const int k_post_self = (int) (a_post_size / 2);  // if self_flag == true, a_post_size should be odd
   for (int k_post = 0; k_post < a_post_size; k_post++) {
      a_post_mask[k_post] = 1;
   }

   int delay = getDelay(arborId);

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
   //int numKernels = conn->numDataPatches();  // per arbor?
   int numCliques = (int)pow(cliquePatchSize, cliqueSize - 1);
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

         PVPatch * wPatch = this->getWeights(kPreExt,arborNdx); // wPatches[arborNdx][kPreExt]; // arbor->plasticIncr;
         size_t postOffset = getAPostOffset(kPreExt, arborNdx);
         const float * aPost = &post->getLayerData()[postOffset];

         size_t dW_offset = wPatch->offset; // dWPatch->data - dW_head;
         pvwdata_t * dwData = get_dwData(arborNdx,kPreExt);
         // WARNING - assumes weight and GSyn patches from task same size
         //         - assumes patch stride sf is 1

         int nkPatch = nfp * wPatch->nx;
         int nyPatch = wPatch->ny;
         int syPatch = syp;

         // TODO - unroll
         for (int y = 0; y < nyPatch; y++) {
            pvpatch_update_clique2(
                  nkPatch,
                  (float *) (dwData + y * syPatch),
                  cliqueProd,
                  (float *) (aPost + y*syPostExt),
                  (float *) (a_post_mask + dW_offset + y * syPatch));
               }

            } // kClique
   } // kPreActive
   free(cliqueActiveIndices);
   free(activeExt);
   free(a_post_mask);
   return PV_BREAK;

}
;
// calc_dW

int CliqueConn::updateWeights(int arborId)
{
   int status = HyPerConn::updateWeights(arborId);
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

int pvpatch_update_clique2(int nk, float* RESTRICT dW, float aPre, float* RESTRICT aPost, float* RESTRICT a_mask)
{
   int k;
   int err = 0;
   for (k = 0; k < nk; k++) {
      dW[k] += aPre * aPost[k] * a_mask[k];
   }
   return err;
}

