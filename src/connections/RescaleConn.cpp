/*
 * RescaleConn.cpp
 *
 *  Created on: Apr 15, 2016
 *      Author: pschultz
 */

#include "RescaleConn.hpp"

namespace PV {

RescaleConn::RescaleConn(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

RescaleConn::RescaleConn() { initialize_base(); }

int RescaleConn::initialize_base() {
   scale = 1.0f;
   return PV_SUCCESS;
}

int RescaleConn::initialize(char const *name, HyPerCol *hc) {
   return IdentConn::initialize(name, hc);
}

int RescaleConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = IdentConn::ioParamsFillGroup(ioFlag);
   ioParam_scale(ioFlag);
   return status;
}

void RescaleConn::ioParam_scale(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "scale", &scale, scale /*default*/, true /*warn if absent*/);
}

int RescaleConn::deliverPresynapticPerspective(PVLayerCube const *activity, int arborID) {
   // Largely a duplicate of IdentConn::deliverPresynapticPerspective, except
   // for two lines inside for-loops with large numbers of iterations.
   // We're discussing ways to eliminate code duplication like this without
   // incurring added computational costs.  For now, leaving the duplicate
   // code as is.  --peteschultz, April 15, 2016

   // Check if we need to update based on connection's channel
   if (getChannel() == CHANNEL_NOUPDATE) {
      return PV_SUCCESS;
   }
   assert(post->getChannel(getChannel()));
   assert(getConvertToRateDeltaTimeFactor() == 1.0);

   const PVLayerLoc *preLoc  = preSynapticLayer()->getLayerLoc();
   const PVLayerLoc *postLoc = postSynapticLayer()->getLayerLoc();

   assert(arborID == 0); // IdentConn can only have one arbor
   const int numExtended = activity->numItems;

#ifdef DEBUG_OUTPUT
   int rank;
   MPI_Comm_rank(parent->getCommunicator()->communicator(), &rank);
   DebugLog(debugMessage);
   debugMessage.printf(
         "[%d]: HyPerLayr::recvSyn: neighbor=%d num=%d actv=%p this=%p conn=%p\n",
         rank,
         0,
         numExtended,
         activity,
         this,
         conn);
   debugMessage.flush();
#endif // DEBUG_OUTPUT

   for (int b = 0; b < parent->getNBatch(); b++) {
      float *activityBatch = activity->data
                             + b * (preLoc->nx + preLoc->halo.rt + preLoc->halo.lt)
                                     * (preLoc->ny + preLoc->halo.up + preLoc->halo.dn)
                                     * preLoc->nf;
      float *gSynPatchHeadBatch =
            post->getChannel(getChannel()) + b * postLoc->nx * postLoc->ny * postLoc->nf;

      if (activity->isSparse) {
         SparseList<float>::Entry const *activeIndicesBatch =
               (SparseList<float>::Entry *)activity->activeIndices
               + b * (preLoc->nx + preLoc->halo.rt + preLoc->halo.lt)
                       * (preLoc->ny + preLoc->halo.up + preLoc->halo.dn) * preLoc->nf;
         int numLoop = activity->numActive[b];
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int loopIndex = 0; loopIndex < numLoop; loopIndex++) {
            int kPre         = activeIndicesBatch[loopIndex].index;
            float a          = scale * activeIndicesBatch[loopIndex].value;
            PVPatch *weights = getWeights(kPre, arborID);
            if (weights->nx > 0 && weights->ny > 0) {
               int f = featureIndex(kPre, preLoc->nx, preLoc->ny, preLoc->nf); // Not taking halo
               // into account, but
               // for feature index,
               // shouldn't matter.
               float *postPatchStart = gSynPatchHeadBatch + getGSynPatchStart(kPre, arborID) + f;
               *postPatchStart += a;
            }
         }
      }
      else {
         PVLayerLoc const *loc = &activity->loc;
         PVHalo const *halo    = &loc->halo;
         int lineSizeExt       = (loc->nx + halo->lt + halo->rt) * loc->nf;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int y = 0; y < loc->ny; y++) {
            float *lineStartPreActivity =
                  &activityBatch[(y + halo->up) * lineSizeExt + halo->lt * loc->nf];
            int nk                   = loc->nx * loc->nf;
            float *lineStartPostGSyn = &gSynPatchHeadBatch[y * nk];
            for (int k = 0; k < nk; k++) {
               lineStartPostGSyn[k] += scale * lineStartPreActivity[k];
            }
         }
      }
   }
   return PV_SUCCESS;
}

RescaleConn::~RescaleConn() {}

} /* namespace PV */
