/*
 * NormalizeMax.cpp
 *
 *  Created on: Apr 8, 2013
 *      Author: pschultz
 */

#include "NormalizeMax.hpp"

namespace PV {

NormalizeMax::NormalizeMax() {
   initialize_base();
}

NormalizeMax::NormalizeMax(const char * name, PVParams * params) {
   initialize(name, params);
}

int NormalizeMax::initialize_base() {
   return PV_SUCCESS;
}

int NormalizeMax::initialize(const char * name, PVParams * params) {
   return NormalizeBase::initialize(name, params);
}

int NormalizeMax::setParams() {
   int status = NormalizeBase::setParams();
   readMinSumTolerated();
   return status;
}

int NormalizeMax::normalizeWeights(HyPerConn * conn) {
   int status = PV_SUCCESS;
#ifdef USE_SHMGET
#ifdef PV_USE_MPI
   if (conn->getShmgetFlag() && !conn->getShmgetOwner(0)) { // Assumes that all arbors are owned by the same process
      MPI_Barrier(conn->getParent()->icCommunicator()->communicator());
      return status;
   }
#endif // PV_USE_MPI
#endif // USE_SHMGET

   float scale_factor = 1.0f;
   if (normalizeFromPostPerspective) {
      KernelConn * kconn = dynamic_cast<KernelConn *>(conn);
      if (!kconn) {
         fprintf(stderr, "NormalizeMax error for connection \"%s\": normalizeFromPostPerspective is true but connection is not a KernelConn\n", conn->getName());
         exit(EXIT_FAILURE);
      }
      scale_factor = ((float) conn->postSynapticLayer()->getNumNeurons())/((float) conn->preSynapticLayer()->getNumNeurons());
   }
   scale_factor *= strength;

   status = NormalizeBase::normalizeWeights(conn); // applies normalize_cutoff threshold and symmetrizeWeights

   int nxp = conn->xPatchSize();
   int nyp = conn->yPatchSize();
   int nfp = conn->fPatchSize();
   int weights_per_patch = nxp*nyp*nfp;
   int nArbors = conn->numberOfAxonalArborLists();
   int numDataPatches = conn->getNumDataPatches();
   if (normalizeArborsIndividually) {
      for (int arborID = 0; arborID<nArbors; arborID++) {
         for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
            pvdata_t * dataStartPatch = conn->get_wDataStart(arborID) + patchindex * weights_per_patch;
            float max = 0.0f;
            accumulateMax(dataStartPatch, weights_per_patch, &max);
            if (max <= minMaxTolerated) {
               fprintf(stderr, "NormalizeMax warning for normalizer \"%s\": sum of weights in patch %d of arbor %d is within minMaxTolerated=%f of zero.  Weights in this patch unchanged.\n", conn->getName(), patchindex, arborID, minMaxTolerated);
               break;
            }
            normalizePatch(dataStartPatch, weights_per_patch, scale_factor/max);
         }
      }
   }
   else {
      for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
         float max = 0.0;
         for (int arborID = 0; arborID<nArbors; arborID++) {
            pvdata_t * dataStartPatch = conn->get_wDataStart(arborID)+patchindex*weights_per_patch;
            accumulateMax(dataStartPatch, weights_per_patch, &max);
         }
         if (max <= minMaxTolerated) {
            fprintf(stderr, "NormalizeMax warning for connection \"%s\": sum of weights in patch %d is within minMaxTolerated=%f of zero. Weights in this patch unchanged.\n", conn->getName(), patchindex, minMaxTolerated);
            break;
         }
         for (int arborID = 0; arborID<nArbors; arborID++) {
            pvdata_t * dataStartPatch = conn->get_wDataStart(arborID)+patchindex*weights_per_patch;
            normalizePatch(dataStartPatch, weights_per_patch, scale_factor/max);
         }
      }
   }
#ifdef USE_SHMGET
#ifdef PV_USE_MPI
   if (conn->getShmgetFlag()) {
      assert(conn->getShmgetOwner(0)); // Assumes that all arbors are owned by the same process
      MPI_Barrier(conn->getParent()->icCommunicator()->communicator());
   }
#endif // PV_USE_MPI
#endif // USE_SHMGET
   return status;
}

NormalizeMax::~NormalizeMax() {
}

} /* namespace PV */
