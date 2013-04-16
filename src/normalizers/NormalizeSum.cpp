/*
 * NormalizeSum.cpp
 *
 *  Created on: Apr 8, 2013
 *      Author: pschultz
 */

#include "NormalizeSum.hpp"

namespace PV {

NormalizeSum::NormalizeSum() {
   initialize_base();
}

NormalizeSum::NormalizeSum(const char * name, PVParams * params) {
   initialize(name, params);
}

int NormalizeSum::initialize_base() {
   return PV_SUCCESS;
}

int NormalizeSum::initialize(const char * name, PVParams * params) {
   int status = NormalizeBase::initialize(name, params);
   if (status == PV_SUCCESS) {
      status = setParams();
   }
   return status;
}

int NormalizeSum::setParams() {
   int status = NormalizeBase::setParams();
   readMinSumTolerated();
   return status;
}

int NormalizeSum::normalizeWeights(HyPerConn * conn) {
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
         fprintf(stderr, "NormalizeSum error for connection \"%s\": normalizeFromPostPerspective is true but connection is not a KernelConn\n", conn->getName());
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
            double sum = 0.0;
            accumulateSum(dataStartPatch, weights_per_patch, &sum);
            if (fabs(sum) <= minSumTolerated) {
               fprintf(stderr, "NormalizeSum warning for normalizer \"%s\": sum of weights in patch %d of arbor %d is within minSumTolerated=%f of zero. Weights in this patch unchanged.\n", conn->getName(), patchindex, arborID, minSumTolerated);
               break;
            }
            normalizePatch(dataStartPatch, weights_per_patch, scale_factor/sum);
         }
      }
   }
   else {
      for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
         double sum = 0.0;
         for (int arborID = 0; arborID<nArbors; arborID++) {
            pvdata_t * dataStartPatch = conn->get_wDataStart(arborID)+patchindex*weights_per_patch;
            accumulateSum(dataStartPatch, weights_per_patch, &sum);
         }
         if (fabs(sum) <= minSumTolerated) {
            fprintf(stderr, "NormalizeSum warning for connection \"%s\": sum of weights in patch %d is within minSumTolerated=%f of zero.  Weights in this patch unchanged.\n", conn->getName(), patchindex, minSumTolerated);
            break;

         }
         for (int arborID = 0; arborID<nArbors; arborID++) {
            pvdata_t * dataStartPatch = conn->get_wDataStart(arborID)+patchindex*weights_per_patch;
            normalizePatch(dataStartPatch, weights_per_patch, scale_factor/sum);
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

NormalizeSum::~NormalizeSum() {
}

} /* namespace PV */
