/*
 * NormalizeL2.cpp
 *
 *  Created on: Apr 8, 2013
 *      Author: pschultz
 */

#include "NormalizeL2.hpp"

namespace PV {

NormalizeL2::NormalizeL2() {
   initialize_base();
}

NormalizeL2::NormalizeL2(HyPerConn * callingConn) {
   initialize(callingConn);
}

int NormalizeL2::initialize_base() {
   return PV_SUCCESS;
}

int NormalizeL2::initialize(HyPerConn * callingConn) {
   return NormalizeBase::initialize(callingConn);
}

int NormalizeL2::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = NormalizeBase::ioParamsFillGroup(ioFlag);
   ioParam_minL2NormTolerated(ioFlag);
   return status;
}

void NormalizeL2::ioParam_minL2NormTolerated(enum ParamsIOFlag ioFlag) {
   parent()->ioParamValue(ioFlag, name, "minL2NormTolerated", &minL2NormTolerated, 0.0f, true/*warnIfAbsent*/);
}

int NormalizeL2::normalizeWeights(HyPerConn * conn) {
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
      if (conn->usingSharedWeights()==false) {
         fprintf(stderr, "NormalizeL2 error for connection \"%s\": normalizeFromPostPerspective is true but connection does not use shared weights.\n", conn->getName());
         exit(EXIT_FAILURE);
      }
      scale_factor = ((float) conn->postSynapticLayer()->getNumNeurons())/((float) conn->preSynapticLayer()->getNumNeurons());
   }
   scale_factor *= strength;

   status = NormalizeBase::normalizeWeights(conn); // applies normalize_cutoff threshold and symmetrizeWeights

   int nxp = conn->xPatchSize();
   int nyp = conn->yPatchSize();
   int nfp = conn->fPatchSize();
   int nxpShrunken = conn->getNxpShrunken();
   int nypShrunken = conn->getNypShrunken();
   int offsetShrunken = conn->getOffsetShrunken();
   int xPatchStride = conn->xPatchStride();
   int yPatchStride = conn->yPatchStride();
   int weights_per_patch = nxp*nyp*nfp;
   int nArbors = conn->numberOfAxonalArborLists();
   int numDataPatches = conn->getNumDataPatches();
   if (normalizeArborsIndividually) {
      for (int arborID = 0; arborID<nArbors; arborID++) {
         for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
            pvwdata_t * dataStartPatch = conn->get_wDataStart(arborID) + patchindex * weights_per_patch;
            double sumsq = 0.0;
            if (offsetShrunken == 0){
            	accumulateSumSquared(dataStartPatch, weights_per_patch, &sumsq);
            }
            else{
            	accumulateSumSquaredShrunken(dataStartPatch, &sumsq,
            			nxpShrunken, nypShrunken, offsetShrunken, xPatchStride, yPatchStride);
            }
            double l2norm = sqrt(sumsq);
            if (fabs(l2norm) <= minL2NormTolerated) {
               fprintf(stderr, "NormalizeL2 warning for normalizer \"%s\": sum of squares of weights in patch %d of arbor %d is within minL2NormTolerated=%f of zero.  Weights in this patch unchanged.\n", conn->getName(), patchindex, arborID, minL2NormTolerated);
               break;
            }
            normalizePatch(dataStartPatch, weights_per_patch, scale_factor/l2norm);
         }
      }
   }
   else {
      for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
         double sumsq = 0.0;
         for (int arborID = 0; arborID<nArbors; arborID++) {
            pvwdata_t * dataStartPatch = conn->get_wDataStart(arborID)+patchindex*weights_per_patch;
            if (offsetShrunken == 0){
            	accumulateSumSquared(dataStartPatch, weights_per_patch, &sumsq);
            }
            else{
            	accumulateSumSquaredShrunken(dataStartPatch, &sumsq,
            			nxpShrunken, nypShrunken, offsetShrunken, xPatchStride, yPatchStride);
            }
         }
         double l2norm = sqrt(sumsq);
         if (fabs(sumsq) <= minL2NormTolerated) {
            fprintf(stderr, "NormalizeL2 warning for connection \"%s\": sum of squares of weights in patch %d is within minL2NormTolerated=%f of zero.  Weights in this patch unchanged.\n", conn->getName(), patchindex, minL2NormTolerated);
            break;
         }
         for (int arborID = 0; arborID<nArbors; arborID++) {
            pvwdata_t * dataStartPatch = conn->get_wDataStart(arborID)+patchindex*weights_per_patch;
            normalizePatch(dataStartPatch, weights_per_patch, scale_factor/l2norm);
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

NormalizeL2::~NormalizeL2() {
}

} /* namespace PV */
