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

NormalizeL2::NormalizeL2(const char * name, HyPerCol * hc, HyPerConn ** connectionList, int numConns) {
   initialize(name, hc, connectionList, numConns);
}

int NormalizeL2::initialize_base() {
   return PV_SUCCESS;
}

int NormalizeL2::initialize(const char * name, HyPerCol * hc, HyPerConn ** connectionList, int numConns) {
   return NormalizeBase::initialize(name, hc, connectionList, numConns);
}

int NormalizeL2::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = NormalizeBase::ioParamsFillGroup(ioFlag);
   ioParam_minL2NormTolerated(ioFlag);
   return status;
}

void NormalizeL2::ioParam_minL2NormTolerated(enum ParamsIOFlag ioFlag) {
   parent()->ioParamValue(ioFlag, name, "minL2NormTolerated", &minL2NormTolerated, 0.0f, true/*warnIfAbsent*/);
}

int NormalizeL2::normalizeWeights() {
   int status = PV_SUCCESS;

   assert(numConnections >= 1);

   // TODO: need to ensure that all connections in connectionList have same sharedWeights,nxp,nyp,nfp,nxpShrunken,nypShrunken,offsetShrunken,sxp,syp,numArbors,numDataPatches,scale_factor
   HyPerConn * conn0 = connectionList[0];

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
      if (conn0->usingSharedWeights()==false) {
         fprintf(stderr, "NormalizeL2 error for connection \"%s\": normalizeFromPostPerspective is true but connection does not use shared weights.\n", conn0->getName());
         exit(EXIT_FAILURE);
      }
      scale_factor = ((float) conn0->postSynapticLayer()->getNumNeurons())/((float) conn0->preSynapticLayer()->getNumNeurons());
   }
   scale_factor *= strength;

   status = NormalizeBase::normalizeWeights(); // applies normalize_cutoff threshold and symmetrizeWeights

   int nxp = conn0->xPatchSize();
   int nyp = conn0->yPatchSize();
   int nfp = conn0->fPatchSize();
   int nxpShrunken = conn0->getNxpShrunken();
   int nypShrunken = conn0->getNypShrunken();
   int offsetShrunken = conn0->getOffsetShrunken();
   int xPatchStride = conn0->xPatchStride();
   int yPatchStride = conn0->yPatchStride();
   int weights_per_patch = nxp*nyp*nfp;
   int nArbors = conn0->numberOfAxonalArborLists();
   int numDataPatches = conn0->getNumDataPatches();
   if (normalizeArborsIndividually) {
      for (int arborID = 0; arborID<nArbors; arborID++) {
         for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
            double sumsq = 0.0;
            for (int c=0; c<numConnections; c++) {
               HyPerConn * conn = connectionList[c];
               pvwdata_t * dataStartPatch = conn0->get_wDataStart(arborID) + patchindex * weights_per_patch;
               if (offsetShrunken == 0){
                   accumulateSumSquared(dataStartPatch, weights_per_patch, &sumsq);
               }
               else{
                   accumulateSumSquaredShrunken(dataStartPatch, &sumsq,
                           nxpShrunken, nypShrunken, offsetShrunken, xPatchStride, yPatchStride);
               }
            }
            double l2norm = sqrt(sumsq);
            if (fabs(l2norm) <= minL2NormTolerated) {
               fprintf(stderr, "Warning for NormalizeL2 \"%s\": sum of squares of weights in patch %d of arbor %d is within minL2NormTolerated=%f of zero.  Weights in this patch unchanged.\n", getName(), patchindex, arborID, minL2NormTolerated);
               break;
            }
            for (int c=0; c<numConnections; c++) {
               HyPerConn * conn = connectionList[c];
               pvwdata_t * dataStartPatch = conn0->get_wDataStart(arborID) + patchindex * weights_per_patch;
               normalizePatch(dataStartPatch, weights_per_patch, scale_factor/l2norm);
            }
         }
      }
   }
   else {
      for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
         double sumsq = 0.0;
         for (int arborID = 0; arborID<nArbors; arborID++) {
            for (int c=0; c<numConnections; c++) {
               HyPerConn * conn = connectionList[c];
               pvwdata_t * dataStartPatch = conn->get_wDataStart(arborID)+patchindex*weights_per_patch;
               if (offsetShrunken == 0){
                   accumulateSumSquared(dataStartPatch, weights_per_patch, &sumsq);
               }
               else{
                   accumulateSumSquaredShrunken(dataStartPatch, &sumsq,
                           nxpShrunken, nypShrunken, offsetShrunken, xPatchStride, yPatchStride);
               }
            }
         }
         double l2norm = sqrt(sumsq);
         if (fabs(sumsq) <= minL2NormTolerated) {
            fprintf(stderr, "Warning for NormalizeL2 \"%s\": sum of squares of weights in patch %d is within minL2NormTolerated=%f of zero.  Weights in this patch unchanged.\n", getName(), patchindex, minL2NormTolerated);
            break;
         }
         for (int arborID = 0; arborID<nArbors; arborID++) {
            for (int c=0; c<numConnections; c++) {
               HyPerConn * conn = connectionList[c];
               pvwdata_t * dataStartPatch = conn->get_wDataStart(arborID)+patchindex*weights_per_patch;
               normalizePatch(dataStartPatch, weights_per_patch, scale_factor/l2norm);
            }
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
