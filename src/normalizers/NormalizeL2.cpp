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

NormalizeL2::NormalizeL2(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

int NormalizeL2::initialize_base() {
   return PV_SUCCESS;
}

int NormalizeL2::initialize(const char * name, HyPerCol * hc) {
   return NormalizeMultiply::initialize(name, hc);
}

int NormalizeL2::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = NormalizeMultiply::ioParamsFillGroup(ioFlag);
   ioParam_minL2NormTolerated(ioFlag);
   return status;
}

void NormalizeL2::ioParam_minL2NormTolerated(enum ParamsIOFlag ioFlag) {
   parent()->ioParamValue(ioFlag, name, "minL2NormTolerated", &minL2NormTolerated, 0.0f, true/*warnIfAbsent*/);
}

int NormalizeL2::normalizeWeights() {
   int status = PV_SUCCESS;

   assert(numConnections >= 1);

   // All connections in the group must have the same values of sharedWeights, numArbors, and numDataPatches
   HyPerConn * conn0 = connectionList[0];
#ifdef OBSOLETE // Marked obsolete Dec 15, 2014.  Moved to base class NormalizeMultiply.
   for (int c=1; c<numConnections; c++) {
      HyPerConn * conn = connectionList[c];
      if (conn->usingSharedWeights()!=conn0->usingSharedWeights()) {
         if (parent()->columnId() == 0) {
            fprintf(stderr, "Normalizer %s: All connections in the normalization group must have the same sharedWeights (Connection \"%s\" has %d; connection \"%s\" has %d).\n",
                  this->getName(), conn0->getName(), conn0->usingSharedWeights(), conn->getName(), conn->usingSharedWeights());
         }
         status = PV_FAILURE;
      }
      if (conn->numberOfAxonalArborLists() != conn0->numberOfAxonalArborLists()) {
         if (parent()->columnId() == 0) {
            fprintf(stderr, "Normalizer %s: All connections in the normalization group must have the same number of arbors (Connection \"%s\" has %d; connection \"%s\" has %d).\n",
                  this->getName(), conn0->getName(), conn0->numberOfAxonalArborLists(), conn->getName(), conn->numberOfAxonalArborLists());
         }
         status = PV_FAILURE;
      }
      if (conn->getNumDataPatches() != conn0->getNumDataPatches()) {
         if (parent()->columnId() == 0) {
            fprintf(stderr, "Normalizer %s: All connections in the normalization group must have the same number of data patches (Connection \"%s\" has %d; connection \"%s\" has %d).\n",
                  this->getName(), conn0->getName(), conn0->getNumDataPatches(), conn->getName(), conn->getNumDataPatches());
         }
         status = PV_FAILURE;
      }
      if (status==PV_FAILURE) {
         MPI_Barrier(parent()->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   }
#endif // OBSOLETE

   float scale_factor = 1.0f;
   if (normalizeFromPostPerspective) {
      if (conn0->usingSharedWeights()==false) {
         fprintf(stderr, "NormalizeL2 error for connection \"%s\": normalizeFromPostPerspective is true but connection does not use shared weights.\n", conn0->getName());
         exit(EXIT_FAILURE);
      }
      scale_factor = ((float) conn0->postSynapticLayer()->getNumNeurons())/((float) conn0->preSynapticLayer()->getNumNeurons());
   }
   scale_factor *= strength;

   status = NormalizeMultiply::normalizeWeights(); // applies normalize_cutoff threshold and rMinX,rMinY

   int nArbors = conn0->numberOfAxonalArborLists();
   int numDataPatches = conn0->getNumDataPatches();
   if (normalizeArborsIndividually) {
      for (int arborID = 0; arborID<nArbors; arborID++) {
         for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
            double sumsq = 0.0;
            for (int c=0; c<numConnections; c++) {
               HyPerConn * conn = connectionList[c];
               int nxp = conn->xPatchSize();
               int nyp = conn->yPatchSize();
               int nfp = conn->fPatchSize();
               int weights_per_patch = nxp*nyp*nfp;
               pvwdata_t * dataStartPatch = conn->get_wDataHead(arborID, patchindex);
               accumulateSumSquared(dataStartPatch, weights_per_patch, &sumsq);
            }
            double l2norm = sqrt(sumsq);
            if (fabs(l2norm) <= minL2NormTolerated) {
               fprintf(stderr, "Warning for NormalizeL2 \"%s\": sum of squares of weights in patch %d of arbor %d is within minL2NormTolerated=%f of zero.  Weights in this patch unchanged.\n", getName(), patchindex, arborID, minL2NormTolerated);
               break;
            }
            for (int c=0; c<numConnections; c++) {
               HyPerConn * conn = connectionList[c];
               int nxp = conn->xPatchSize();
               int nyp = conn->yPatchSize();
               int nfp = conn->fPatchSize();
               int weights_per_patch = nxp*nyp*nfp;
               pvwdata_t * dataStartPatch = conn0->get_wDataHead(arborID, patchindex);
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
               int nxp = conn->xPatchSize();
               int nyp = conn->yPatchSize();
               int nfp = conn->fPatchSize();
               int xPatchStride = conn->xPatchStride();
               int yPatchStride = conn->yPatchStride();
               int weights_per_patch = nxp*nyp*nfp;
               pvwdata_t * dataStartPatch = conn->get_wDataHead(arborID, patchindex);
               accumulateSumSquared(dataStartPatch, weights_per_patch, &sumsq);
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
               int nxp = conn->xPatchSize();
               int nyp = conn->yPatchSize();
               int nfp = conn->fPatchSize();
               int weights_per_patch = nxp*nyp*nfp;
               pvwdata_t * dataStartPatch = conn->get_wDataHead(arborID, patchindex);
               normalizePatch(dataStartPatch, weights_per_patch, scale_factor/l2norm);
            }
         }
      }
   }
   return status;
}

NormalizeL2::~NormalizeL2() {
}

} /* namespace PV */
