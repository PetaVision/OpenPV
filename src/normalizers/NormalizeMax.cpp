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

NormalizeMax::NormalizeMax(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

int NormalizeMax::initialize_base() {
   return PV_SUCCESS;
}

int NormalizeMax::initialize(const char * name, HyPerCol * hc) {
   return NormalizeMultiply::initialize(name, hc);
}


int NormalizeMax::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = NormalizeMultiply::ioParamsFillGroup(ioFlag);
   ioParam_minMaxTolerated(ioFlag);
   return status;
}

void NormalizeMax::ioParam_minMaxTolerated(enum ParamsIOFlag ioFlag) {
   parent()->ioParamValue(ioFlag, name, "minMaxTolerated", &minMaxTolerated, 0.0f, true/*warnIfAbsent*/);
}

int NormalizeMax::normalizeWeights() {
   int status = PV_SUCCESS;

   assert(numConnections >= 1);

   // All connections in the group must have the same values of sharedWeights, numArbors, and numDataPatches
   HyPerConn * conn0 = connectionList[0];

#ifdef OBSOLETE // Marked obsolete Dec 9, 2014.
#ifdef USE_SHMGET
#ifdef PV_USE_MPI
   if (conn->getShmgetFlag() && !conn->getShmgetOwner(0)) { // Assumes that all arbors are owned by the same process
      MPI_Barrier(conn->getParent()->icCommunicator()->communicator());
      return status;
   }
#endif // PV_USE_MPI
#endif // USE_SHMGET
#endif // OBSOLETE

   float scale_factor = 1.0f;
   if (normalizeFromPostPerspective) {
      if (conn0->usingSharedWeights()==false) {
         fprintf(stderr, "NormalizeMax error for connection \"%s\": normalizeFromPostPerspective is true but connection does not use shared weights.\n", getName());
         exit(EXIT_FAILURE);
      }
      scale_factor = ((float) conn0->postSynapticLayer()->getNumNeurons())/((float) conn0->preSynapticLayer()->getNumNeurons());
   }
   scale_factor *= strength;

   status = NormalizeMultiply::normalizeWeights(); // applies normalize_cutoff threshold and symmetrizeWeights

   int nArbors = conn0->numberOfAxonalArborLists();
   int numDataPatches = conn0->getNumDataPatches();
   if (normalizeArborsIndividually) {
      for (int arborID = 0; arborID<nArbors; arborID++) {
         for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
            float max = 0.0f;
            for (int c=0; c<numConnections; c++) {
               HyPerConn * conn = connectionList[c];
               int nxp = conn->xPatchSize();
               int nyp = conn->yPatchSize();
               int nfp = conn->fPatchSize();
               int weights_per_patch = nxp*nyp*nfp;
               pvwdata_t * dataStartPatch = conn->get_wDataHead(arborID, patchindex);
               accumulateMax(dataStartPatch, weights_per_patch, &max);
            }
            if (max <= minMaxTolerated) {
               fprintf(stderr, "Warning for NormalizeMax \"%s\": max of weights in patch %d of arbor %d is within minMaxTolerated=%f of zero.  Weights in this patch unchanged.\n", getName(), patchindex, arborID, minMaxTolerated);
               break; // TODO: continue?
            }
            for (int c=0; c<numConnections; c++) {
               HyPerConn * conn = connectionList[c];
               int nxp = conn->xPatchSize();
               int nyp = conn->yPatchSize();
               int nfp = conn->fPatchSize();
               int weights_per_patch = nxp*nyp*nfp;
               pvwdata_t * dataStartPatch = conn->get_wDataHead(arborID, patchindex);
               normalizePatch(dataStartPatch, weights_per_patch, scale_factor/max);
            }
         }
      }
   }
   else {
      for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
         float max = 0.0;
         for (int arborID = 0; arborID<nArbors; arborID++) {
            for (int c=0; c<numConnections; c++) {
               HyPerConn * conn = connectionList[c];
               int nxp = conn->xPatchSize();
               int nyp = conn->yPatchSize();
               int nfp = conn->fPatchSize();
               int weights_per_patch = nxp*nyp*nfp;
               pvwdata_t * dataStartPatch = conn->get_wDataHead(arborID,patchindex);
               accumulateMax(dataStartPatch, weights_per_patch, &max);
            }
         }
         if (max <= minMaxTolerated) {
            fprintf(stderr, "Warning for NormalizeMax \"%s\": max of weights in patch %d is within minMaxTolerated=%f of zero. Weights in this patch unchanged.\n", getName(), patchindex, minMaxTolerated);
            break; // TODO: continue?
         }
         for (int arborID = 0; arborID<nArbors; arborID++) {
            for (int c=0; c<numConnections; c++) {
               HyPerConn * conn = connectionList[c];
               int nxp = conn->xPatchSize();
               int nyp = conn->yPatchSize();
               int nfp = conn->fPatchSize();
               int weights_per_patch = nxp*nyp*nfp;
               pvwdata_t * dataStartPatch = conn->get_wDataHead(arborID,patchindex);
               normalizePatch(dataStartPatch, weights_per_patch, scale_factor/max);
            }
         }
      }
   }
#ifdef OBSOLETE // Marked obsolete Dec 9, 2014.
#ifdef USE_SHMGET
#ifdef PV_USE_MPI
   if (conn->getShmgetFlag()) {
      assert(conn->getShmgetOwner(0)); // Assumes that all arbors are owned by the same process
      MPI_Barrier(conn->getParent()->icCommunicator()->communicator());
   }
#endif // PV_USE_MPI
#endif // USE_SHMGET
#endif // OBSOLETE
   return status;
}

NormalizeMax::~NormalizeMax() {
}

} /* namespace PV */
