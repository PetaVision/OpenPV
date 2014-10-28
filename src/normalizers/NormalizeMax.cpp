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

NormalizeMax::NormalizeMax(const char * name, HyPerCol * hc, HyPerConn ** connectionList, int numConnections) {
   initialize(name, hc, connectionList, numConnections);
}

int NormalizeMax::initialize_base() {
   return PV_SUCCESS;
}

int NormalizeMax::initialize(const char * name, HyPerCol * hc, HyPerConn ** connectionList, int numConnections) {
   return NormalizeMultiply::initialize(name, hc, connectionList, numConnections);
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
         fprintf(stderr, "NormalizeMax error for connection \"%s\": normalizeFromPostPerspective is true but connection does not use shared weights.\n", getName());
         exit(EXIT_FAILURE);
      }
      scale_factor = ((float) conn0->postSynapticLayer()->getNumNeurons())/((float) conn0->preSynapticLayer()->getNumNeurons());
   }
   scale_factor *= strength;

   status = NormalizeMultiply::normalizeWeights(); // applies normalize_cutoff threshold and symmetrizeWeights

   int nxp = conn0->xPatchSize();
   int nyp = conn0->yPatchSize();
   int nfp = conn0->fPatchSize();
   int weights_per_patch = nxp*nyp*nfp;
   int nArbors = conn0->numberOfAxonalArborLists();
   int numDataPatches = conn0->getNumDataPatches();
   if (normalizeArborsIndividually) {
      for (int arborID = 0; arborID<nArbors; arborID++) {
         for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
            float max = 0.0f;
            for (int c=0; c<numConnections; c++) {
               HyPerConn * conn = connectionList[c];
               pvwdata_t * dataStartPatch = conn0->get_wDataStart(arborID) + patchindex * weights_per_patch;
               accumulateMax(dataStartPatch, weights_per_patch, &max);
            }
            if (max <= minMaxTolerated) {
               fprintf(stderr, "Warning for NormalizeMax \"%s\": max of weights in patch %d of arbor %d is within minMaxTolerated=%f of zero.  Weights in this patch unchanged.\n", getName(), patchindex, arborID, minMaxTolerated);
               break; // TODO: continue?
            }
            for (int c=0; c<numConnections; c++) {
               HyPerConn * conn = connectionList[c];
               pvwdata_t * dataStartPatch = conn0->get_wDataStart(arborID) + patchindex * weights_per_patch;
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
               pvwdata_t * dataStartPatch = conn0->get_wDataStart(arborID) + patchindex * weights_per_patch;
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
               pvwdata_t * dataStartPatch = conn0->get_wDataStart(arborID) + patchindex * weights_per_patch;
               normalizePatch(dataStartPatch, weights_per_patch, scale_factor/max);
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

NormalizeMax::~NormalizeMax() {
}

} /* namespace PV */
