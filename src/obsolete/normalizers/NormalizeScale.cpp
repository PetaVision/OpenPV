/*
 * NormalizeScale.cpp (based on the code of NormalizeScale.cpp)
 *
 *  Created on: Mar 14, 2014
 *      Author: mpelko
 *
 * The name of normalize is misleading here. All this normalzition does is
 * multiply the weights by the strength parameter.
 *
 * Useful when doing Identity connection with non-one connection strength.
 * Usefull when you read the weights from a file and want to scale them
 * (without normalizations).
 */

#include "NormalizeScale.hpp"

namespace PV {

NormalizeScale::NormalizeScale() {
   initialize_base();
}

NormalizeScale::NormalizeScale(const char * name, HyPerCol * hc, HyPerConn ** connectionList, int numConnections) {
   initialize(name, hc, connectionList, numConnections);
}

int NormalizeScale::initialize_base() {
   return PV_SUCCESS;
}

int NormalizeScale::initialize(const char * name, HyPerCol * hc, HyPerConn ** connectionList, int numConnections) {
   return NormalizeMultiply::initialize(name, hc, connectionList, numConnections);
}

int NormalizeScale::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = NormalizeMultiply::ioParamsFillGroup(ioFlag);
   return status;
}

void NormalizeScale::ioParam_normalizeArborsIndividually(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      normalizeArborsIndividually = true;
      parent()->parameters()->handleUnnecessaryParameter(name, "normalizeArborsIndividually");
   }
}

int NormalizeScale::normalizeWeights() {
   int status = PV_SUCCESS;

   assert(numConnections >= 1);

   // All connections in the group must have the same values of sharedWeights, numArbors, and numDataPatches
   HyPerConn * conn0 = connectionList[0];

#ifdef USE_SHMGET
#ifdef PV_USE_MPI
   if (conn->getShmgetFlag() && !callingConn->getShmgetOwner(0)) { // Assumes that all arbors are owned by the same process
      MPI_Barrier(conn->getParent()->icCommunicator()->communicator());
      return status;
   }
#endif // PV_USE_MPI
#endif // USE_SHMGET

   float scale_factor = strength;

   status = NormalizeMultiply::normalizeWeights(); // applies normalize_cutoff threshold and symmetrizeWeights

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

   for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
      for (int arborID = 0; arborID<nArbors; arborID++) {
         for (int c=0; c<numConnections; c++) {
            HyPerConn * conn = connectionList[c];
            pvwdata_t * dataStartPatch = conn->get_wDataStart(arborID)+patchindex*weights_per_patch;
            normalizePatch(dataStartPatch, weights_per_patch, scale_factor);
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

NormalizeScale::~NormalizeScale() {
}

} /* namespace PV */
