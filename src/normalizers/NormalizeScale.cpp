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

NormalizeScale::NormalizeScale(HyPerConn * callingConn) {
   initialize(callingConn);
}

int NormalizeScale::initialize_base() {
   return PV_SUCCESS;
}

int NormalizeScale::initialize(HyPerConn * callingConn) {
   return NormalizeBase::initialize(callingConn);
}

int NormalizeScale::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = NormalizeBase::ioParamsFillGroup(ioFlag);
   return status;
}

int NormalizeScale::normalizeWeights(HyPerConn * conn) {
   int status = PV_SUCCESS;
#ifdef USE_SHMGET
#ifdef PV_USE_MPI
   if (conn->getShmgetFlag() && !conn->getShmgetOwner(0)) { // Assumes that all arbors are owned by the same process
      MPI_Barrier(conn->getParent()->icCommunicator()->communicator());
      return status;
   }
#endif // PV_USE_MPI
#endif // USE_SHMGET

   float scale_factor = strength;

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
  
  for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
     for (int arborID = 0; arborID<nArbors; arborID++) {
        pvdata_t * dataStartPatch = conn->get_wDataStart(arborID)+patchindex*weights_per_patch;
        normalizePatch(dataStartPatch, weights_per_patch, scale_factor);
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

NormalizeScale::~NormalizeScale() {
}

} /* namespace PV */
