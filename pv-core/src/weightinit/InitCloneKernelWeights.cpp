/*
 * InitCloneKernelWeights.cpp
 *
 *      Author: Pete Schultz
 */

#include "InitCloneKernelWeights.hpp"

namespace PV {

InitCloneKernelWeights::InitCloneKernelWeights() {
   initialize_base();
}

InitCloneKernelWeights::~InitCloneKernelWeights() {
}

int InitCloneKernelWeights::initialize_base() {
   return PV_SUCCESS;
}

int InitCloneKernelWeights::calcWeights(/*PVPatch * patch*/ pvdata_t * patchDataStart, int patchIndex, int arborId) {
   // Don't modify anything; CloneKernelConn doesn't allocate new weights,
   // but points weight patches at an existing connection's patch.
   return PV_SUCCESS;
}

}  // end namespace PV


