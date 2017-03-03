/*
 * VaryingHyPerConn.cpp
 *
 *  Created on: Nov 10, 2011
 *      Author: pschultz
 */

#include "VaryingHyPerConn.hpp"

namespace PV {

VaryingHyPerConn::VaryingHyPerConn(const char *name, HyPerCol *hc) : HyPerConn() {
   initialize(name, hc);
}

VaryingHyPerConn::~VaryingHyPerConn() {}

int VaryingHyPerConn::initialize(const char *name, HyPerCol *hc) {
   return HyPerConn::initialize(name, hc);
}

int VaryingHyPerConn::allocateDataStructures() {
   HyPerConn::allocateDataStructures();
   return PV_SUCCESS;
}

int VaryingHyPerConn::updateWeights(int axonId) {
   int nPatch   = fPatchSize() * xPatchSize() * yPatchSize();
   for (int patchIndex = 0; patchIndex  < getNumDataPatches(); patchIndex++) {
      float *Wdata  = get_wDataHead(axonId, patchIndex);
      float *dWdata = get_dwDataHead(axonId, patchIndex);
      for (int k = 0; k < nPatch ; k++) {
         Wdata[k] += 1.0f;
      }
   }
   return PV_SUCCESS;
}

int VaryingHyPerConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   HyPerConn::ioParamsFillGroup(ioFlag);

   return 0;
}

} // end of namespace PV block
