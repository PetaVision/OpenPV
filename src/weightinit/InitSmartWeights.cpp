/*
 * InitSmartWeights.cpp
 *
 *  Created on: Aug 8, 2011
 *      Author: kpeterson
 */

#include "InitSmartWeights.hpp"

namespace PV {

InitSmartWeights::InitSmartWeights(char const *name, HyPerCol *hc) : InitWeights() {

   InitSmartWeights::initialize_base();
   InitSmartWeights::initialize(name, hc);
}

InitSmartWeights::InitSmartWeights() { initialize_base(); }

InitSmartWeights::~InitSmartWeights() {}

int InitSmartWeights::initialize_base() { return PV_SUCCESS; }

int InitSmartWeights::initialize(char const *name, HyPerCol *hc) {
   int status = InitWeights::initialize(name, hc);
   return status;
}

void InitSmartWeights::calcWeights(float *dataStart, int patchIndex, int arborId) {
   // smart weights doesn't have any params to load and is too simple to
   // actually need to save anything to work on...

   smartWeights(dataStart, patchIndex);
}

void InitSmartWeights::smartWeights(float *dataStart, int k) {

   const int nxp = mCallingConn->xPatchSize();
   const int nyp = mCallingConn->yPatchSize();
   const int nfp = mCallingConn->fPatchSize();

   const int sxp = mCallingConn->xPatchStride();
   const int syp = mCallingConn->yPatchStride();
   const int sfp = mCallingConn->fPatchStride();

   // loop over all post-synaptic cells in patch
   for (int y = 0; y < nyp; y++) {
      for (int x = 0; x < nxp; x++) {
         for (int f = 0; f < nfp; f++) {
            dataStart[x * sxp + y * syp + f * sfp] = mCallingConn->dataIndexToUnitCellIndex(k);
         }
      }
   }
}

} /* namespace PV */
