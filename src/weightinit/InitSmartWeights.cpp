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

int InitSmartWeights::calcWeights(
      /* PVPatch * patch */ pvdata_t *dataStart,
      int patchIndex,
      int arborId) {
   // smart weights doesn't have any params to load and is too simple to
   // actually need to save anything to work on...

   smartWeights(dataStart, patchIndex, weightParams);
   return PV_SUCCESS;
}

InitWeightsParams *InitSmartWeights::createNewWeightParams() {
   InitWeightsParams *tempPtr = new InitWeightsParams(name, parent);
   return tempPtr;
}

int InitSmartWeights::smartWeights(
      /* PVPatch * wp */ pvdata_t *dataStart,
      int k,
      InitWeightsParams *weightParams) {

   const int nxp = weightParams->getnxPatch();
   const int nyp = weightParams->getnyPatch();
   const int nfp = weightParams->getnfPatch();

   const int sxp = weightParams->getsx();
   const int syp = weightParams->getsy();
   const int sfp = weightParams->getsf();

   // loop over all post-synaptic cells in patch
   for (int y = 0; y < nyp; y++) {
      for (int x = 0; x < nxp; x++) {
         for (int f = 0; f < nfp; f++) {
            dataStart[x * sxp + y * syp + f * sfp] =
                  weightParams->getParentConn()->dataIndexToUnitCellIndex(k);
         }
      }
   }

   return 0;
}

} /* namespace PV */
