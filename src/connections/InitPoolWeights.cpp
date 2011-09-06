/*
 * InitPoolWeights.cpp
 *
 *  Created on: Aug 13, 2011
 *      Author: kpeterson
 */

#include "InitPoolWeights.hpp"
#include "InitPoolWeightsParams.hpp"

namespace PV {

InitPoolWeights::InitPoolWeights()
{
   initialize_base();
}

InitPoolWeights::~InitPoolWeights()
{
   // TODO Auto-generated destructor stub
}

int InitPoolWeights::initialize_base() {
   return PV_SUCCESS;
}

InitWeightsParams * InitPoolWeights::createNewWeightParams(HyPerConn * callingConn) {
   InitWeightsParams * tempPtr = new InitPoolWeightsParams(callingConn);
   return tempPtr;
}

int InitPoolWeights::calcWeights(PVPatch * patch, int patchIndex, int arborId,
                                   InitWeightsParams *weightParams) {

   InitPoolWeightsParams *weightParamPtr = dynamic_cast<InitPoolWeightsParams*>(weightParams);


   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }


   weightParamPtr->calcOtherParams(patch, patchIndex);

   poolWeights(patch, weightParamPtr);

   return 1;

}

int InitPoolWeights::poolWeights(PVPatch * patch, InitPoolWeightsParams * weightParamPtr) {
   int nfPatch_tmp = weightParamPtr->getnfPatch_tmp();
   int nyPatch_tmp = weightParamPtr->getnyPatch_tmp();
   int nxPatch_tmp = weightParamPtr->getnxPatch_tmp();
   float strength=weightParamPtr->getStrength();
   int sx_tmp=weightParamPtr->getsx_tmp();
   int sy_tmp=weightParamPtr->getsy_tmp();
   int sf_tmp=weightParamPtr->getsf_tmp();

   pvdata_t * w_tmp = patch->data;

   // initialize connections of OFF and ON cells to 0
   for (int f = 0; f < nfPatch_tmp; f++) {
      for (int j = 0; j < nyPatch_tmp; j++) {
         for (int i = 0; i < nxPatch_tmp; i++) {
            w_tmp[i*nxPatch_tmp + j*nyPatch_tmp + f*nfPatch_tmp] = 0;
         }
      }
   }

   // connect an OFF cells to all OFF cells (and vice versa)

   for (int f = (weightParamPtr->getFPre() % 2); f < nfPatch_tmp; f += 2) {
      w_tmp[0*sx_tmp + 0*sy_tmp + f*sf_tmp] = 1;
   }
   float factor = strength;
   for (int f = 0; f < nfPatch_tmp; f++) {
      for (int i = 0; i < nxPatch_tmp*nyPatch_tmp; i++) w_tmp[f + i*nfPatch_tmp] *= factor;
   }

   return 0;
}

} /* namespace PV */
