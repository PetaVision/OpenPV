/*
 * InitPoolWeights.cpp
 *
 *  Created on: Aug 13, 2011
 *      Author: kpeterson
 */

#include "InitPoolWeights.hpp"
#include "InitPoolWeightsParams.hpp"

namespace PV {

InitPoolWeights::InitPoolWeights(HyPerConn * conn)
{
   initialize_base();
   initialize(conn);
}

InitPoolWeights::InitPoolWeights()
{
   initialize_base();
}

InitPoolWeights::~InitPoolWeights()
{
}

int InitPoolWeights::initialize_base() {
   return PV_SUCCESS;
}

int InitPoolWeights::initialize(HyPerConn * conn) {
   int status = InitGauss2DWeights::initialize(conn);
   return status;
}

InitWeightsParams * InitPoolWeights::createNewWeightParams() {
   InitWeightsParams * tempPtr = new InitPoolWeightsParams(callingConn);
   return tempPtr;
}

int InitPoolWeights::calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId) {

   InitPoolWeightsParams *weightParamPtr = dynamic_cast<InitPoolWeightsParams*>(weightParams);


   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }


   weightParamPtr->calcOtherParams(patchIndex);

   poolWeights(dataStart, weightParamPtr);

   return PV_SUCCESS; // return 1;

}

int InitPoolWeights::poolWeights(/* PVPatch * patch */ pvdata_t * dataStart, InitPoolWeightsParams * weightParamPtr) {
   int nfPatch_tmp = weightParamPtr->getnfPatch();
   int nyPatch_tmp = weightParamPtr->getnyPatch();
   int nxPatch_tmp = weightParamPtr->getnxPatch();
   float strength=weightParamPtr->getStrength();
   int sx_tmp=weightParamPtr->getsx();
   int sy_tmp=weightParamPtr->getsy();
   int sf_tmp=weightParamPtr->getsf();

   // pvdata_t * w_tmp = patch->data;

   // initialize connections of OFF and ON cells to 0
   for (int f = 0; f < nfPatch_tmp; f++) {
      for (int j = 0; j < nyPatch_tmp; j++) {
         for (int i = 0; i < nxPatch_tmp; i++) {
            dataStart[i*nxPatch_tmp + j*nyPatch_tmp + f*nfPatch_tmp] = 0;
         }
      }
   }

   // connect an OFF cells to all OFF cells (and vice versa)

   for (int f = (weightParamPtr->getFPre() % 2); f < nfPatch_tmp; f += 2) {
      dataStart[0*sx_tmp + 0*sy_tmp + f*sf_tmp] = 1;
   }
   float factor = strength;
   for (int f = 0; f < nfPatch_tmp; f++) {
      for (int i = 0; i < nxPatch_tmp*nyPatch_tmp; i++) dataStart[f + i*nfPatch_tmp] *= factor;
   }

   return 0;
}

} /* namespace PV */
