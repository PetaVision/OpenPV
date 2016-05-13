/*
 * InitOneToOneWeightsWithDelays.cpp
 *
 *  Created on: Sep 20, 2013
 *      Author: wchavez
 */

#include "InitOneToOneWeightsWithDelays.hpp"
#include "InitOneToOneWeightsWithDelaysParams.hpp"

namespace PV {

InitOneToOneWeightsWithDelays::InitOneToOneWeightsWithDelays(char const * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

InitOneToOneWeightsWithDelays::InitOneToOneWeightsWithDelays()
{
   initialize_base();
}

InitOneToOneWeightsWithDelays::~InitOneToOneWeightsWithDelays()
{
}

int InitOneToOneWeightsWithDelays::initialize_base() {
   return PV_SUCCESS;
}

int InitOneToOneWeightsWithDelays::initialize(char const * name, HyPerCol * hc) {
   int status = InitWeights::initialize(name, hc);
   return status;
}

InitWeightsParams * InitOneToOneWeightsWithDelays::createNewWeightParams() {
   InitWeightsParams * tempPtr = new InitOneToOneWeightsWithDelaysParams(name, parent);
   return tempPtr;
}

int InitOneToOneWeightsWithDelays::calcWeights(pvdata_t * dataStart, int patchIndex, int arborId) {

   InitOneToOneWeightsWithDelaysParams *weightParamPtr = dynamic_cast<InitOneToOneWeightsWithDelaysParams*>(weightParams);


   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(PV_FAILURE);
   }


   weightParamPtr->calcOtherParams(patchIndex);

   const float iWeight = weightParamPtr->getInitWeight();

   return createOneToOneConnectionWithDelays(dataStart, patchIndex, iWeight, weightParamPtr, arborId);
}

int InitOneToOneWeightsWithDelays::createOneToOneConnectionWithDelays(pvdata_t * dataStart, int dataPatchIndex, float iWeight, InitWeightsParams * weightParamPtr, int arborId) {

   const int nArbors = callingConn->numberOfAxonalArborLists();
   int k=weightParamPtr->getParentConn()->dataIndexToUnitCellIndex(dataPatchIndex);

   const int nfp = weightParamPtr->getnfPatch();
   const int nxp = weightParamPtr->getnxPatch();
   const int nyp = weightParamPtr->getnyPatch();

   const int sxp = weightParamPtr->getsx();
   const int syp = weightParamPtr->getsy();
   const int sfp = weightParamPtr->getsf();

   // clear all weights in patch
   memset(dataStart, 0, nxp*nyp*nfp);
   // then set the center point of the patch for each feature
   int x = (int) (nxp/2);
   int y = (int) (nyp/2);
   for (int f=0; f < nfp; f++) {
      dataStart[x * sxp + y * syp + f * sfp] = f == nArbors*k+arborId ? iWeight : 0;
   }

   return PV_SUCCESS;

}

BaseObject * createInitOneToOneWeightsWithDelays(char const * name, HyPerCol * hc) {
   return hc ? new InitOneToOneWeightsWithDelays(name, hc) : NULL;
}

} /* namespace PV */
