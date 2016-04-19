/*
 * InitSmartWeights.cpp
 *
 *  Created on: Aug 8, 2011
 *      Author: kpeterson
 */

#include "InitSmartWeights.hpp"

namespace PV {

InitSmartWeights::InitSmartWeights(char const * name, HyPerCol * hc) : InitWeights() {

   InitSmartWeights::initialize_base();
   InitSmartWeights::initialize(name, hc);
}

InitSmartWeights::InitSmartWeights()
{
   initialize_base();
}

InitSmartWeights::~InitSmartWeights()
{
}

int InitSmartWeights::initialize_base() {
   return PV_SUCCESS;
}

int InitSmartWeights::initialize(char const * name, HyPerCol * hc) {
   int status = InitWeights::initialize(name, hc);
   return status;
}

int InitSmartWeights::calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId) {
   //smart weights doesn't have any params to load and is too simple to
   //actually need to save anything to work on...

   smartWeights(dataStart, patchIndex, weightParams);
   return PV_SUCCESS; // return 1;
}

InitWeightsParams * InitSmartWeights::createNewWeightParams() {
   InitWeightsParams * tempPtr = new InitWeightsParams(name, parent);
   return tempPtr;
}

int InitSmartWeights::smartWeights(/* PVPatch * wp */ pvdata_t * dataStart, int k, InitWeightsParams *weightParams) {
   // pvdata_t * w = wp->data;

   const int nxp = weightParams->getnxPatch(); // wp->nx;
   const int nyp = weightParams->getnyPatch(); // wp->ny;
   const int nfp = weightParams->getnfPatch(); //wp->nf;

   const int sxp = weightParams->getsx(); //wp->sx;
   const int syp = weightParams->getsy(); //wp->sy;
   const int sfp = weightParams->getsf(); //wp->sf;

   // loop over all post-synaptic cells in patch
   for (int y = 0; y < nyp; y++) {
      for (int x = 0; x < nxp; x++) {
         for (int f = 0; f < nfp; f++) {
            dataStart[x * sxp + y * syp + f * sfp] = weightParams->getParentConn()->dataIndexToUnitCellIndex(k);
         }
      }
   }

   return 0;
}

BaseObject * createInitSmartWeights(char const * name, HyPerCol * hc) {
   return hc ? new InitSmartWeights(name, hc) : NULL;
}

} /* namespace PV */



