/*
 * InitRuleWeights.cpp
 *
 *  Created on: Aug 14, 2011
 *      Author: kpeterson
 */

#include "InitRuleWeights.hpp"
#include "InitRuleWeightsParams.hpp"

namespace PV {

InitRuleWeights::InitRuleWeights(HyPerConn * conn) {
   initialize_base();
   initialize(conn);
}

InitRuleWeights::InitRuleWeights()
{
   initialize_base();
}

InitRuleWeights::~InitRuleWeights()
{
}

int InitRuleWeights::initialize_base() {
   return PV_SUCCESS;
}

int InitRuleWeights::initialize(HyPerConn * conn) {
   int status = InitGauss2DWeights::initialize(conn);
   return status;
}

InitWeightsParams * InitRuleWeights::createNewWeightParams() {
   InitWeightsParams * tempPtr = new InitRuleWeightsParams(callingConn);
   return tempPtr;
}

int InitRuleWeights::calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId) {

   InitRuleWeightsParams *weightParamPtr = dynamic_cast<InitRuleWeightsParams*>(weightParams);


   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }


   weightParamPtr->calcOtherParams(patchIndex);

   ruleWeights(dataStart, weightParamPtr);

   return PV_SUCCESS; // return 1;

}

int InitRuleWeights::ruleWeights(/* PVPatch * patch */ pvdata_t * dataStart, InitRuleWeightsParams * weightParamPtr) {
   int nfPatch_tmp = weightParamPtr->getnfPatch();
   int nyPatch_tmp = weightParamPtr->getnyPatch();
   int nxPatch_tmp = weightParamPtr->getnxPatch();
   float strength=weightParamPtr->getStrength();
   int sx_tmp=weightParamPtr->getsx();
   int sy_tmp=weightParamPtr->getsy();
   int sf_tmp=weightParamPtr->getsf();
   int fPre = weightParamPtr->getFPre();

   // pvdata_t * dataStart = patch->data;

   // rule 16 (only 100 applies, left neighbor fires, I fire, all other patterns fire 0)
    // left (post view) -> right (pre view) -> 100 -> 000

    // loop over all post sy_tmpnaptic neurons in patch

    // initialize connections of OFF and ON cells to 0
    for (int f = 0; f < nfPatch_tmp; f++) {
       for (int j = 0; j < nyPatch_tmp; j++) {
          for (int i = 0; i < nxPatch_tmp; i++) {
             dataStart[i*sx_tmp + j*sy_tmp + f*sf_tmp] = 0;
          }
       }
    }

    // now set the actual pattern for rule 16 (0 0 0 1 0 0 0 0)

    // pre-sy_tmpnaptic neuron is an OFF cell
    if (fPre == 0) {
       for (int j = 0; j < nyPatch_tmp; j++) {
          // sub-rule 000 (first OFF cell fires)
          int f = 0;
          dataStart[0*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;
          dataStart[1*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;
          dataStart[2*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;

          // sub-rule 001 (second OFF cell fires)
          f = 2;
          dataStart[1*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;
          dataStart[2*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;

          // sub-rule 010 (third OFF cell fires)
          f = 4;
          dataStart[0*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;
          dataStart[2*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;

          // sub-rule 011 (fourth OFF cell fires)
          f = 6;
          dataStart[2*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;

          // sub-rule 100 (fifth _ON_ cell fires)
          f = 9;
          dataStart[0*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;
          dataStart[1*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;

          // sub-rule 101 (six OFF cell fires)
          f = 10;
          dataStart[1*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;

          // sub-rule 110 (seventh OFF cell fires)
          f = 12;
          dataStart[0*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;

          // sub-rule 111 (eighth OFF cell fires)
          f = 14;
       }
    }

    // pre-sy_tmpnaptic neuron is an ON cell
    if (fPre == 1) {
       for (int j = 0; j < nyPatch_tmp; j++) {
          // sub-rule 000 (first OFF cell fires)
          int f = 0;

          // sub-rule 001 (second OFF cell fires)
          f = 2;
          dataStart[0*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;

          // sub-rule 010 (third OFF cell fires)
          f = 4;
          dataStart[1*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;

          // sub-rule 011 (fourth OFF cell fires)
          f = 6;
          dataStart[0*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;
          dataStart[1*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;

          // sub-rule 100 (fifth _ON_ cell fires)
          f = 9;
          dataStart[2*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;

          // sub-rule 101 (six OFF cell fires)
          f = 10;
          dataStart[0*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;
          dataStart[2*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;

          // sub-rule 110 (seventh OFF cell fires)
          f = 12;
          dataStart[1*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;
          dataStart[2*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;

          // sub-rule 111 (eighth OFF cell fires)
          f = 14;
          dataStart[0*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;
          dataStart[1*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;
          dataStart[2*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;
       }
    }

    for (int f = 0; f < nfPatch_tmp; f++) {
       float factor = strength;
       for (int i = 0; i < nxPatch_tmp*nyPatch_tmp; i++) dataStart[f + i*nfPatch_tmp] *= factor;
    }

    return 0;

}

} /* namespace PV */
