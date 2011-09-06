/*
 * InitRuleWeights.cpp
 *
 *  Created on: Aug 14, 2011
 *      Author: kpeterson
 */

#include "InitRuleWeights.hpp"
#include "InitRuleWeightsParams.hpp"

namespace PV {

InitRuleWeights::InitRuleWeights()
{
   initialize_base();
}

InitRuleWeights::~InitRuleWeights()
{
   // TODO Auto-generated destructor stub
}


int InitRuleWeights::initialize_base() {
   return PV_SUCCESS;
}

InitWeightsParams * InitRuleWeights::createNewWeightParams(HyPerConn * callingConn) {
   InitWeightsParams * tempPtr = new InitRuleWeightsParams(callingConn);
   return tempPtr;
}

int InitRuleWeights::calcWeights(PVPatch * patch, int patchIndex, int arborId,
                                   InitWeightsParams *weightParams) {

   InitRuleWeightsParams *weightParamPtr = dynamic_cast<InitRuleWeightsParams*>(weightParams);


   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }


   weightParamPtr->calcOtherParams(patch, patchIndex);

   ruleWeights(patch, weightParamPtr);

   return 1;

}

int InitRuleWeights::ruleWeights(PVPatch * patch, InitRuleWeightsParams * weightParamPtr) {
   int nfPatch_tmp = weightParamPtr->getnfPatch_tmp();
   int nyPatch_tmp = weightParamPtr->getnyPatch_tmp();
   int nxPatch_tmp = weightParamPtr->getnxPatch_tmp();
   float strength=weightParamPtr->getStrength();
   int sx_tmp=weightParamPtr->getsx_tmp();
   int sy_tmp=weightParamPtr->getsy_tmp();
   int sf_tmp=weightParamPtr->getsf_tmp();
   int fPre = weightParamPtr->getFPre();

   pvdata_t * w_tmp = patch->data;
   // rule 16 (only 100 applies, left neighbor fires, I fire, all other patterns fire 0)
    // left (post view) -> right (pre view) -> 100 -> 000

    // loop over all post sy_tmpnaptic neurons in patch

    // initialize connections of OFF and ON cells to 0
    for (int f = 0; f < nfPatch_tmp; f++) {
       for (int j = 0; j < nyPatch_tmp; j++) {
          for (int i = 0; i < nxPatch_tmp; i++) {
             w_tmp[i*sx_tmp + j*sy_tmp + f*sf_tmp] = 0;
          }
       }
    }

    // now set the actual pattern for rule 16 (0 0 0 1 0 0 0 0)

    // pre-sy_tmpnaptic neuron is an OFF cell
    if (fPre == 0) {
       for (int j = 0; j < nyPatch_tmp; j++) {
          // sub-rule 000 (first OFF cell fires)
          int f = 0;
          w_tmp[0*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;
          w_tmp[1*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;
          w_tmp[2*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;

          // sub-rule 001 (second OFF cell fires)
          f = 2;
          w_tmp[1*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;
          w_tmp[2*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;

          // sub-rule 010 (third OFF cell fires)
          f = 4;
          w_tmp[0*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;
          w_tmp[2*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;

          // sub-rule 011 (fourth OFF cell fires)
          f = 6;
          w_tmp[2*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;

          // sub-rule 100 (fifth _ON_ cell fires)
          f = 9;
          w_tmp[0*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;
          w_tmp[1*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;

          // sub-rule 101 (six OFF cell fires)
          f = 10;
          w_tmp[1*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;

          // sub-rule 110 (seventh OFF cell fires)
          f = 12;
          w_tmp[0*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;

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
          w_tmp[0*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;

          // sub-rule 010 (third OFF cell fires)
          f = 4;
          w_tmp[1*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;

          // sub-rule 011 (fourth OFF cell fires)
          f = 6;
          w_tmp[0*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;
          w_tmp[1*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;

          // sub-rule 100 (fifth _ON_ cell fires)
          f = 9;
          w_tmp[2*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;

          // sub-rule 101 (six OFF cell fires)
          f = 10;
          w_tmp[0*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;
          w_tmp[2*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;

          // sub-rule 110 (seventh OFF cell fires)
          f = 12;
          w_tmp[1*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;
          w_tmp[2*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;

          // sub-rule 111 (eighth OFF cell fires)
          f = 14;
          w_tmp[0*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;
          w_tmp[1*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;
          w_tmp[2*sx_tmp + j*sy_tmp + f*sf_tmp] = 1;
       }
    }

    for (int f = 0; f < nfPatch_tmp; f++) {
       float factor = strength;
       for (int i = 0; i < nxPatch_tmp*nyPatch_tmp; i++) w_tmp[f + i*nfPatch_tmp] *= factor;
    }

    return 0;

}

} /* namespace PV */
