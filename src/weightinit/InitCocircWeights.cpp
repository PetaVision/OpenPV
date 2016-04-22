/*
 * InitCocircWeights.cpp
 *
 *  Created on: Aug 8, 2011
 *      Author: kpeterson
 */

#include "InitCocircWeights.hpp"
#include "InitCocircWeightsParams.hpp"

namespace PV {

InitCocircWeights::InitCocircWeights(char const * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

InitCocircWeights::InitCocircWeights()
{
   initialize_base();
}

InitCocircWeights::~InitCocircWeights()
{
}

int InitCocircWeights::initialize_base() {
   return PV_SUCCESS;
}

int InitCocircWeights::initialize(char const * name, HyPerCol * hc) {
   int status = InitGauss2DWeights::initialize(name, hc);
   return status;
}

InitWeightsParams * InitCocircWeights::createNewWeightParams() {
   InitWeightsParams * tempPtr = new InitCocircWeightsParams(name, parent);
   return tempPtr;
}

int InitCocircWeights::calcWeights(pvdata_t * dataStart, int patchIndex, int arborId) {

   InitCocircWeightsParams *weightParamPtr = dynamic_cast<InitCocircWeightsParams*>(weightParams);


   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }


   weightParamPtr->calcOtherParams(patchIndex);

   cocircCalcWeights(dataStart, weightParamPtr);

   return PV_SUCCESS;

}

int InitCocircWeights::cocircCalcWeights(pvdata_t * w_tmp, InitCocircWeightsParams * weightParamPtr) {

   //load stored params:
   int nfPatch_tmp = weightParamPtr->getnfPatch();
   int nyPatch_tmp = weightParamPtr->getnyPatch();
   int nxPatch_tmp = weightParamPtr->getnxPatch();
   int sx_tmp=weightParamPtr->getsx();
   int sy_tmp=weightParamPtr->getsy();
   int sf_tmp=weightParamPtr->getsf();
   float min_weight=weightParamPtr->getmin_weight();

   // pvdata_t * w_tmp = patch->data;

   // loop over all post synaptic neurons in patch
   for (int kfPost = 0; kfPost < nfPatch_tmp; kfPost++) {
      float thPost = weightParamPtr->calcThPost(kfPost);



      weightParamPtr->calcKurvePostAndSigmaKurvePost(kfPost);

     if(weightParamPtr->checkThetaDiff(thPost)) continue;

     for (int jPost = 0; jPost < nyPatch_tmp; jPost++) {
        float yDelta = weightParamPtr->calcYDelta(jPost);
        for (int iPost = 0; iPost < nxPatch_tmp; iPost++) {
           float xDelta = weightParamPtr->calcXDelta(iPost);

           weightParamPtr->initializeDistChordCocircKurvePreAndKurvePost();


            if(calcDistChordCocircKurvePreNKurvePost(xDelta, yDelta, kfPost, weightParamPtr, thPost))
              continue;


            //update weights based on calculated values:
            float weight_tmp = weightParamPtr->calculateWeight();
            if (weight_tmp < min_weight) continue;
            w_tmp[iPost * sx_tmp + jPost * sy_tmp + kfPost * sf_tmp] = weight_tmp;



        }
     }
   }

   return 0;

}

bool InitCocircWeights::calcDistChordCocircKurvePreNKurvePost(
         float xDelta, float yDelta, int kfPost, InitCocircWeightsParams *weightParamPtr, float thPost) {
   const float aspect = weightParamPtr->getAspect();
   const float shift = weightParamPtr->getShift();
   const float sigma = weightParamPtr->getSigma();
   const float sigma2 = 2 * sigma * sigma;
   const double r2Max = weightParamPtr->getr2Max();
   const int numFlanks = weightParamPtr->getNumFlanks();
   float thetaPre = weightParamPtr->getthPre();

   // rotate the reference frame by th
   float dxP = +xDelta * cosf(thetaPre) + yDelta * sinf(thetaPre);
   float dyP = -xDelta * sinf(thetaPre) + yDelta * cosf(thetaPre);

   // include shift to flanks
   float dyP_shift = dyP - shift;
   float dyP_shift2 = dyP + shift;
   float d2 = dxP * dxP + aspect * dyP * aspect * dyP;
   float d2_shift = dxP * dxP + (aspect * (dyP_shift) * aspect * (dyP_shift));
   float d2_shift2 = dxP * dxP + (aspect * (dyP_shift2) * aspect * (dyP_shift2));
   if (d2_shift <= r2Max) {
      weightParamPtr->addToGDist(expf(-d2_shift / sigma2));
   }
   if (numFlanks > 1) {
      // include shift in opposite direction
      if (d2_shift2 <= r2Max) {
         weightParamPtr->addToGDist(expf(-d2_shift2 / sigma2));
      }
   }


   if (weightParamPtr->getGDist() == 0.0) return true;
   if (d2 == 0) {
     if(weightParamPtr->checkSameLoc(kfPost)) return true;

   }
   else { // d2 > 0


     // compute curvature of cocircular contour
     float cocircKurve_shift = d2_shift > 0 ? fabsf(2 * dyP_shift) / d2_shift
           : 0.0f;

      weightParamPtr->updateCocircNChord(
           thPost, dyP_shift, dxP,cocircKurve_shift, d2_shift);

     if(weightParamPtr->checkFlags(dyP_shift, dxP))
        return true;


      //calculate values for gKurvePre and gKurvePost:
      weightParamPtr->updategKurvePreNgKurvePost(cocircKurve_shift);

     if (numFlanks > 1) {

        float cocircKurve_shift2 = d2_shift2 > 0 ? fabsf(2 * dyP_shift2)
              / d2_shift2 : 0.0f;

        weightParamPtr->updateCocircNChord(
             thPost, dyP_shift2, dxP,cocircKurve_shift2, d2_shift);

        if(weightParamPtr->checkFlags(dyP_shift2, dxP))
           return true;


        //calculate values for gKurvePre and gKurvePost:
        weightParamPtr->updategKurvePreNgKurvePost(cocircKurve_shift2);

     }
   }

   return false;
}

BaseObject * createInitCocircWeights(char const * name, HyPerCol * hc) {
   return hc ? new InitCocircWeights(name, hc) : NULL;
}

} /* namespace PV */
