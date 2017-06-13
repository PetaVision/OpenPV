/*
 * InitCocircWeights.cpp
 *
 *  Created on: Aug 8, 2011
 *      Author: kpeterson
 */

#include "InitCocircWeights.hpp"

namespace PV {

InitCocircWeights::InitCocircWeights(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

InitCocircWeights::InitCocircWeights() { initialize_base(); }

InitCocircWeights::~InitCocircWeights() {}

int InitCocircWeights::initialize_base() { return PV_SUCCESS; }

int InitCocircWeights::initialize(char const *name, HyPerCol *hc) {
   int status = InitGauss2DWeights::initialize(name, hc);
   return status;
}

int InitCocircWeights::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitGauss2DWeights::ioParamsFillGroup(ioFlag);
   ioParam_sigmaCocirc(ioFlag);
   ioParam_sigmaKurve(ioFlag);
   ioParam_cocircSelf(ioFlag);
   ioParam_deltaRadiusCurvature(ioFlag);
   // Should minWeight, posKurveFlag, and saddleFlag be parameters?
   return status;
}

void InitCocircWeights::ioParam_sigmaCocirc(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "sigmaCocirc", &mSigmaCocirc, mSigmaCocirc);
}

void InitCocircWeights::ioParam_sigmaKurve(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "sigmaKurve", &mSigmaKurve, mSigmaKurve);
}

void InitCocircWeights::ioParam_cocircSelf(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "cocircSelf", &mCocircSelf, mCocircSelf);
}

void InitCocircWeights::ioParam_deltaRadiusCurvature(enum ParamsIOFlag ioFlag) {
   // from pv_common.h
   // // DK (1.0/(6*(NK-1)))   /*1/(sqrt(DX*DX+DY*DY)*(NK-1))*/         //  change in curvature
   parent->parameters()->ioParamValue(
         ioFlag, name, "deltaRadiusCurvature", &mDeltaRadiusCurvature, mDeltaRadiusCurvature);
}

void InitCocircWeights::calcWeights(float *dataStart, int patchIndex, int arborId) {
   calcOtherParams(patchIndex);
   nKurvePre  = mPreLayer->getLayerLoc()->nf / mNumOrientationsPre;
   nKurvePost = mPostLayer->getLayerLoc()->nf / mNumOrientationsPost;
   cocircCalcWeights(dataStart);
}

void InitCocircWeights::cocircCalcWeights(float *dataStart) {

   // load stored params:
   int nfPatch = mCallingConn->fPatchSize();
   int nyPatch = mCallingConn->yPatchSize();
   int nxPatch = mCallingConn->xPatchSize();
   int sx      = mCallingConn->xPatchStride();
   int sy      = mCallingConn->yPatchStride();
   int sf      = mCallingConn->fPatchStride();

   // loop over all post synaptic neurons in patch
   for (int kfPost = 0; kfPost < nfPatch; kfPost++) {
      float thPost = calcThPost(kfPost);

      calcKurvePostAndSigmaKurvePost(kfPost);

      if (checkThetaDiff(thPost)) {
         continue;
      }

      for (int jPost = 0; jPost < nyPatch; jPost++) {
         float yDelta = calcYDelta(jPost);
         for (int iPost = 0; iPost < nxPatch; iPost++) {
            float xDelta = calcXDelta(iPost);

            initializeDistChordCocircKurvePreAndKurvePost();

            if (calcDistChordCocircKurvePreNKurvePost(xDelta, yDelta, kfPost, thPost)) {
               continue;
            }

            // update weights based on calculated values:
            float weight = calculateWeight();
            if (weight < mMinWeight) {
               continue;
            }
            dataStart[iPost * sx + jPost * sy + kfPost * sf] = weight;
         }
      }
   }
}

float InitCocircWeights::calcKurvePostAndSigmaKurvePost(int kfPost) {
   int iKvPost       = kfPost % nKurvePost;
   float radKurvPost = calcKurveAndSigmaKurve(
         iKvPost, nKurvePost, sigma_kurve_post, kurvePost, iPosKurvePost, iSaddlePost);
   sigma_kurve_post2 = 2 * sigma_kurve_post * sigma_kurve_post;
   return radKurvPost;
}

float InitCocircWeights::calcKurveAndSigmaKurve(
      int kf,
      int &nKurve,
      float &sigma_kurve_temp,
      float &kurve_tmp,
      bool &iPosKurve,
      bool &iSaddle) {
   int iKv          = kf % nKurve;
   iPosKurve        = false;
   iSaddle          = false;
   float radKurv    = mDeltaRadiusCurvature + iKv * mDeltaRadiusCurvature;
   sigma_kurve_temp = mSigmaKurve * radKurv;

   kurve_tmp = (radKurv != 0.0f) ? 1 / radKurv : 1.0f;

   int iKvPostAdj = iKv;
   if (mPosKurveFlag) {
      assert(nKurve >= 2);
      iPosKurve = iKv >= (int)(nKurve / 2);
      if (mSaddleFlag) {
         assert(nKurve >= 4);
         iSaddle    = (iKv % 2 == 0) ? 0 : 1;
         iKvPostAdj = ((iKv % (nKurve / 2)) / 2);
      }
      else { // mSaddleFlag
         iKvPostAdj = (iKv % (nKurve / 2));
      }
   } // mPosKurveFlag
   radKurv   = mDeltaRadiusCurvature + iKvPostAdj * mDeltaRadiusCurvature;
   kurve_tmp = (radKurv != 0.0f) ? 1 / radKurv : 1.0f;
   return radKurv;
}

void InitCocircWeights::initializeDistChordCocircKurvePreAndKurvePost() {
   gDist      = 0.0f;
   gCocirc    = 1.0f;
   gKurvePre  = 1.0f;
   gKurvePost = 1.0f;
}

bool InitCocircWeights::calcDistChordCocircKurvePreNKurvePost(
      float xDelta,
      float yDelta,
      int kfPost,
      float thPost) {
   const float sigmaSquared = 2 * mSigma * mSigma;

   // rotate the reference frame by th
   float dxP = +xDelta * std::cos(mThetaPre) + yDelta * std::sin(mThetaPre);
   float dyP = -xDelta * std::sin(mThetaPre) + yDelta * std::cos(mThetaPre);

   // include shift to flanks
   float dyP_shift  = dyP - mFlankShift;
   float dyP_shift2 = dyP + mFlankShift;
   float d2         = dxP * dxP + mAspect * dyP * mAspect * dyP;
   float d2_shift   = dxP * dxP + (mAspect * (dyP_shift)*mAspect * (dyP_shift));
   float d2_shift2  = dxP * dxP + (mAspect * (dyP_shift2)*mAspect * (dyP_shift2));
   if (d2_shift <= mRMaxSquared) {
      addToGDist(std::exp(-d2_shift / sigmaSquared));
   }
   if (mNumFlanks > 1) {
      // include shift in opposite direction
      if (d2_shift2 <= mRMaxSquared) {
         addToGDist(std::exp(-d2_shift2 / sigmaSquared));
      }
   }
   if (gDist == 0.0f) {
      return true;
   }
   if (d2 == 0) {
      if (checkSameLoc(kfPost)) {
         return true;
      }
   }
   else { // d2 > 0

      // compute curvature of cocircular contour
      float cocircKurve_shift = d2_shift > 0 ? std::abs(2 * dyP_shift) / d2_shift : 0.0f;

      updateCocircNChord(thPost, dyP_shift, dxP, cocircKurve_shift, d2_shift);

      if (checkFlags(dyP_shift, dxP)) {
         return true;
      }

      // calculate values for gKurvePre and gKurvePost:
      updategKurvePreNgKurvePost(cocircKurve_shift);

      if (mNumFlanks > 1) {

         float cocircKurve_shift2 = d2_shift2 > 0 ? fabsf(2 * dyP_shift2) / d2_shift2 : 0.0f;

         updateCocircNChord(thPost, dyP_shift2, dxP, cocircKurve_shift2, d2_shift);

         if (checkFlags(dyP_shift2, dxP)) {
            return true;
         }

         // calculate values for gKurvePre and gKurvePost:
         updategKurvePreNgKurvePost(cocircKurve_shift2);
      }
   }

   return false;
}

void InitCocircWeights::addToGDist(float inc) { gDist += inc; }

bool InitCocircWeights::checkSameLoc(int kfPost) {
   const float mSigmaCocirc2 = 2 * mSigmaCocirc * mSigmaCocirc;
   bool sameLoc              = (mFeaturePre == kfPost);
   if ((!sameLoc) || (mCocircSelf)) {
      gCocirc = mSigmaCocirc > 0 ? expf(-mDeltaTheta * mDeltaTheta / mSigmaCocirc2)
                                 : expf(-mDeltaTheta * mDeltaTheta / mSigmaCocirc2) - 1.0f;
      if ((nKurvePre > 1) && (nKurvePost > 1)) {
         gKurvePre =
               expf(-(kurvePre - kurvePost) * (kurvePre - kurvePost)
                    / (sigma_kurve_pre2 + sigma_kurve_post2));
      }
   }
   else { // sameLoc && !cocircSelf
      gCocirc = 0.0f;
      return true;
   }
   return false;
}

void InitCocircWeights::updateCocircNChord(
      float thPost,
      float dyP_shift,
      float dxP,
      float cocircKurve_shift,
      float d2_shift) {

   const float sigmaCocirc2 = 2 * mSigmaCocirc * mSigmaCocirc;

   float atanx2_shift = mThetaPre + 2.0f * atan2f(dyP_shift, dxP); // preferred angle (rad)
   atanx2_shift += 2.0f * PI;
   atanx2_shift    = fmodf(atanx2_shift, PI);
   atanx2_shift    = fabsf(atanx2_shift - thPost);
   float chi_shift = atanx2_shift;
   if (chi_shift >= PI / 2.0f) {
      chi_shift = PI - chi_shift;
   }
   if (mNumOrientationsPre > 1 && mNumOrientationsPost > 1) {
      gCocirc = sigmaCocirc2 > 0 ? expf(-chi_shift * chi_shift / sigmaCocirc2)
                                 : expf(-chi_shift * chi_shift / sigmaCocirc2) - 1.0f;
   }
}

bool InitCocircWeights::checkFlags(float dyP_shift, float dxP) {
   if (mPosKurveFlag) {
      if (mSaddleFlag) {
         if ((iPosKurvePre) && !(iSaddlePre) && (dyP_shift < 0)) {
            return true;
         }
         if (!(iPosKurvePre) && !(iSaddlePre) && (dyP_shift > 0)) {
            return true;
         }
         if ((iPosKurvePre) && (iSaddlePre)
             && (((dyP_shift > 0) && (dxP < 0)) || ((dyP_shift > 0) && (dxP < 0)))) {
            return true;
         }
         if (!(iPosKurvePre) && (iSaddlePre)
             && (((dyP_shift > 0) && (dxP > 0)) || ((dyP_shift < 0) && (dxP < 0)))) {
            return true;
         }
      }
      else { // mSaddleFlag
         if ((iPosKurvePre) && (dyP_shift < 0)) {
            return true;
         }
         if (!(iPosKurvePre) && (dyP_shift > 0)) {
            return true;
         }
      }
   } // mPosKurveFlag
   return false;
}

void InitCocircWeights::updategKurvePreNgKurvePost(float cocircKurve_shift) {
   const float sigmaCocirc2 = 2 * mSigmaCocirc * mSigmaCocirc;

   gKurvePre =
         (nKurvePre > 1)
               ? std::exp(
                       -std::pow((cocircKurve_shift - std::abs(kurvePre)), 2.0f) / sigma_kurve_pre2)
               : 1.0f;
   gKurvePost = ((nKurvePre > 1) && (nKurvePost > 1) && (sigmaCocirc2 > 0))
                      ? std::exp(
                              -std::pow((cocircKurve_shift - std::abs(kurvePost)), 2.0f)
                              / sigma_kurve_post2)
                      : 1.0f;
}

float InitCocircWeights::calculateWeight() { return gDist * gKurvePre * gKurvePost * gCocirc; }

} /* namespace PV */
