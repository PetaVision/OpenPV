/*
 * InitGauss2DWeights.cpp
 *
 *  Created on: Apr 8, 2013
 *      Author: garkenyon
 */

#include "InitGauss2DWeights.hpp"

namespace PV {

InitGauss2DWeights::InitGauss2DWeights(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

InitGauss2DWeights::InitGauss2DWeights() { initialize_base(); }

InitGauss2DWeights::~InitGauss2DWeights() {}

int InitGauss2DWeights::initialize_base() { return PV_SUCCESS; }

int InitGauss2DWeights::initialize(char const *name, HyPerCol *hc) {
   int status = InitWeights::initialize(name, hc);
   return status;
}

int InitGauss2DWeights::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitWeights::ioParamsFillGroup(ioFlag);
   ioParam_aspect(ioFlag);
   ioParam_sigma(ioFlag);
   ioParam_rMax(ioFlag);
   ioParam_rMin(ioFlag);
   ioParam_strength(ioFlag);
   if (ioFlag != PARAMS_IO_READ) {
      // numOrientationsPost and numOrientationsPre are only meaningful if
      // relevant layers have nf>1, so reading those params and params that
      // depend on them is delayed until communicateParamsInfo, so that
      // pre&post will have been defined.
      ioParam_numOrientationsPost(ioFlag);
      ioParam_numOrientationsPre(ioFlag);
      ioParam_aspectRelatedParams(ioFlag);
   }
   return status;
}

void InitGauss2DWeights::ioParam_aspect(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "aspect", &mAspect, mAspect);
}

void InitGauss2DWeights::ioParam_sigma(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "sigma", &mSigma, mSigma);
}

void InitGauss2DWeights::ioParam_rMax(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "rMax", &mRMax, mRMax);
   if (ioFlag == PARAMS_IO_READ) {
      double rMaxd = (double)mRMax;
      mRMaxSquared = rMaxd * rMaxd;
   }
}

void InitGauss2DWeights::ioParam_rMin(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "rMin", &mRMin, mRMin);
   if (ioFlag == PARAMS_IO_READ) {
      double rMind = (double)mRMin;
      mRMinSquared = rMind * rMind;
   }
}

void InitGauss2DWeights::ioParam_strength(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "strength", &mStrength, mStrength /*default value*/, true /*warnIfAbsent*/);
}

void InitGauss2DWeights::ioParam_numOrientationsPost(enum ParamsIOFlag ioFlag) {
   pvAssert(mPostLayer);
   int nfPost = mPostLayer->getLayerLoc()->nf;
   if (nfPost > 1) {
      parent->parameters()->ioParamValue(
            ioFlag, name, "numOrientationsPost", &mNumOrientationsPost, nfPost);
   }
}

void InitGauss2DWeights::ioParam_numOrientationsPre(enum ParamsIOFlag ioFlag) {
   pvAssert(mPreLayer);
   int nfPre = mPreLayer->getLayerLoc()->nf;
   if (nfPre > 1) {
      parent->parameters()->ioParamValue(
            ioFlag, name, "numOrientationsPre", &mNumOrientationsPre, nfPre);
   }
}

void InitGauss2DWeights::ioParam_deltaThetaMax(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "deltaThetaMax", &mDeltaThetaMax, mDeltaThetaMax);
}

void InitGauss2DWeights::ioParam_thetaMax(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "thetaMax", &mThetaMax, mThetaMax);
}

void InitGauss2DWeights::ioParam_numFlanks(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "numFlanks", &mNumFlanks, mNumFlanks);
}

void InitGauss2DWeights::ioParam_flankShift(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "flankShift", &mFlankShift, mFlankShift);
}

void InitGauss2DWeights::ioParam_rotate(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "rotate", &mRotate, mRotate);
}

void InitGauss2DWeights::ioParam_bowtieFlag(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "bowtieFlag", &mBowtieFlag, mBowtieFlag);
}

void InitGauss2DWeights::ioParam_bowtieAngle(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "bowtieFlag"));
   if (mBowtieFlag) {
      parent->parameters()->ioParamValue(ioFlag, name, "bowtieAngle", &mBowtieAngle, mBowtieAngle);
   }
}

void InitGauss2DWeights::ioParam_aspectRelatedParams(enum ParamsIOFlag ioFlag) {
   if (needAspectParams()) {
      ioParam_deltaThetaMax(ioFlag);
      ioParam_thetaMax(ioFlag);
      ioParam_numFlanks(ioFlag);
      ioParam_flankShift(ioFlag);
      ioParam_rotate(ioFlag);
      ioParam_bowtieFlag(ioFlag);
      ioParam_bowtieAngle(ioFlag);
   }
}

bool InitGauss2DWeights::needAspectParams() {
   pvAssert(mPreLayer && mPostLayer);
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "aspect"));
   if (mPostLayer->getLayerLoc()->nf > 1) {
      pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "numOrientationsPost"));
   }
   if (mPreLayer->getLayerLoc()->nf > 1) {
      pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "numOrientationsPre"));
   }
   return (mAspect != 1.0f && ((mNumOrientationsPre <= 1) or (mNumOrientationsPost <= 1)));
}

int InitGauss2DWeights::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = InitWeights::communicateInitInfo(message);
   if (status != PV_SUCCESS) {
      return status;
   }
   // Handle params that depend on pre and post layers' nf
   ioParam_numOrientationsPost(PARAMS_IO_READ);
   ioParam_numOrientationsPre(PARAMS_IO_READ);
   ioParam_aspectRelatedParams(PARAMS_IO_READ);
   return status;
}

void InitGauss2DWeights::calcWeights(float *dataStart, int dataPatchIndex, int arborId) {
   calcOtherParams(dataPatchIndex);
   gauss2DCalcWeights(dataStart);
   // Weight does not depend on the arborId.
}

void InitGauss2DWeights::calcOtherParams(int patchIndex) {
   const int kfPre_tmp = kernelIndexCalculations(patchIndex);
   calculateThetas(kfPre_tmp, patchIndex);
}

void InitGauss2DWeights::gauss2DCalcWeights(float *dataStart) {
   // load necessary params:
   int nfPatch = mCallingConn->fPatchSize();
   int nyPatch = mCallingConn->yPatchSize();
   int nxPatch = mCallingConn->xPatchSize();
   int sx      = mCallingConn->xPatchStride();
   int sy      = mCallingConn->yPatchStride();
   int sf      = mCallingConn->fPatchStride();

   float normalizer = 1.0f / (2.0f * mSigma * mSigma);

   // loop over all post-synaptic cells in temporary patch
   for (int fPost = 0; fPost < nfPatch; fPost++) {
      float thPost = calcThPost(fPost);
      // TODO: add additional weight factor for difference between thPre and thPost
      if (checkThetaDiff(thPost)) {
         continue;
      }
      if (checkColorDiff(fPost)) {
         continue;
      }
      for (int jPost = 0; jPost < nyPatch; jPost++) {
         float yDelta = calcYDelta(jPost);
         for (int iPost = 0; iPost < nxPatch; iPost++) {
            float xDelta = calcXDelta(iPost);

            if (isSameLocOrSelf(xDelta, yDelta, fPost)) {
               continue;
            }

            // rotate the reference frame by th (change sign of thPost?)
            float xp = +xDelta * std::cos(thPost) + yDelta * std::sin(thPost);
            float yp = -xDelta * std::sin(thPost) + yDelta * std::cos(thPost);

            if (checkBowtieAngle(yp, xp)) {
               continue;
            }

            // include shift to flanks
            float d2  = xp * xp + (mAspect * (yp - mFlankShift) * mAspect * (yp - mFlankShift));
            int index = iPost * sx + jPost * sy + fPost * sf;

            dataStart[index] = 0.0f;
            if ((d2 <= mRMaxSquared) and (d2 >= mRMinSquared)) {
               dataStart[index] += mStrength * std::exp(-d2 * normalizer);
            }
            if (mNumFlanks > 1) {
               // shift in opposite direction
               d2 = xp * xp + (mAspect * (yp + mFlankShift) * mAspect * (yp + mFlankShift));
               if ((d2 <= mRMaxSquared) and (d2 >= mRMinSquared)) {
                  dataStart[index] += mStrength * std::exp(-d2 * normalizer);
               }
            }
         }
      }
   }
}

void InitGauss2DWeights::calculateThetas(int kfPre_tmp, int patchIndex) {
   mDeltaThetaPost    = PI * mThetaMax / (float)mNumOrientationsPost;
   mTheta0Post        = mRotate * mDeltaThetaPost / 2.0f;
   const float dthPre = PI * mThetaMax / (float)mNumOrientationsPre;
   const float th0Pre = mRotate * dthPre / 2.0f;
   mFeaturePre        = patchIndex % mPreLayer->getLayerLoc()->nf;
   assert(mFeaturePre == kfPre_tmp);
   const int iThPre = patchIndex % mNumOrientationsPre;
   mThetaPre        = th0Pre + iThPre * dthPre;
}

float InitGauss2DWeights::calcThPost(int fPost) {
   int oPost = fPost % mNumOrientationsPost;
   float thPost;
   if (mNumOrientationsPost == 1 && mNumOrientationsPre > 1) {
      thPost = mThetaPre;
   }
   else {
      thPost = mTheta0Post + oPost * mDeltaThetaPost;
   }
   return thPost;
}

bool InitGauss2DWeights::checkThetaDiff(float thPost) {
   if ((mDeltaTheta = std::abs(mThetaPre - mTheta0Post)) > mDeltaThetaMax) {
      // the following is obviously not ideal. But cocirc needs this mDeltaTheta:
      mDeltaTheta = (mDeltaTheta <= PI / 2.0f) ? mDeltaTheta : PI - mDeltaTheta;
      return true;
   }
   mDeltaTheta = (mDeltaTheta <= PI / 2.0f) ? mDeltaTheta : PI - mDeltaTheta;
   return false;
}

bool InitGauss2DWeights::checkColorDiff(int fPost) {
   int postColor = (int)(fPost / mNumOrientationsPost);
   int preColor  = (int)(mFeaturePre / mNumOrientationsPre);
   if (postColor != preColor) {
      return true;
   }
   return false;
}

bool InitGauss2DWeights::isSameLocOrSelf(float xDelta, float yDelta, int fPost) {
   bool sameLoc = ((mFeaturePre == fPost) && (xDelta == 0.0f) && (yDelta == 0.0f));
   if ((sameLoc) && (mPreLayer == mPostLayer)) {
      return true;
   }
   return false;
}

bool InitGauss2DWeights::checkBowtieAngle(float xp, float yp) {
   if (mBowtieFlag == 1) {
      float offaxis_angle = atan2(yp, xp);
      if (((offaxis_angle > mBowtieAngle) && (offaxis_angle < (PI - mBowtieAngle)))
          || ((offaxis_angle < -mBowtieAngle) && (offaxis_angle > (-PI + mBowtieAngle)))) {
         return true;
      }
   }
   return false;
}

} /* namespace PV */
