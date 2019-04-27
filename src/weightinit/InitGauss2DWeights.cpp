/*
 * InitGauss2DWeights.cpp
 *
 *  Created on: Apr 8, 2013
 *      Author: garkenyon
 */

#include "InitGauss2DWeights.hpp"
#include "components/StrengthParam.hpp"
#include "connections/BaseConnection.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

InitGauss2DWeights::InitGauss2DWeights(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

InitGauss2DWeights::InitGauss2DWeights() {}

InitGauss2DWeights::~InitGauss2DWeights() {}

void InitGauss2DWeights::initialize(char const *name, PVParams *params, Communicator const *comm) {
   InitWeights::initialize(name, params, comm);
}

int InitGauss2DWeights::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitWeights::ioParamsFillGroup(ioFlag);
   ioParam_aspect(ioFlag);
   ioParam_sigma(ioFlag);
   ioParam_rMax(ioFlag);
   ioParam_rMin(ioFlag);
   ioParam_numOrientationsPost(ioFlag);
   ioParam_numOrientationsPre(ioFlag);
   ioParam_deltaThetaMax(ioFlag);
   ioParam_thetaMax(ioFlag);
   ioParam_numFlanks(ioFlag);
   ioParam_flankShift(ioFlag);
   ioParam_rotate(ioFlag);
   ioParam_bowtieFlag(ioFlag);
   ioParam_bowtieAngle(ioFlag);
   return status;
}

void InitGauss2DWeights::ioParam_aspect(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "aspect", &mAspect, mAspect);
}

void InitGauss2DWeights::ioParam_sigma(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "sigma", &mSigma, mSigma);
}

void InitGauss2DWeights::ioParam_rMax(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "rMax", &mRMax, mRMax);
   if (ioFlag == PARAMS_IO_READ) {
      double rMaxd = (double)mRMax;
      mRMaxSquared = rMaxd * rMaxd;
   }
}

void InitGauss2DWeights::ioParam_rMin(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "rMin", &mRMin, mRMin);
   if (ioFlag == PARAMS_IO_READ) {
      double rMind = (double)mRMin;
      mRMinSquared = rMind * rMind;
   }
}

void InitGauss2DWeights::ioParam_numOrientationsPost(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "numOrientationsPost", &mNumOrientationsPost, -1);
}

void InitGauss2DWeights::ioParam_numOrientationsPre(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "numOrientationsPre", &mNumOrientationsPre, -1);
}

void InitGauss2DWeights::ioParam_deltaThetaMax(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "deltaThetaMax", &mDeltaThetaMax, mDeltaThetaMax);
}

void InitGauss2DWeights::ioParam_thetaMax(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "thetaMax", &mThetaMax, mThetaMax);
}

void InitGauss2DWeights::ioParam_numFlanks(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "numFlanks", &mNumFlanks, mNumFlanks);
}

void InitGauss2DWeights::ioParam_flankShift(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "flankShift", &mFlankShift, mFlankShift);
}

void InitGauss2DWeights::ioParam_rotate(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "rotate", &mRotate, mRotate);
}

void InitGauss2DWeights::ioParam_bowtieFlag(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "bowtieFlag", &mBowtieFlag, mBowtieFlag);
}

void InitGauss2DWeights::ioParam_bowtieAngle(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "bowtieFlag"));
   if (mBowtieFlag) {
      parameters()->ioParamValue(ioFlag, name, "bowtieAngle", &mBowtieAngle, mBowtieAngle);
   }
}

Response::Status
InitGauss2DWeights::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = InitWeights::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   pvAssert(mWeights);

   // set NumOrientationsPre and NumOrientationsPost if they were not set in params
   if (mNumOrientationsPost <= 0) {
      mNumOrientationsPost = mWeights->getGeometry()->getPostLoc().nf;
   }
   if (mNumOrientationsPre <= 0) {
      mNumOrientationsPre = mWeights->getGeometry()->getPreLoc().nf;
   }

   // Hacky way of handling Strength parameter, because weight normalizers and InitGauss2DWeights
   // use the parameter, but a connection that doesn't use either of those classes doesn't need it.
   // So that HyPerConn does not need to know any details of the InitWeights subclasses,
   // IntGauss2DWeights creates a StrengthParam object if one doesn't exist.
   // It can be added to the connection, but not to the AllObjects data member in the
   // CommunicateInitInfoMessage. So we need to get the StrenghtParam component from the
   // connection, instead of from AllObjects, as we usually would.
   auto objectTable           = message->mObjectTable;
   BaseConnection *parentConn = objectTable->findObject<BaseConnection>(getName());
   FatalIf(
         parentConn == nullptr,
         "%s could not find a connection named \"%s\".\n",
         getDescription_c(),
         getName());
   auto *strengthParam = parentConn->getComponentByType<StrengthParam>();
   if (strengthParam) {
      if (strengthParam->getInitInfoCommunicatedFlag()) {
         mStrength = strengthParam->getStrength();
         status    = status + Response::SUCCESS;
      }
      else {
         status = status + Response::POSTPONE;
      }
   }
   else {
      strengthParam = new StrengthParam(name, parameters(), mCommunicator);
      parentConn->addUniqueComponent(strengthParam);
      status = status + Response::POSTPONE;
   }
   return status;
}

void InitGauss2DWeights::calcWeights(int dataPatchIndex, int arborId) {
   calcOtherParams(dataPatchIndex);
   gauss2DCalcWeights(mWeights->getDataFromDataIndex(arborId, dataPatchIndex));
   // Weight does not depend on the arborId.
}

void InitGauss2DWeights::calcOtherParams(int patchIndex) {
   const int kfPre_tmp = kernelIndexCalculations(patchIndex);
   calculateThetas(kfPre_tmp, patchIndex);
}

void InitGauss2DWeights::gauss2DCalcWeights(float *dataStart) {
   int nfPatch = mWeights->getPatchSizeF();
   int nyPatch = mWeights->getPatchSizeY();
   int nxPatch = mWeights->getPatchSizeX();
   int sx      = mWeights->getGeometry()->getPatchStrideX();
   int sy      = mWeights->getGeometry()->getPatchStrideY();
   int sf      = mWeights->getGeometry()->getPatchStrideF();

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

            if (isSameLocAndSelf(xDelta, yDelta, fPost)) {
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
   mFeaturePre        = patchIndex % mWeights->getGeometry()->getPreLoc().nf;
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
   if ((mDeltaTheta = std::abs(mThetaPre - thPost)) > mDeltaThetaMax) {
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

bool InitGauss2DWeights::isSameLocAndSelf(float xDelta, float yDelta, int fPost) {
   bool sameLoc        = ((mFeaturePre == fPost) && (xDelta == 0.0f) && (yDelta == 0.0f));
   bool selfConnection = mWeights->getGeometry()->getSelfConnectionFlag();
   return sameLoc and selfConnection;
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
