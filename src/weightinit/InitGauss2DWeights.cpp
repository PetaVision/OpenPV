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
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

InitGauss2DWeights::InitGauss2DWeights() {}

InitGauss2DWeights::~InitGauss2DWeights() {}

void InitGauss2DWeights::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   InitWeights::initialize(params, defaults, comm);
}

int InitGauss2DWeights::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = InitWeights::ioParamsFillGroup(ioSwitch);
   ioParam_aspect(ioSwitch);
   ioParam_sigma(ioSwitch);
   ioParam_rMax(ioSwitch);
   ioParam_rMin(ioSwitch);
   ioParam_numOrientationsPost(ioSwitch);
   ioParam_numOrientationsPre(ioSwitch);
   ioParam_deltaThetaMax(ioSwitch);
   ioParam_thetaMax(ioSwitch);
   ioParam_numFlanks(ioSwitch);
   ioParam_flankShift(ioSwitch);
   ioParam_rotate(ioSwitch);
   ioParam_bowtieFlag(ioSwitch);
   ioParam_bowtieAngle(ioSwitch);
   return status;
}

void InitGauss2DWeights::ioParam_aspect(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "aspect", &mAspect);
}

void InitGauss2DWeights::ioParam_sigma(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "sigma", &mSigma);
}

void InitGauss2DWeights::ioParam_rMax(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "rMax", &mRMax);
   if (ioSwitch == ParamsIOSwitch::Read) {
      double rMaxd = (double)mRMax;
      mRMaxSquared = rMaxd * rMaxd;
   }
}

void InitGauss2DWeights::ioParam_rMin(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "rMin", &mRMin);
   if (ioSwitch == ParamsIOSwitch::Read) {
      double rMind = (double)mRMin;
      mRMinSquared = rMind * rMind;
   }
}

void InitGauss2DWeights::ioParam_numOrientationsPost(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "numOrientationsPost", &mNumOrientationsPost);
}

void InitGauss2DWeights::ioParam_numOrientationsPre(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "numOrientationsPre", &mNumOrientationsPre);
}

void InitGauss2DWeights::ioParam_deltaThetaMax(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "deltaThetaMax", &mDeltaThetaMax);
}

void InitGauss2DWeights::ioParam_thetaMax(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "thetaMax", &mThetaMax);
}

void InitGauss2DWeights::ioParam_numFlanks(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "numFlanks", &mNumFlanks);
}

void InitGauss2DWeights::ioParam_flankShift(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "flankShift", &mFlankShift);
}

void InitGauss2DWeights::ioParam_rotate(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "rotate", &mRotate);
}

void InitGauss2DWeights::ioParam_bowtieFlag(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "bowtieFlag", &mBowtieFlag);
}

void InitGauss2DWeights::ioParam_bowtieAngle(ParamsIOSwitch ioSwitch) {
   pvAssert(!mParamsIO->presentAndNotBeenRead("bowtieFlag"));
   if (mBowtieFlag) {
      mParamsIO->ioParam(ioSwitch, "bowtieAngle", &mBowtieAngle);
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
      strengthParam = new StrengthParam(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
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
