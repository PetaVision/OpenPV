/*
 * InitGauss2DWeightsParams.cpp
 *
 *  Created on: Aug 10, 2011
 *      Author: kpeterson
 */

#include "InitGauss2DWeightsParams.hpp"

namespace PV {

InitGauss2DWeightsParams::InitGauss2DWeightsParams()
{
   initialize_base();
}
InitGauss2DWeightsParams::InitGauss2DWeightsParams(const char * name, HyPerCol * hc)
                     : InitWeightsParams() {
   initialize_base();
   initialize(name, hc);
}

InitGauss2DWeightsParams::~InitGauss2DWeightsParams()
{
}

int InitGauss2DWeightsParams::initialize_base() {

   // default values
   aspect = 1.0f; // circular (not line oriented)
   sigma = 0.8f;
   rMax = 1.4f;
   rMin = 0.0f;
   strength = 1.0f;
   setDeltaThetaMax(2.0f * PI);  // max difference in orientation in units of PI
   setThetaMax(1.0f); // max orientation in units of PI
   numFlanks = 1;
   shift = 0.0f;
   setRotate(0.0f);  // rotate so that axis isn't aligned
   bowtieFlag = 0.0f;  // flag for setting bowtie angle
   bowtieAngle = PI * 2.0f;  // bowtie angle
   numOrientationsPost = 1;
   numOrientationsPre = 1;
   deltaTheta=0;
   return PV_SUCCESS;
}

int InitGauss2DWeightsParams::initialize(const char * name, HyPerCol * hc) {
   InitWeightsParams::initialize(name, hc);

   PVParams * params = parent->parameters();
   int status = PV_SUCCESS;

   return status;

}

int InitGauss2DWeightsParams::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitWeightsParams::ioParamsFillGroup(ioFlag);
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

void InitGauss2DWeightsParams::ioParam_aspect(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "aspect", &aspect, aspect);
}

void InitGauss2DWeightsParams::ioParam_sigma(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "sigma", &sigma, sigma);
}

void InitGauss2DWeightsParams::ioParam_rMax(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "rMax", &rMax, rMax);
   if (ioFlag==PARAMS_IO_READ) {
      double rMaxd = (double) rMax;
      r2Max = rMaxd*rMaxd;
   }
}

void InitGauss2DWeightsParams::ioParam_rMin(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "rMin", &rMin, rMin);
   if (ioFlag==PARAMS_IO_READ) {
      double rMind = (double) rMin;
      r2Min = rMind*rMind;
   }
}

void InitGauss2DWeightsParams::ioParam_strength(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "strength", &strength, strength/*default value*/, true/*warnIfAbsent*/);
}

void InitGauss2DWeightsParams::ioParam_numOrientationsPost(enum ParamsIOFlag ioFlag) {
   if (post->getLayerLoc()->nf > 1) {
      parent->parameters()->ioParamValue(ioFlag, name, "numOrientationsPost", &numOrientationsPost, this->post->getLayerLoc()->nf);
   }
}

void InitGauss2DWeightsParams::ioParam_numOrientationsPre(enum ParamsIOFlag ioFlag) {
   if (pre->getLayerLoc()->nf > 1) {
      parent->parameters()->ioParamValue(ioFlag, name, "numOrientationsPre", &numOrientationsPre, this->pre->getLayerLoc()->nf);
   }
}

void InitGauss2DWeightsParams::ioParam_deltaThetaMax(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "deltaThetaMax", &deltaThetaMax, getDeltaThetaMax());
}

void InitGauss2DWeightsParams::ioParam_thetaMax(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "thetaMax", &thetaMax, getThetaMax());
}

void InitGauss2DWeightsParams::ioParam_numFlanks(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "numFlanks", &numFlanks, numFlanks);
}

void InitGauss2DWeightsParams::ioParam_flankShift(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "flankShift", &shift, shift);
}

void InitGauss2DWeightsParams::ioParam_rotate(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "rotate", &rotate, rotate);
}

void InitGauss2DWeightsParams::ioParam_bowtieFlag(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "bowtieFlag", &bowtieFlag, bowtieFlag);
}

void InitGauss2DWeightsParams::ioParam_bowtieAngle(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "bowtieFlag"));
   if (bowtieFlag) {
      parent->parameters()->ioParamValue(ioFlag, name, "bowtieAngle", &bowtieAngle, bowtieAngle);
   }
}

void InitGauss2DWeightsParams::ioParam_aspectRelatedParams(enum ParamsIOFlag ioFlag) {
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

bool InitGauss2DWeightsParams::needAspectParams() {
   assert(pre && post);
   assert(!parent->parameters()->presentAndNotBeenRead(name, "aspect"));
   if (post->getLayerLoc()->nf>1) {
      assert(!parent->parameters()->presentAndNotBeenRead(name, "numOrientationsPost"));
   }
   if (pre->getLayerLoc()->nf>1) {
      assert(!parent->parameters()->presentAndNotBeenRead(name, "numOrientationsPre"));
   }
   return (aspect != 1.0f && ((this->numOrientationsPre <= 1)||(this->numOrientationsPost <= 1)));
}

int InitGauss2DWeightsParams::communicateParamsInfo() {
   int status = InitWeightsParams::communicateParamsInfo();
   // Handle params that use pre and post to determine if they
   // need to be read
   ioParam_numOrientationsPost(PARAMS_IO_READ);
   ioParam_numOrientationsPre(PARAMS_IO_READ);
   ioParam_aspectRelatedParams(PARAMS_IO_READ);
   //calculate other values:
   self = (pre != post);
   return status;
}

void InitGauss2DWeightsParams::calcOtherParams(int patchIndex) {
   this->getcheckdimensionsandstrides();
   const int kfPre_tmp = this->kernelIndexCalculations(patchIndex);
   this->calculateThetas(kfPre_tmp, patchIndex);
}


bool InitGauss2DWeightsParams::isSameLocOrSelf(float xDelta, float yDelta, int fPost) {
   bool sameLoc = ((getFPre() == fPost) && (xDelta == 0.0f) && (yDelta == 0.0f));
   if ((sameLoc) && (!self)) {
      return true;
   }
   return false;
}

bool InitGauss2DWeightsParams::checkBowtieAngle(float xp, float yp) {
   if (bowtieFlag == 1){
      float offaxis_angle = atan2(yp, xp);
      if ( ((offaxis_angle > bowtieAngle) && (offaxis_angle < (PI - bowtieAngle))) ||
            ((offaxis_angle < -bowtieAngle) && (offaxis_angle > (-PI + bowtieAngle))) ){
         return true;
      }
   }
   return false;
}


void InitGauss2DWeightsParams::calculateThetas(int kfPre_tmp, int patchIndex) {
   dthPost = PI*thetaMax / (float) numOrientationsPost;
   th0Post = rotate * dthPost / 2.0f;
   const float dthPre = calcDthPre();
   const float th0Pre = calcTh0Pre(dthPre);
   fPre = patchIndex % pre->getLayerLoc()->nf;
   assert(fPre == kfPre_tmp);
   const int iThPre = patchIndex % numOrientationsPre;
   thPre = th0Pre + iThPre * dthPre;
}

float InitGauss2DWeightsParams::calcDthPre() {
   return PI*thetaMax / (float) numOrientationsPre;
}

float InitGauss2DWeightsParams::calcTh0Pre(float dthPre) {
   return rotate * dthPre / 2.0f;
}

float InitGauss2DWeightsParams::calcThPost(int fPost) {
   int oPost = fPost % numOrientationsPost;
   float thPost = th0Post + oPost * dthPost;
   if (numOrientationsPost == 1 && numOrientationsPre > 1) {
      thPost = thPre;
   }
   return thPost;
}

bool InitGauss2DWeightsParams::checkThetaDiff(float thPost) {
  if ((deltaTheta = fabsf(thPre - thPost)) > deltaThetaMax) {
     //the following is obviously not ideal. But cocirc needs this deltaTheta:
     deltaTheta = (deltaTheta <= PI / 2.0f) ? deltaTheta : PI - deltaTheta;
      return true;
   }
  deltaTheta = (deltaTheta <= PI / 2.0f) ? deltaTheta : PI - deltaTheta;
   return false;
}

bool InitGauss2DWeightsParams::checkColorDiff(int fPost) {
	int postColor = (int) (fPost / numOrientationsPost);
	int preColor = (int) (fPre / numOrientationsPre);
  if (postColor != preColor) {
      return true;
   }
   return false;
}



} /* namespace PV */
