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
InitGauss2DWeightsParams::InitGauss2DWeightsParams(HyPerConn * parentConn)
                     : InitWeightsParams() {
   initialize_base();
   initialize(parentConn);
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

int InitGauss2DWeightsParams::initialize(HyPerConn * parentConn) {
   InitWeightsParams::initialize(parentConn);

   PVParams * params = parent->parameters();
   int status = PV_SUCCESS;

   aspect   = params->value(getName(), "aspect", aspect);
   sigma    = params->value(getName(), "sigma", sigma);
   rMax     = params->value(getName(), "rMax", rMax);
   rMin     = params->value(getName(), "rMin", rMin);
   strength = params->value(getName(), "strength", strength);
   if (this->post->getLayerLoc()->nf > 1){
		numOrientationsPost = (int) params->value(post->getName(),
				"numOrientationsPost", this->post->getLayerLoc()->nf);
   }
	if (this->pre->getLayerLoc()->nf > 1) {
		numOrientationsPre = (int) params->value(post->getName(),
				"numOrientationsPre", this->pre->getLayerLoc()->nf);
	}
   if (aspect != 1.0 && ((this->numOrientationsPre <= 1)||(this->numOrientationsPost <= 1))) {
      setDeltaThetaMax(params->value(getName(), "deltaThetaMax", getDeltaThetaMax()));
      setThetaMax(params->value(getName(), "thetaMax", getThetaMax()));
      numFlanks = (int) params->value(getName(), "numFlanks", (float) numFlanks);
      shift = params->value(getName(), "flankShift", shift);
      setRotate(params->value(getName(), "rotate", getRotate()));
      bowtieFlag = (bool)params->value(getName(), "bowtieFlag", bowtieFlag);
      if (bowtieFlag == 1.0f) {
         bowtieAngle = params->value(getName(), "bowtieAngle", bowtieAngle);
      }
   }

   double r2Maxd = (double) rMax;
   r2Max = r2Maxd*r2Maxd;
   double r2Mind = (double) rMin;
   r2Min = r2Mind*r2Mind;


//calculate other values:
   self = (pre != post);
   return status;

}

void InitGauss2DWeightsParams::calcOtherParams(int patchIndex) {
	InitWeightsParams::calcOtherParams(patchIndex);
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
   //numOrientationsPost = post->getLayerLoc()->nf;  // to allow for color bands, can't assume numOrientations
   dthPost = PI*thetaMax / (float) numOrientationsPost;
   th0Post = rotate * dthPost / 2.0f;
   //numOrientationsPre = pre->getLayerLoc()->nf; // to allow for color bands, can't assume numOrientations
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

bool InitGauss2DWeightsParams::checkTheta(float thPost) {
  if ((deltaTheta = fabs(thPre - thPost)) > deltaThetaMax) {
     //the following is obviously not ideal. But cocirc needs this deltaTheta:
     deltaTheta = (deltaTheta <= PI / 2.0) ? deltaTheta : PI - deltaTheta;
      return true;
   }
  deltaTheta = (deltaTheta <= PI / 2.0) ? deltaTheta : PI - deltaTheta;
   return false;
}



} /* namespace PV */
