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
   // TODO Auto-generated destructor stub
}

int InitGauss2DWeightsParams::initialize_base() {

   // default values (chosen for center on cell of one pixel)
   //int noPost = parentConn->fPatchSize();
   aspect = 1.0; // circular (not line oriented)
   sigma = 0.8;
   rMax = 1.4;
   strength = 1.0;
   setDeltaThetaMax(2.0f * PI);  // max orientation in units of PI
   setThetaMax(1.0); // max orientation in units of PI
   numFlanks = 1;
   shift = 0.0f;
   setRotate(0.0f);  // rotate so that axis isn't aligned
   bowtieFlag = 0.0f;  // flag for setting bowtie angle
   bowtieAngle = PI * 2.0f;  // bowtie angle


   return 1;
}
int InitGauss2DWeightsParams::initialize(HyPerConn * parentConn) {
   InitWeightsParams::initialize(parentConn);

   PVParams * params = parent->parameters();
   int status = PV_SUCCESS;

   aspect   = params->value(getName(), "aspect", aspect);
   sigma    = params->value(getName(), "sigma", sigma);
   rMax     = params->value(getName(), "rMax", rMax);
   strength = params->value(getName(), "strength", strength);
   if (parentConn->fPatchSize() > 1) {
      //noPost = (int) params->value(post->getName(), "no", parentConn->fPatchSize());
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


//calculate other values:
   self = (pre != post);


   return status;

}

void InitGauss2DWeightsParams::calcOtherParams(PVPatch * patch, int patchIndex) {

   this->getcheckdimensionsandstrides(patch);

   const int kfPre_tmp = this->kernelIndexCalculations(patch, patchIndex);



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

} /* namespace PV */
