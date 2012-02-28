/*
 * Init3DGaussWeightsParams.cpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#include "Init3DGaussWeightsParams.hpp"

namespace PV {

Init3DGaussWeightsParams::Init3DGaussWeightsParams()
{
   initialize_base();
}
Init3DGaussWeightsParams::Init3DGaussWeightsParams(HyPerConn * parentConn)
                     : InitWeightsParams() {
   initialize_base();
   initialize(parentConn);
}

Init3DGaussWeightsParams::~Init3DGaussWeightsParams()
{
   // TODO Auto-generated destructor stub
}

int Init3DGaussWeightsParams::initialize_base() {

   // default values (chosen for center on cell of one pixel)
   //int noPost = parentConn->fPatchSize();
   yaspect = 1.0; // circular (not line oriented)
   taspect = 10.0; // circular (not line oriented)
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
   thetaXT = PI/4;
   dT = 1;
   time=0;
   shiftT = -10;


   return 1;
}
int Init3DGaussWeightsParams::initialize(HyPerConn * parentConn) {
   InitWeightsParams::initialize(parentConn);

   PVParams * params = parent->parameters();
   int status = PV_SUCCESS;

   yaspect   = params->value(getName(), "yaspect", yaspect);
   taspect   = params->value(getName(), "taspect", taspect);
   sigma    = params->value(getName(), "sigma", sigma);
   rMax     = params->value(getName(), "rMax", rMax);
   strength = params->value(getName(), "strength", strength);
   dT = params->value(getName(), "dT", dT);
   shiftT = params->value(getName(), "shiftT", shiftT);
   float flowSpeed = params->value(getName(), "flowSpeed", 1.0f); //pixels per time step

   thetaXT = atanf(flowSpeed);
   setRotate(params->value(getName(), "rotate", getRotate()));

   numFlanks = (int) params->value(getName(), "numFlanks", (float) numFlanks);
   shift = params->value(getName(), "flankShift", shift);

   if (parentConn->fPatchSize() > 1) {
      //noPost = (int) params->value(post->getName(), "no", parentConn->fPatchSize());
      setDeltaThetaMax(params->value(getName(), "deltaThetaMax", getDeltaThetaMax()));
      setThetaMax(params->value(getName(), "thetaMax", getThetaMax()));

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

void Init3DGaussWeightsParams::calcOtherParams(int patchIndex) {

   this->getcheckdimensionsandstrides();

   const int kfPre_tmp = this->kernelIndexCalculations(patchIndex);



   this->calculateThetas(kfPre_tmp, patchIndex);

}


bool Init3DGaussWeightsParams::isSameLocOrSelf(float xDelta, float yDelta, int fPost) {
   bool sameLoc = ((getFPre() == fPost) && (xDelta == 0.0f) && (yDelta == 0.0f));
   if ((sameLoc) && (!self)) {
      return true;
   }
   return false;
}

bool Init3DGaussWeightsParams::checkBowtieAngle(float xp, float yp) {
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
