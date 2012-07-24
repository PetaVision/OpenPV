/*
 * InitGauss2DWeightsParams.cpp
 *
 *  Created on: Aug 10, 2011
 *      Author: kpeterson
 */

#include "InitBIDSWeightsParams.hpp"

namespace PV {

InitBIDSWeightsParams::InitBIDSWeightsParams()
{
   initialize_base();
}
InitBIDSWeightsParams::InitBIDSWeightsParams(HyPerConn * parentConn)
                     : InitWeightsParams() {
   initialize_base();
   initialize(parentConn);
}

InitBIDSWeightsParams::~InitBIDSWeightsParams()
{
   // TODO Auto-generated destructor stub
}

int InitBIDSWeightsParams::initialize_base() {

   // default values (chosen for center on cell of one pixel)
   //int noPost = parentConn->fPatchSize();
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


   return 1;
}

int InitBIDSWeightsParams::initialize(HyPerConn * parentConn) {
   InitWeightsParams::initialize(parentConn);

   PVParams * params = parent->parameters();
   int status = PV_SUCCESS;

   aspect   = params->value(getName(), "aspect", aspect);
   sigma    = params->value(getName(), "sigma", sigma);
   rMax     = params->value(getName(), "rMax", rMax);
   rMin     = params->value(getName(), "rMin", rMin);
   strength = params->value(getName(), "strength", strength);
   // old if condition failed to account for connections between oriented to non-oriented cells
//   if (parentConn->fPatchSize() > 1) {
   if (aspect != 1.0) {      //noPost = (int) params->value(post->getName(), "no", parentConn->fPatchSize());
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

   BIDSLayer * post = dynamic_cast<BIDSLayer *>(parentConn->postSynapticLayer());
   coords = post->getCoords();
   numNodes = post->numNodes;

   double r2Maxd = (double) rMax;
   r2Max = r2Maxd*r2Maxd;
   double r2Mind = (double) rMin;
   r2Min = r2Mind*r2Mind;


//calculate other values:
   self = (pre != post);
   return status;

}

void InitBIDSWeightsParams::calcOtherParams(int patchIndex) {

   this->getcheckdimensionsandstrides();

   const int kfPre_tmp = this->kernelIndexCalculations(patchIndex);

   this->calculateThetas(kfPre_tmp, patchIndex);
}

bool InitBIDSWeightsParams::isSameLocOrSelf(float xDelta, float yDelta, int fPost) {
   bool sameLoc = ((getFPre() == fPost) && (xDelta == 0.0f) && (yDelta == 0.0f));
   if ((sameLoc) && (!self)) {
      return true;
   }
   return false;
}

bool InitBIDSWeightsParams::checkBowtieAngle(float xp, float yp) {
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
