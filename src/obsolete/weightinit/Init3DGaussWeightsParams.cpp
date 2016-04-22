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
                     : InitGauss2DWeightsParams() {
   initialize_base();
   initialize(parentConn);
}

Init3DGaussWeightsParams::~Init3DGaussWeightsParams()
{
}

int Init3DGaussWeightsParams::initialize_base() {

   // default values (chosen for center on cell of one pixel)
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
   return InitGauss2DWeightsParams::initialize(parentConn);
}

int Init3DGaussWeightsParams::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitGauss2DWeightsParams::ioParamsFillGroup(ioFlag);
   ioParam_yaspect(ioFlag);
   ioParam_taspect(ioFlag);
   ioParam_dT(ioFlag);
   ioParam_shiftT(ioFlag);
   ioParam_flowSpeed(ioFlag);
   return status;
}

void Init3DGaussWeightsParams::ioParam_yaspect(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "yaspect", &yaspect, yaspect);
}

void Init3DGaussWeightsParams::ioParam_taspect(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "taspect", &taspect, taspect);
}

void Init3DGaussWeightsParams::ioParam_dT(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "dT", &dT, dT);
}

void Init3DGaussWeightsParams::ioParam_shiftT(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "shiftT", &shiftT, shiftT);
}

void Init3DGaussWeightsParams::ioParam_flowSpeed(enum ParamsIOFlag ioFlag) {
   float flowSpeed = 1.0f;
   parent->ioParamValue(ioFlag, name, "flowSpeed", &flowSpeed, flowSpeed);
   if (ioFlag == PARAMS_IO_READ) {
      thetaXT = atanf(flowSpeed);
   }
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
