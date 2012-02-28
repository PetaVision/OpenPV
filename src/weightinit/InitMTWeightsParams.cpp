/*
 * InitMTWeightsParams.cpp
 *
 *  Created on: Oct 25, 2011
 *      Author: kpeterson
 */

#include "InitMTWeightsParams.hpp"

namespace PV {

InitMTWeightsParams::InitMTWeightsParams()
{
   initialize_base();
}
InitMTWeightsParams::InitMTWeightsParams(HyPerConn * parentConn)
                     : InitWeightsParams() {
   initialize_base();
   initialize(parentConn);
}

InitMTWeightsParams::~InitMTWeightsParams()
{
   // TODO Auto-generated destructor stub
}

int InitMTWeightsParams::initialize_base() {

   // default values (chosen for center on cell of one pixel)
   setDeltaThetaMax(2.0f * PI);  // max orientation in units of PI
   setThetaMax(1.0); // max orientation in units of PI
   setRotate(0.0f);  // rotate so that axis isn't aligned

   tunedSpeed=0;
   inputV1Speed=0;
   inputV1Rotate=0;
   inputV1ThetaMax=1;


   return 1;
}
int InitMTWeightsParams::initialize(HyPerConn * parentConn) {
   InitWeightsParams::initialize(parentConn);

   PVParams * params = parent->parameters();
   int status = PV_SUCCESS;

   tunedSpeed = params->value(getName(), "tunedSpeed", tunedSpeed);
   inputV1Speed = params->value(getName(), "inputV1Speed", inputV1Speed);
   setRotate(params->value(getName(), "rotate", getRotate()));
   inputV1Rotate = params->value(getName(), "inputV1Rotate", inputV1Rotate);


   if (parentConn->fPatchSize() > 1) {
      setDeltaThetaMax(params->value(getName(), "deltaThetaMax", getDeltaThetaMax()));
      setThetaMax(params->value(getName(), "thetaMax", getThetaMax()));
      inputV1ThetaMax = params->value(getName(), "inputV1ThetaMax", inputV1ThetaMax);
   }


   return status;

}

float InitMTWeightsParams::calcDthPre() {
   return PI*inputV1ThetaMax / (float) noPre;
}
float InitMTWeightsParams::calcTh0Pre(float dthPre) {
   return inputV1Rotate * dthPre / 2.0f;
}

void InitMTWeightsParams::calcOtherParams(int patchIndex) {

   this->getcheckdimensionsandstrides();

   const int kfPre_tmp = this->kernelIndexCalculations(patchIndex);



   this->calculateThetas(kfPre_tmp, patchIndex);

}

} /* namespace PV */
