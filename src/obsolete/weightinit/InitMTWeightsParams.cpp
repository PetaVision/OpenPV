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
                     : InitGauss2DWeightsParams() {
   initialize_base();
   initialize(parentConn);
}

InitMTWeightsParams::~InitMTWeightsParams()
{
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
   return InitWeightsParams::initialize(parentConn);
}

int InitMTWeightsParams::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitGauss2DWeightsParams::ioParamsFillGroup(ioFlag);
   ioParam_tunedSpeed(ioFlag);
   ioParam_inputV1Speed(ioFlag);
   ioParam_inputV1Rotate(ioFlag);
   if (ioFlag != PARAMS_IO_READ) {
      ioParam_nfpRelatedParams(ioFlag);
   }
   return status;
}

int InitMTWeightsParams::communicateParamsInfo() {
   int status = InitGauss2DWeightsParams::communicateParamsInfo();
   // deltaThetaMax, thetaMax and inputV1ThetaMax are meaningful
   // only if the connection has nfp > 1
   ioParam_nfpRelatedParams(PARAMS_IO_READ);
   return status;
}

void InitMTWeightsParams::ioParam_tunedSpeed(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "tunedSpeed", &tunedSpeed, tunedSpeed);
}

void InitMTWeightsParams::ioParam_inputV1Speed(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "inputV1Speed", &inputV1Speed, inputV1Speed);
}

void InitMTWeightsParams::ioParam_inputV1Rotate(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "inputV1Rotate", &inputV1Rotate, inputV1Rotate);
}

void InitMTWeightsParams::ioParam_nfpRelatedParams(enum ParamsIOFlag ioFlag) {
   // Fix this: on output, deltaThetaMax and thetaMax may get printed twice
   // since the ioParam routines get called by ioParam_aspectRelatedParams()
   // and ioParam_nfpRelatedParams()
   if (parentConn->fPatchSize()>1) {
      ioParam_deltaThetaMax(ioFlag);
      ioParam_thetaMax(ioFlag);
      ioParam_inputV1ThetaMax(ioFlag);
   }
}

void InitMTWeightsParams::ioParam_inputV1ThetaMax(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "inputV1ThetaMax", &inputV1ThetaMax, inputV1ThetaMax);
}

float InitMTWeightsParams::calcDthPre() {
   return PI*inputV1ThetaMax / (float) numOrientationsPre;
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
