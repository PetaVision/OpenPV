/*
 * InitSpreadOverArborsWeightsParams.cpp
 *
 *  Created on: Sep 1, 2011
 *      Author: kpeterson
 */

#include "InitSpreadOverArborsWeightsParams.hpp"

namespace PV {

InitSpreadOverArborsWeightsParams::InitSpreadOverArborsWeightsParams()
{
   initialize_base();
}
InitSpreadOverArborsWeightsParams::InitSpreadOverArborsWeightsParams(HyPerConn * parentConn)
                     : InitGauss2DWeightsParams() {
   initialize_base();
   initialize(parentConn);
}

InitSpreadOverArborsWeightsParams::~InitSpreadOverArborsWeightsParams()
{
}


int InitSpreadOverArborsWeightsParams::initialize_base() {

   initWeight = 1;
   setDeltaThetaMax(0.0f);
   setThetaMax(0.0f);
   setRotate(0.0f);
   return 1;
}

int InitSpreadOverArborsWeightsParams::initialize(HyPerConn * parentConn) {
   return InitWeightsParams::initialize(parentConn);
}

int InitSpreadOverArborsWeightsParams::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitGauss2DWeightsParams::ioParamsFillGroup(ioFlag);
   ioParam_weightInit(ioFlag);
   return status;
}

void InitSpreadOverArborsWeightsParams::ioParam_weightInit(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "weightInit", &initWeight, initWeight);
}

void InitSpreadOverArborsWeightsParams::calcOtherParams(int patchIndex) {

   this->getcheckdimensionsandstrides();

   const int kfPre_tmp = this->kernelIndexCalculations(patchIndex);



   this->calculateThetas(kfPre_tmp, patchIndex);

}

} /* namespace PV */
