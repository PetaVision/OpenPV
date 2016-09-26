/*
 * InitGaborWeightsParams.cpp
 *
 *  Created on: Aug 13, 2011
 *      Author: kpeterson
 */

#include "InitGaborWeightsParams.hpp"

namespace PV {

InitGaborWeightsParams::InitGaborWeightsParams()
{
   initialize_base();
}
InitGaborWeightsParams::InitGaborWeightsParams(const char * name, HyPerCol * hc)
                     : InitGauss2DWeightsParams() {
   initialize_base();
   initialize(name, hc);
}

InitGaborWeightsParams::~InitGaborWeightsParams()
{
}

int InitGaborWeightsParams::initialize_base() {

   aspect = 4.0f; // circular (not line oriented)
   sigma = 2.0f;
   rMax = 8.0f;
   strength = 1.0f;
   r2Max = rMax * rMax;

   //numFlanks = 1;
   shift = 0.0f;
   //rotate = 1.0f; // rotate so that axis isn't aligned
   setRotate(0.0f); // rotate so that axis isn't aligned

   lambda = (int)(sigma/0.8f);
   phi=0;
   invert=true;

   setThetaMax(1.0f); // max orientation in units of PI

   return 1;
}

int InitGaborWeightsParams::initialize(const char * name, HyPerCol * hc) {
   return InitGauss2DWeightsParams::initialize(name, hc);
}

int InitGaborWeightsParams::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitGauss2DWeightsParams::ioParamsFillGroup(ioFlag);
   ioParam_lambda(ioFlag);
   ioParam_phi(ioFlag);
   ioParam_invert(ioFlag);
   return status;
}

void InitGaborWeightsParams::ioParam_lambda(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "lambda", &lambda, lambda);
}

void InitGaborWeightsParams::ioParam_phi(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "phi", &phi, phi);
}

void InitGaborWeightsParams::ioParam_invert(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "invert", &invert, invert);
}

void InitGaborWeightsParams::calcOtherParams(int patchIndex) {

   this->getcheckdimensionsandstrides();

   const int kfPre_tmp = this->kernelIndexCalculations(patchIndex);



   this->calculateThetas(kfPre_tmp, patchIndex);

}


} /* namespace PV */
