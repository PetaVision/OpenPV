/*
 * InitUnivormRandomWeightsParams.cpp
 *
 *  Created on: Aug 12, 2011
 *      Author: kpeterson
 */

#include "InitUniformRandomWeightsParams.hpp"

namespace PV {

InitUniformRandomWeightsParams::InitUniformRandomWeightsParams() { initialize_base(); }

InitUniformRandomWeightsParams::InitUniformRandomWeightsParams(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

InitUniformRandomWeightsParams::~InitUniformRandomWeightsParams() {}

int InitUniformRandomWeightsParams::initialize_base() {
   mWMin           = 0;
   mWMax           = 1;
   mSparseFraction = 0;
   mMinNNZ         = 0;
   return 1;
}

int InitUniformRandomWeightsParams::initialize(const char *name, HyPerCol *hc) {
   return InitWeightsParams::initialize(name, hc);
}

int InitUniformRandomWeightsParams::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitRandomWeightsParams::ioParamsFillGroup(ioFlag);
   ioParam_wMinInit(ioFlag);
   ioParam_wMaxInit(ioFlag);
   ioParam_sparseFraction(ioFlag);
   ioParam_minNNZ(ioFlag);
   return status;
}

void InitUniformRandomWeightsParams::ioParam_wMinInit(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "wMinInit", &mWMin, mWMin);
}

void InitUniformRandomWeightsParams::ioParam_wMaxInit(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "wMaxInit", &mWMax, mWMax);
}

void InitUniformRandomWeightsParams::ioParam_sparseFraction(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "sparseFraction", &mSparseFraction, mSparseFraction);
}

void InitUniformRandomWeightsParams::ioParam_minNNZ(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "minNNZ", &mMinNNZ, mMinNNZ);
}

} /* namespace PV */
