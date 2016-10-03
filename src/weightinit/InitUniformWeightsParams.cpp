/*
 * InitUniformWeightsParams.cpp
 *
 *  Created on: Aug 23, 2011
 *      Author: kpeterson
 */

#include "InitUniformWeightsParams.hpp"

namespace PV {

InitUniformWeightsParams::InitUniformWeightsParams() { initialize_base(); }

InitUniformWeightsParams::InitUniformWeightsParams(const char *name, HyPerCol *hc)
      : InitWeightsParams() {
   initialize_base();
   initialize(name, hc);
}

InitUniformWeightsParams::~InitUniformWeightsParams() {}

int InitUniformWeightsParams::initialize_base() {

   initWeight              = 0;
   connectOnlySameFeatures = false;
   return 1;
}

int InitUniformWeightsParams::initialize(const char *name, HyPerCol *hc) {
   return InitWeightsParams::initialize(name, hc);
}

int InitUniformWeightsParams::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitWeightsParams::ioParamsFillGroup(ioFlag);
   ioParam_weightInit(ioFlag);
   ioParam_connectOnlySameFeatures(ioFlag);
   return status;
}

void InitUniformWeightsParams::ioParam_weightInit(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "weightInit", &initWeight, initWeight);
}

void InitUniformWeightsParams::ioParam_connectOnlySameFeatures(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag,
         name,
         "connectOnlySameFeatures",
         &connectOnlySameFeatures,
         connectOnlySameFeatures);
}

} /* namespace PV */
