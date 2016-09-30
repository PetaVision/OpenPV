/*
 * InitMaxPoolingWeightsParams.cpp
 *
 *  Created on: Jan 14, 2015
 *      Author: gkenyon
 */

#include "InitMaxPoolingWeightsParams.hpp"

namespace PV {

InitMaxPoolingWeightsParams::InitMaxPoolingWeightsParams() { initialize_base(); }

InitMaxPoolingWeightsParams::InitMaxPoolingWeightsParams(const char *name, HyPerCol *hc)
      : InitWeightsParams() {
   initialize_base();
   initialize(name, hc);
}

InitMaxPoolingWeightsParams::~InitMaxPoolingWeightsParams() {}

int InitMaxPoolingWeightsParams::initialize_base() { return PV_SUCCESS; }

int InitMaxPoolingWeightsParams::initialize(const char *name, HyPerCol *hc) {
   return InitWeightsParams::initialize(name, hc);
}

int InitMaxPoolingWeightsParams::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitWeightsParams::ioParamsFillGroup(ioFlag);
   return status;
}

} /* namespace PV */
