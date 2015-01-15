/*
 * InitMaxPoolingWeightsParams.cpp
 *
 *  Created on: Jan 14, 2015
 *      Author: gkenyon
 */

#include "InitMaxPoolingWeightsParams.hpp"

namespace PV {

InitMaxPoolingWeightsParams::InitMaxPoolingWeightsParams()
{
   initialize_base();
}

InitMaxPoolingWeightsParams::InitMaxPoolingWeightsParams(HyPerConn * parentConn)
                     : InitWeightsParams() {
   initialize_base();
   initialize(parentConn);
}

InitMaxPoolingWeightsParams::~InitMaxPoolingWeightsParams()
{
}

int InitMaxPoolingWeightsParams::initialize_base() {

   return PV_SUCCESS;
}

int InitMaxPoolingWeightsParams::initialize(HyPerConn * parentConn) {
   return InitWeightsParams::initialize(parentConn);
}

int InitMaxPoolingWeightsParams::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitWeightsParams::ioParamsFillGroup(ioFlag);
   return status;
}

} /* namespace PV */
