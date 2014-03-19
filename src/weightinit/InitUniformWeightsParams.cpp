/*
 * InitUniformWeightsParams.cpp
 *
 *  Created on: Aug 23, 2011
 *      Author: kpeterson
 */

#include "InitUniformWeightsParams.hpp"

namespace PV {

InitUniformWeightsParams::InitUniformWeightsParams()
{
   initialize_base();
}

InitUniformWeightsParams::InitUniformWeightsParams(HyPerConn * parentConn)
                     : InitWeightsParams() {
   initialize_base();
   initialize(parentConn);
}

InitUniformWeightsParams::~InitUniformWeightsParams()
{
}

int InitUniformWeightsParams::initialize_base() {

   initWeight = 0;
   connectOnlySameFeatures = false;
   return 1;
}

int InitUniformWeightsParams::initialize(HyPerConn * parentConn) {
   return InitWeightsParams::initialize(parentConn);
}

int InitUniformWeightsParams::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitWeightsParams::ioParamsFillGroup(ioFlag);
   ioParam_weightInit(ioFlag);
   ioParam_connectOnlySameFeatures(ioFlag);
   return status;
}

void InitUniformWeightsParams::ioParam_weightInit(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "weightInit", &initWeight, initWeight);
}

void InitUniformWeightsParams::ioParam_connectOnlySameFeatures(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "connectOnlySameFeatures", &connectOnlySameFeatures, connectOnlySameFeatures);
}


} /* namespace PV */
