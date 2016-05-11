/*
 * InitOneToOneWeightsParams.cpp
 *
 *  Created on: Sep 28, 2011
 *      Author: kpeterson
 */

#include "InitOneToOneWeightsParams.hpp"

namespace PV {

InitOneToOneWeightsParams::InitOneToOneWeightsParams()
{
   initialize_base();
}
InitOneToOneWeightsParams::InitOneToOneWeightsParams(const char * name, HyPerCol * hc)
                     : InitWeightsParams() {
   initialize_base();
   initialize(name, hc);
}

InitOneToOneWeightsParams::~InitOneToOneWeightsParams()
{
}

int InitOneToOneWeightsParams::initialize_base() {

   initWeight = 1;
   return 1;
}

int InitOneToOneWeightsParams::initialize(const char * name, HyPerCol * hc) {
   return InitWeightsParams::initialize(name, hc);
}

int InitOneToOneWeightsParams::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitWeightsParams::ioParamsFillGroup(ioFlag);
   ioParam_weightInit(ioFlag);
   return status;
}

void InitOneToOneWeightsParams::ioParam_weightInit(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, getName(), "weightInit", &initWeight, initWeight);
}

void InitOneToOneWeightsParams::calcOtherParams(int patchIndex) {
   this->getcheckdimensionsandstrides();
}


} /* namespace PV */
