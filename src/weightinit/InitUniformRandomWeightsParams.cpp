/*
 * InitUnivormRandomWeightsParams.cpp
 *
 *  Created on: Aug 12, 2011
 *      Author: kpeterson
 */

#include "InitUniformRandomWeightsParams.hpp"

namespace PV {

InitUniformRandomWeightsParams::InitUniformRandomWeightsParams()
{
   initialize_base();
}

InitUniformRandomWeightsParams::InitUniformRandomWeightsParams(HyPerConn * parentConn)
{
   initialize_base();
   initialize(parentConn);
}

InitUniformRandomWeightsParams::~InitUniformRandomWeightsParams()
{
}

int InitUniformRandomWeightsParams::initialize_base() {

   wMin = 0;
   wMax = 1;
   sparseFraction = 0;
   return 1;
}

int InitUniformRandomWeightsParams::initialize(HyPerConn * parentConn) {
   return InitWeightsParams::initialize(parentConn);
}

int InitUniformRandomWeightsParams::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitRandomWeightsParams::ioParamsFillGroup(ioFlag);
   ioParam_wMinInit(ioFlag);
   ioParam_wMaxInit(ioFlag);
   ioParam_sparseFraction(ioFlag);
   return status;
}

void InitUniformRandomWeightsParams::ioParam_wMinInit(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "wMinInit", &wMin, wMin);
}

void InitUniformRandomWeightsParams::ioParam_wMaxInit(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "wMaxInit", &wMax, wMax);
}

void InitUniformRandomWeightsParams::ioParam_sparseFraction(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "sparseFraction", &sparseFraction, sparseFraction);
}


} /* namespace PV */
