/*
 * InitGaussianRandomWeightsParams.cpp
 *
 *  Created on: Aug 13, 2011
 *      Author: kpeterson
 */

#include "InitGaussianRandomWeightsParams.hpp"

namespace PV {

InitGaussianRandomWeightsParams::InitGaussianRandomWeightsParams()
{
   initialize_base();
}

InitGaussianRandomWeightsParams::InitGaussianRandomWeightsParams(const char * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

InitGaussianRandomWeightsParams::~InitGaussianRandomWeightsParams()
{
}

int InitGaussianRandomWeightsParams::initialize_base() {

   wGaussMean = 0;
   wGaussStdev = 1;
   return PV_SUCCESS;
}

int InitGaussianRandomWeightsParams::initialize(const char * name, HyPerCol * hc) {
   return InitRandomWeightsParams::initialize(name, hc);
}

int InitGaussianRandomWeightsParams::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitRandomWeightsParams::ioParamsFillGroup(ioFlag);
   ioParam_wGaussMean(ioFlag);
   ioParam_wGaussStdev(ioFlag);
   return status;
}

void InitGaussianRandomWeightsParams::ioParam_wGaussMean(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "wGaussMean", &wGaussMean, wGaussMean);
}

void InitGaussianRandomWeightsParams::ioParam_wGaussStdev(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "wGaussStdev", &wGaussStdev, wGaussStdev);
}

} /* namespace PV */
