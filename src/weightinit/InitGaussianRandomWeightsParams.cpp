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

InitGaussianRandomWeightsParams::InitGaussianRandomWeightsParams(HyPerConn * parentConn)
                     : InitWeightsParams() {
   initialize_base();
   initialize(parentConn);
}

InitGaussianRandomWeightsParams::~InitGaussianRandomWeightsParams()
{
   // TODO Auto-generated destructor stub
}

int InitGaussianRandomWeightsParams::initialize_base() {

   wGaussMean = 0;
   wGaussStdev = 1;
   return 1;
}

int InitGaussianRandomWeightsParams::initialize(HyPerConn * parentConn) {
   InitWeightsParams::initialize(parentConn);

   PVParams * params = parent->parameters();
   int status = PV_SUCCESS;

   wGaussMean = params->value(getName(), "wGaussMean", wGaussMean);
   wGaussStdev = params->value(getName(), "wGaussStdev", wGaussStdev);


   return status;

}

} /* namespace PV */
