/*
 * InitRandomWeightsParams.cpp
 *
 *  Created on: Feb 26, 2014
 *      Author: pschultz
 */

#include "InitRandomWeightsParams.hpp"

namespace PV {


InitRandomWeightsParams::InitRandomWeightsParams()
{
   // initialize_base();
}

InitRandomWeightsParams::InitRandomWeightsParams(const char * name, HyPerCol * hc) {
   // initialize_base();
   initialize(name, hc);
}

InitRandomWeightsParams::~InitRandomWeightsParams() {
}

} /* namespace PV */
