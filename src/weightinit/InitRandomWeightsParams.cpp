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

InitRandomWeightsParams::InitRandomWeightsParams(HyPerConn * pConn) {
   // initialize_base();
   initialize(pConn);
}

InitRandomWeightsParams::~InitRandomWeightsParams() {
}

} /* namespace PV */
