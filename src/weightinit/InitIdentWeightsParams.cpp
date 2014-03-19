/*
 * InitIdentWeightsParams.cpp
 *
 *  Created on: Aug 14, 2011
 *      Author: kpeterson
 */

#include "InitIdentWeightsParams.hpp"

namespace PV {

InitIdentWeightsParams::InitIdentWeightsParams()
{
   initialize_base();
}
InitIdentWeightsParams::InitIdentWeightsParams(HyPerConn * parentConn)
                     : InitWeightsParams() {
   initialize_base();
   initialize(parentConn);
}

InitIdentWeightsParams::~InitIdentWeightsParams()
{
}

int InitIdentWeightsParams::initialize_base() {
   return PV_SUCCESS;
}

int InitIdentWeightsParams::initialize(HyPerConn * parentConn) {
   return InitWeightsParams::initialize(parentConn);
}

void InitIdentWeightsParams::calcOtherParams(int patchIndex) {

   this->getcheckdimensionsandstrides();

}

} /* namespace PV */
