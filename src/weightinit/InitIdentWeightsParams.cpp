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
InitIdentWeightsParams::InitIdentWeightsParams(const char * name, HyPerCol * hc)
                     : InitWeightsParams() {
   initialize_base();
   initialize(name, hc);
}

InitIdentWeightsParams::~InitIdentWeightsParams()
{
}

int InitIdentWeightsParams::initialize_base() {
   return PV_SUCCESS;
}

int InitIdentWeightsParams::initialize(const char * name, HyPerCol * hc) {
   return InitWeightsParams::initialize(name, hc);
}

void InitIdentWeightsParams::calcOtherParams(int patchIndex) {

   this->getcheckdimensionsandstrides();

}

} /* namespace PV */
