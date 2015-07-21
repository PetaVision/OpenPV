/*
 * InitPoolWeightsParams.cpp
 *
 *  Created on: Aug 13, 2011
 *      Author: kpeterson
 */

#include "InitPoolWeightsParams.hpp"
#include "InitGauss2DWeightsParams.hpp"

namespace PV {

InitPoolWeightsParams::InitPoolWeightsParams()
{
   initialize_base();
}
InitPoolWeightsParams::InitPoolWeightsParams(HyPerConn * parentConn)
                     : InitGauss2DWeightsParams() {
   initialize_base();
   initialize(parentConn);
}

InitPoolWeightsParams::~InitPoolWeightsParams()
{
}

int InitPoolWeightsParams::initialize_base() {

   strength = 1.0;
   return 1;
}

int InitPoolWeightsParams::initialize(HyPerConn * parentConn) {
   return InitGauss2DWeightsParams::initialize(parentConn);
}

void InitPoolWeightsParams::calcOtherParams(int patchIndex) {

   this->getcheckdimensionsandstrides();

   const int kfPre_tmp = this->kernelIndexCalculations(patchIndex);



   this->calculateThetas(kfPre_tmp, patchIndex);

}


} /* namespace PV */
