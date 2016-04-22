/*
 * InitRuleWeightsParams.cpp
 *
 *  Created on: Aug 14, 2011
 *      Author: kpeterson
 */

#include "InitRuleWeightsParams.hpp"

namespace PV {

InitRuleWeightsParams::InitRuleWeightsParams()
{
   initialize_base();
}
InitRuleWeightsParams::InitRuleWeightsParams(HyPerConn * parentConn)
                     : InitGauss2DWeightsParams() {
   initialize_base();
   initialize(parentConn);
}
InitRuleWeightsParams::~InitRuleWeightsParams()
{
}

int InitRuleWeightsParams::initialize_base() {

   strength = 1.0;
   return 1;
}

int InitRuleWeightsParams::initialize(HyPerConn * parentConn) {
   return InitGauss2DWeightsParams::initialize(parentConn);
}

void InitRuleWeightsParams::calcOtherParams(int patchIndex) {

   this->getcheckdimensionsandstrides();

   const int kfPre_tmp = this->kernelIndexCalculations(patchIndex);



   this->calculateThetas(kfPre_tmp, patchIndex);

}

} /* namespace PV */
