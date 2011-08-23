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
                     : InitWeightsParams() {
   initialize_base();
   initialize(parentConn);
}
InitRuleWeightsParams::~InitRuleWeightsParams()
{
   // TODO Auto-generated destructor stub
}

int InitRuleWeightsParams::initialize_base() {

   strength = 1.0;
   return 1;
}

int InitRuleWeightsParams::initialize(HyPerConn * parentConn) {
   InitWeightsParams::initialize(parentConn);

   PVParams * params = parent->parameters();
   int status = PV_SUCCESS;

   strength = params->value(name, "strength", strength);
   return status;

}

void InitRuleWeightsParams::calcOtherParams(PVPatch * patch, int patchIndex) {

   this->getcheckdimensionsandstrides(patch);

   const int kfPre_tmp = this->kernalIndexCalculations(patch, patchIndex);



   this->calculateThetas(kfPre_tmp, patchIndex);

}

} /* namespace PV */
