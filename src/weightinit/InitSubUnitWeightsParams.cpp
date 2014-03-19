/*
 * InitSubUnitWeightsParams.cpp
 *
 *  Created on: Aug 14, 2011
 *      Author: kpeterson
 */

#include "InitSubUnitWeightsParams.hpp"

namespace PV {

InitSubUnitWeightsParams::InitSubUnitWeightsParams()
{
   initialize_base();
}
InitSubUnitWeightsParams::InitSubUnitWeightsParams(HyPerConn * parentConn)
                     : InitWeightsParams() {
   initialize_base();
   initialize(parentConn);
}

InitSubUnitWeightsParams::~InitSubUnitWeightsParams()
{
}

int InitSubUnitWeightsParams::initialize_base() {
   return PV_SUCCESS;
}

int InitSubUnitWeightsParams::initialize(HyPerConn * parentConn) {
   InitWeightsParams::initialize(parentConn);

   int status = PV_SUCCESS;

   return status;

}

void InitSubUnitWeightsParams::calcOtherParams(int patchIndex) {

   this->getcheckdimensionsandstrides();

   //const int kfPre_tmp = this->kernalIndexCalculations(patch, patchIndex);



   //this->calculateThetas(kfPre_tmp, patchIndex);

}

} /* namespace PV */
