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
   // TODO Auto-generated destructor stub
}

int InitIdentWeightsParams::initialize_base() {
   return 1;
}

int InitIdentWeightsParams::initialize(HyPerConn * parentConn) {
   InitWeightsParams::initialize(parentConn);

   int status = PV_SUCCESS;

   return status;

}

void InitIdentWeightsParams::calcOtherParams(int patchIndex) {

   this->getcheckdimensionsandstrides();

   //const int kfPre_tmp = this->kernalIndexCalculations(patch, patchIndex);



   //this->calculateThetas(kfPre_tmp, patchIndex);

}

} /* namespace PV */
