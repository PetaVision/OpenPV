/*
 * InitOneToOneWeightsParams.cpp
 *
 *  Created on: Sep 28, 2011
 *      Author: kpeterson
 */

#include "InitOneToOneWeightsParams.hpp"

namespace PV {

InitOneToOneWeightsParams::InitOneToOneWeightsParams()
{
   initialize_base();
}
InitOneToOneWeightsParams::InitOneToOneWeightsParams(HyPerConn * parentConn)
                     : InitWeightsParams() {
   initialize_base();
   initialize(parentConn);
}

InitOneToOneWeightsParams::~InitOneToOneWeightsParams()
{
   // TODO Auto-generated destructor stub
}

int InitOneToOneWeightsParams::initialize_base() {

   initWeight = 1;
   return 1;
}

int InitOneToOneWeightsParams::initialize(HyPerConn * parentConn) {
   InitWeightsParams::initialize(parentConn);

   PVParams * params = parent->parameters();
   int status = PV_SUCCESS;

   initWeight = params->value(getName(), "weightInit", initWeight);


   return status;

}

void InitOneToOneWeightsParams::calcOtherParams(PVPatch * patch, int patchIndex) {

   this->getcheckdimensionsandstrides(patch);

   //const int kfPre_tmp = this->kernalIndexCalculations(patch, patchIndex);



   //this->calculateThetas(kfPre_tmp, patchIndex);

}


} /* namespace PV */
