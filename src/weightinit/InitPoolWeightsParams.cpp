/*
 * InitPoolWeightsParams.cpp
 *
 *  Created on: Aug 13, 2011
 *      Author: kpeterson
 */

#include "InitPoolWeightsParams.hpp"

namespace PV {

InitPoolWeightsParams::InitPoolWeightsParams()
{
   initialize_base();
}
InitPoolWeightsParams::InitPoolWeightsParams(HyPerConn * parentConn)
                     : InitWeightsParams() {
   initialize_base();
   initialize(parentConn);
}

InitPoolWeightsParams::~InitPoolWeightsParams()
{
   // TODO Auto-generated destructor stub
}

int InitPoolWeightsParams::initialize_base() {

   strength = 1.0;
   return 1;
}

int InitPoolWeightsParams::initialize(HyPerConn * parentConn) {
   InitWeightsParams::initialize(parentConn);

   PVParams * params = parent->parameters();
   int status = PV_SUCCESS;

   strength = params->value(name, "strength", strength);
   return status;

}

void InitPoolWeightsParams::calcOtherParams(int patchIndex) {

   this->getcheckdimensionsandstrides();

   const int kfPre_tmp = this->kernelIndexCalculations(patchIndex);



   this->calculateThetas(kfPre_tmp, patchIndex);

}


} /* namespace PV */
