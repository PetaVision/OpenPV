/*
 * InitTransposeWeightsParams.cpp
 *
 *  Created on: Aug 15, 2011
 *      Author: kpeterson
 */

#include "InitTransposeWeightsParams.hpp"

namespace PV {

InitTransposeWeightsParams::InitTransposeWeightsParams()
{
   initialize_base();
}
InitTransposeWeightsParams::InitTransposeWeightsParams(HyPerConn * parentConn)
                     : InitWeightsParams() {
   initialize_base();
   initialize(parentConn);
}

InitTransposeWeightsParams::~InitTransposeWeightsParams()
{
   // TODO Auto-generated destructor stub
}

int InitTransposeWeightsParams::initialize_base() {
   return 1;
}

int InitTransposeWeightsParams::initialize(HyPerConn * parentConn) {
   InitWeightsParams::initialize(parentConn);

   int status = PV_SUCCESS;

   return status;

}

void InitTransposeWeightsParams::calcOtherParams(PVPatch * patch, int patchIndex) {

   this->getcheckdimensionsandstrides(patch);

   //const int kfPre_tmp = this->kernalIndexCalculations(patch, patchIndex);



   //this->calculateThetas(kfPre_tmp, patchIndex);

}

} /* namespace PV */
