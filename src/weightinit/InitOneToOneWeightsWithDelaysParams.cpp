/*
 * InitOneToOneWeightsWithDelaysParams.cpp
 *
 *  Created on: Sep 20, 2013
 *      Author: wchavez
 */

#include "InitOneToOneWeightsWithDelaysParams.hpp"

namespace PV {

InitOneToOneWeightsWithDelaysParams::InitOneToOneWeightsWithDelaysParams()
{
   initialize_base();
}
InitOneToOneWeightsWithDelaysParams::InitOneToOneWeightsWithDelaysParams(HyPerConn * parentConn)
                     : InitWeightsParams() {
   initialize_base();
   initialize(parentConn);
}

InitOneToOneWeightsWithDelaysParams::~InitOneToOneWeightsWithDelaysParams()
{
   // TODO Auto-generated destructor stub
}

int InitOneToOneWeightsWithDelaysParams::initialize_base() {

   initWeight = 1;
   numArbors = 1;
   return 1;
}

int InitOneToOneWeightsWithDelaysParams::initialize(HyPerConn * parentConn) {
   InitWeightsParams::initialize(parentConn);

   PVParams * params = parent->parameters();
   int status = PV_SUCCESS;

   initWeight = params->value(getName(), "weightInit", initWeight);
   numArbors = parentConn->numberOfAxonalArborLists();

   return status;

}

void InitOneToOneWeightsWithDelaysParams::calcOtherParams(int patchIndex) {

   this->getcheckdimensionsandstrides();

   //const int kfPre_tmp = this->kernalIndexCalculations(patch, patchIndex);



   //this->calculateThetas(kfPre_tmp, patchIndex);

}


} /* namespace PV */
