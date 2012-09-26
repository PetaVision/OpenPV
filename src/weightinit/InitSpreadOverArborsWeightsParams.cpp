/*
 * InitSpreadOverArborsWeightsParams.cpp
 *
 *  Created on: Sep 1, 2011
 *      Author: kpeterson
 */

#include "InitSpreadOverArborsWeightsParams.hpp"

namespace PV {

InitSpreadOverArborsWeightsParams::InitSpreadOverArborsWeightsParams()
{
   initialize_base();
}
InitSpreadOverArborsWeightsParams::InitSpreadOverArborsWeightsParams(HyPerConn * parentConn)
                     : InitWeightsParams() {
   initialize_base();
   initialize(parentConn);
}

InitSpreadOverArborsWeightsParams::~InitSpreadOverArborsWeightsParams()
{
   // TODO Auto-generated destructor stub
}


int InitSpreadOverArborsWeightsParams::initialize_base() {

   initWeight = 1;
   numArbors = 1;
   setDeltaThetaMax(0.0f);
   setThetaMax(0.0f);
   setRotate(0.0f);
   return 1;
}

int InitSpreadOverArborsWeightsParams::initialize(HyPerConn * parentConn) {
   InitWeightsParams::initialize(parentConn);

   PVParams * params = parent->parameters();
   int status = PV_SUCCESS;

   numArbors = parentConn->numberOfAxonalArborLists();
   initWeight = params->value(getName(), "weightInit", initWeight);
   // initWeight /= numArbors;
   setDeltaThetaMax(params->value(getName(), "deltaThetaMax", getDeltaThetaMax()));
   setThetaMax(params->value(getName(), "thetaMax", getThetaMax()));
   setRotate(params->value(getName(), "rotate", getRotate()));

   return status;

}

void InitSpreadOverArborsWeightsParams::calcOtherParams(int patchIndex) {

   this->getcheckdimensionsandstrides();

   const int kfPre_tmp = this->kernelIndexCalculations(patchIndex);



   this->calculateThetas(kfPre_tmp, patchIndex);

}

} /* namespace PV */
