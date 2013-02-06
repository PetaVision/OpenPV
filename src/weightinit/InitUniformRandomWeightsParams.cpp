/*
 * InitUnivormRandomWeightsParams.cpp
 *
 *  Created on: Aug 12, 2011
 *      Author: kpeterson
 */

#include "InitUniformRandomWeightsParams.hpp"

namespace PV {

InitUniformRandomWeightsParams::InitUniformRandomWeightsParams()
{
   initialize_base();
}

InitUniformRandomWeightsParams::InitUniformRandomWeightsParams(HyPerConn * parentConn)
                     : InitWeightsParams() {
   initialize_base();
   initialize(parentConn);
}

InitUniformRandomWeightsParams::~InitUniformRandomWeightsParams()
{
   // TODO Auto-generated destructor stub
}

int InitUniformRandomWeightsParams::initialize_base() {

   wMin = 0;
   wMax = 1;
   sparseFraction = 0;
   return 1;
}

int InitUniformRandomWeightsParams::initialize(HyPerConn * parentConn) {
   InitWeightsParams::initialize(parentConn);

   PVParams * params = parent->parameters();
   int status = PV_SUCCESS;

   // minWeight and maxWeight require arborId, use default values instead if wMinInit, wMaxInit not provided by user
   //wMin = (float) parentConn->minWeight();
   //wMax = (float) parentConn->maxWeight();
   wMin = params->value(getName(), "wMinInit", wMin);
   wMax = params->value(getName(), "wMaxInit", wMax);
   sparseFraction = params->value(getName(), "sparseFraction", sparseFraction);


   return status;

}

} /* namespace PV */
