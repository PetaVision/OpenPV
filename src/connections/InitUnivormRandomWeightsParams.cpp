/*
 * InitUnivormRandomWeightsParams.cpp
 *
 *  Created on: Aug 12, 2011
 *      Author: kpeterson
 */

#include "InitUnivormRandomWeightsParams.hpp"

namespace PV {

InitUnivormRandomWeightsParams::InitUnivormRandomWeightsParams()
{
   initialize_base();
}

InitUnivormRandomWeightsParams::InitUnivormRandomWeightsParams(HyPerConn * parentConn)
                     : InitWeightsParams() {
   initialize_base();
   initialize(parentConn);
}

InitUnivormRandomWeightsParams::~InitUnivormRandomWeightsParams()
{
   // TODO Auto-generated destructor stub
}

int InitUnivormRandomWeightsParams::initialize_base() {

   wMin = 0;
   wMax = 1;
   return 1;
}

int InitUnivormRandomWeightsParams::initialize(HyPerConn * parentConn) {
   InitWeightsParams::initialize(parentConn);

   PVParams * params = parent->parameters();
   int status = PV_SUCCESS;

   wMin = (float) parentConn->minWeight();
   wMax = (float) parentConn->maxWeight();
   wMin = params->value(getName(), "wMinInit", wMin);
   wMax = params->value(getName(), "wMaxInit", wMax);


   return status;

}

} /* namespace PV */
