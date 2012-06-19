/*
 * InitDistributedWeightsParams.cpp
 *
 *  Created on: Jun 18, 2012
 *      Author: bnowers
 */

#include "InitDistributedWeightsParams.hpp"

namespace PV {

InitDistributedWeightsParams::InitDistributedWeightsParams()
{
   initialize_base();
}

InitDistributedWeightsParams::InitDistributedWeightsParams(HyPerConn * parentConn)
                     : InitWeightsParams() {
   initialize_base();
   initialize(parentConn);
}

InitDistributedWeightsParams::~InitDistributedWeightsParams()
{
   // TODO Auto-generated destructor stub
}

int InitDistributedWeightsParams::initialize_base() {

   wMin = 0;
   wMax = 1;
   numNodes = 1;
   return 1;
}

int InitDistributedWeightsParams::initialize(HyPerConn * parentConn) {
   InitWeightsParams::initialize(parentConn);

   PVParams * params = parent->parameters();
   int status = PV_SUCCESS;

   // minWeight and maxWeight require arborId, use default values instead if wMinInit, wMaxInit not provided by user
   //wMin = (float) parentConn->minWeight();
   //wMax = (float) parentConn->maxWeight();
   wMin = params->value(getName(), "wMinInit", wMin);
   wMax = params->value(getName(), "wMaxInit", wMax);


   return status;

}

} /* namespace PV */
