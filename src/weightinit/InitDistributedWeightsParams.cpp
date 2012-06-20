/*
 * InitDistributedWeightsParams.cpp
 *
 *  Created on: Jun 18, 2012
 *      Author: bnowers
 *
 *  NOTES: This weight initialization class can ONLY be used in a HyPer Connection. It will
 *  not work with a Kernel Connection. The purpose of this class is to sparsely fill the patch
 *  matrix with a specified amount of neurons (nodes) that are randomly distributed throughout
 *  the matrix. To specify the number of nodes, add a numNodes parameter to the HyPerConn you
 *  wish to use in the params file.
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
   numNodes = params->value(getName(), "numNodes", numNodes);

   return status;

}

} /* namespace PV */
