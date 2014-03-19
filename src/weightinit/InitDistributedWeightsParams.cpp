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
}

int InitDistributedWeightsParams::initialize_base() {
   numNodes = 1;
   return 1;
}

int InitDistributedWeightsParams::initialize(HyPerConn * parentConn) {
   return InitWeightsParams::initialize(parentConn);
}

int InitDistributedWeightsParams::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitWeightsParams::ioParamsFillGroup(ioFlag);
   parent->ioParamValue(ioFlag, name, "numNodes", &numNodes, numNodes);
   return status;
}

} /* namespace PV */
