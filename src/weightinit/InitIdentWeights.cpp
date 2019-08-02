/*
 * InitIdentWeights.cpp
 *
 *  Created on: Aug 14, 2011
 *      Author: kpeterson
 */

#include "InitIdentWeights.hpp"

namespace PV {

InitIdentWeights::InitIdentWeights(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

InitIdentWeights::InitIdentWeights() {}

InitIdentWeights::~InitIdentWeights() {}

void InitIdentWeights::initialize(char const *name, PVParams *params, Communicator const *comm) {
   InitOneToOneWeights::initialize(name, params, comm);
}

void InitIdentWeights::ioParam_weightInit(enum ParamsIOFlag ioFlag) {
   mWeightInit = 1.0f;
   parameters()->handleUnnecessaryParameter(name, "weightInit", 1.0f);
}

} /* namespace PV */
