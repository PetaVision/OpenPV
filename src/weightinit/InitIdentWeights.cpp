/*
 * InitIdentWeights.cpp
 *
 *  Created on: Aug 14, 2011
 *      Author: kpeterson
 */

#include "InitIdentWeights.hpp"

namespace PV {

InitIdentWeights::InitIdentWeights(char const *name, HyPerCol *hc) { initialize(name, hc); }

InitIdentWeights::InitIdentWeights() {}

InitIdentWeights::~InitIdentWeights() {}

int InitIdentWeights::initialize(char const *name, HyPerCol *hc) {
   int status = InitOneToOneWeights::initialize(name, hc);
   return status;
}

void InitIdentWeights::ioParam_weightInit(enum ParamsIOFlag ioFlag) {
   mWeightInit = 1.0f;
   parent->parameters()->handleUnnecessaryParameter(name, "weightInit", 1.0f);
}

} /* namespace PV */
