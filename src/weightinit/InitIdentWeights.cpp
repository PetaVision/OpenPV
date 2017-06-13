/*
 * InitIdentWeights.cpp
 *
 *  Created on: Aug 14, 2011
 *      Author: kpeterson
 */

#include "InitIdentWeights.hpp"

namespace PV {

InitIdentWeights::InitIdentWeights(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

InitIdentWeights::InitIdentWeights() { initialize_base(); }

InitIdentWeights::~InitIdentWeights() {}

int InitIdentWeights::initialize_base() { return PV_SUCCESS; }

int InitIdentWeights::initialize(char const *name, HyPerCol *hc) {
   int status = InitOneToOneWeights::initialize(name, hc);
   return status;
}

void InitIdentWeights::ioParam_weightInit(enum ParamsIOFlag ioFlag) {
   mWeightInit = 1.0f;
   parent->parameters()->handleUnnecessaryParameter(name, "weightInit", 1.0f);
}

} /* namespace PV */
