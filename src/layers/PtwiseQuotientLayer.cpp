/*
 * PtwiseQuotientLayer.cpp
 *
 * The output V is the pointwise division of GSynExc by GSynInh
 *
 * "Exc" and "Inh" are really misnomers for this class, but the
 * terminology is inherited from the base class.
 *
 * created by gkenyon, 06/2016g
 * based on PtwiseProductLayer Created on: Apr 25, 2011
 *      Author: peteschultz
 */

#include "PtwiseQuotientLayer.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "components/ActivityComponentWithInternalState.hpp"
#include "components/PtwiseQuotientInternalStateBuffer.hpp"

namespace PV {

PtwiseQuotientLayer::PtwiseQuotientLayer() { initialize_base(); }

PtwiseQuotientLayer::PtwiseQuotientLayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
} // end PtwiseQuotientLayer::PtwiseQuotientLayer(const char *, HyPerCol *)

PtwiseQuotientLayer::~PtwiseQuotientLayer() {}

int PtwiseQuotientLayer::initialize_base() { return PV_SUCCESS; }

int PtwiseQuotientLayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

ActivityComponent *PtwiseQuotientLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<PtwiseQuotientInternalStateBuffer,
                                                 HyPerActivityBuffer>(getName(), parent);
}

Response::Status PtwiseQuotientLayer::allocateDataStructures() {
   auto status = HyPerLayer::allocateDataStructures();
   return status;
}

} // end namespace PV
