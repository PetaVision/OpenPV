/*
 * PtwiseProductLayer.cpp
 *
 * The output V is the pointwise product of GSynExc and GSynInh
 *
 * "Exc" and "Inh" are really misnomers for this class, but the
 * terminology is inherited from the base class.
 *
 *  Created on: Apr 25, 2011
 *      Author: peteschultz
 */

#include "PtwiseProductLayer.hpp"
#include "components/PtwiseProductInternalStateBuffer.hpp"

namespace PV {

PtwiseProductLayer::PtwiseProductLayer() { initialize_base(); }

PtwiseProductLayer::PtwiseProductLayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
} // end PtwiseProductLayer::PtwiseProductLayer(const char *, HyPerCol *)

PtwiseProductLayer::~PtwiseProductLayer() {}

int PtwiseProductLayer::initialize_base() { return PV_SUCCESS; }

int PtwiseProductLayer::initialize(const char *name, HyPerCol *hc) {
   int status = ANNLayer::initialize(name, hc);
   mLayerInput->requireChannel(0);
   mLayerInput->requireChannel(1);
   return status;
}

InternalStateBuffer *PtwiseProductLayer::createInternalState() {
   return new PtwiseProductInternalStateBuffer(getName(), parent);
}

Response::Status PtwiseProductLayer::allocateDataStructures() {
   auto status = ANNLayer::allocateDataStructures();
   pvAssert(mLayerInput->getNumChannels() >= 2);
   return status;
}

} // end namespace PV
