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
#include "components/HyPerActivityComponent.hpp"
#include "components/HyPerInternalStateBuffer.hpp"
#include "components/PtwiseQuotientGSynAccumulator.hpp"

namespace PV {

PtwiseQuotientLayer::PtwiseQuotientLayer() { initialize_base(); }

PtwiseQuotientLayer::PtwiseQuotientLayer(const char *name, PVParams *params, Communicator *comm) {
   initialize_base();
   initialize(name, params, comm);
} // end PtwiseQuotientLayer::PtwiseQuotientLayer(const char *, HyPerCol *)

PtwiseQuotientLayer::~PtwiseQuotientLayer() {}

int PtwiseQuotientLayer::initialize_base() { return PV_SUCCESS; }

void PtwiseQuotientLayer::initialize(const char *name, PVParams *params, Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *PtwiseQuotientLayer::createActivityComponent() {
   return new HyPerActivityComponent<PtwiseQuotientGSynAccumulator,
                                     HyPerInternalStateBuffer,
                                     HyPerActivityBuffer>(getName(), parameters(), mCommunicator);
}

Response::Status PtwiseQuotientLayer::allocateDataStructures() {
   auto status = HyPerLayer::allocateDataStructures();
   return status;
}

} // end namespace PV
