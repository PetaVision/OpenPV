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
#include "components/ANNActivityBuffer.hpp"
#include "components/HyPerActivityComponent.hpp"
#include "components/HyPerInternalStateBuffer.hpp"
#include "components/PtwiseProductGSynAccumulator.hpp"

namespace PV {

PtwiseProductLayer::PtwiseProductLayer() { initialize_base(); }

PtwiseProductLayer::PtwiseProductLayer(const char *name, PVParams *params, Communicator *comm) {
   initialize_base();
   initialize(name, params, comm);
} // end PtwiseProductLayer::PtwiseProductLayer(const char *, HyPerCol *)

PtwiseProductLayer::~PtwiseProductLayer() {}

int PtwiseProductLayer::initialize_base() { return PV_SUCCESS; }

void PtwiseProductLayer::initialize(const char *name, PVParams *params, Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *PtwiseProductLayer::createActivityComponent() {
   return new HyPerActivityComponent<PtwiseProductGSynAccumulator,
                                     HyPerInternalStateBuffer,
                                     ANNActivityBuffer>(getName(), parameters(), mCommunicator);
}

Response::Status PtwiseProductLayer::allocateDataStructures() {
   auto status = HyPerLayer::allocateDataStructures();
   return status;
}

} // end namespace PV
