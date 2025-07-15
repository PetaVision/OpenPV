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

PtwiseQuotientLayer::PtwiseQuotientLayer() {}

PtwiseQuotientLayer::PtwiseQuotientLayer(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
} // end PtwiseQuotientLayer::PtwiseQuotientLayer(const char *, HyPerCol *)

PtwiseQuotientLayer::~PtwiseQuotientLayer() {}

void PtwiseQuotientLayer::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
}

ActivityComponent *PtwiseQuotientLayer::createActivityComponent() {
   return new HyPerActivityComponent<PtwiseQuotientGSynAccumulator,
                                     HyPerInternalStateBuffer,
                                     HyPerActivityBuffer>(mParamsIO, mCommunicator);
}

Response::Status PtwiseQuotientLayer::allocateDataStructures() {
   auto status = HyPerLayer::allocateDataStructures();
   return status;
}

} // end namespace PV
