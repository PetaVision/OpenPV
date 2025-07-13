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

PtwiseProductLayer::PtwiseProductLayer() {}

PtwiseProductLayer::PtwiseProductLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
} // end PtwiseProductLayer::PtwiseProductLayer(const char *, HyPerCol *)

PtwiseProductLayer::~PtwiseProductLayer() {}

void PtwiseProductLayer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerLayer::initialize(params, defaults, comm);
}

ActivityComponent *PtwiseProductLayer::createActivityComponent() {
   return new HyPerActivityComponent<PtwiseProductGSynAccumulator,
                                     HyPerInternalStateBuffer,
                                     ANNActivityBuffer>(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

Response::Status PtwiseProductLayer::allocateDataStructures() {
   auto status = HyPerLayer::allocateDataStructures();
   return status;
}

} // end namespace PV
