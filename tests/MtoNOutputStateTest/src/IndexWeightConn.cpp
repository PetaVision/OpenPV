/*
 * IndexWeightConn.cpp
 *
 *  Created on: Mar 2, 2017
 *      Author: pschultz
 */

#include "IndexWeightConn.hpp"
#include "IndexWeightUpdater.hpp"

namespace PV {

IndexWeightConn::IndexWeightConn(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm)
      : HyPerConn() {
   initialize(params, defaults, comm);
}

IndexWeightConn::~IndexWeightConn() {}

void IndexWeightConn::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerConn::initialize(params, defaults, comm);
}

InitWeights *IndexWeightConn::createWeightInitializer() {
   mParamsIO->handleUnnecessaryParameter("weightInitType", std::string(""));
   return nullptr;
}

BaseWeightUpdater *IndexWeightConn::createWeightUpdater() {
   return new IndexWeightUpdater(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

Response::Status
IndexWeightConn::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   auto *weightUpdater = getComponentByType<IndexWeightUpdater>();
   pvAssert(weightUpdater);
   weightUpdater->respond(message);
   return Response::SUCCESS;
}

} // end of namespace PV block
