/*
 * IndexWeightConn.cpp
 *
 *  Created on: Mar 2, 2017
 *      Author: pschultz
 */

#include "IndexWeightConn.hpp"
#include "IndexWeightUpdater.hpp"

namespace PV {

IndexWeightConn::IndexWeightConn(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm)
      : HyPerConn() {
   initialize(paramsIO, comm);
}

IndexWeightConn::~IndexWeightConn() {}

void IndexWeightConn::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   HyPerConn::initialize(paramsIO, comm);
}

InitWeights *IndexWeightConn::createWeightInitializer() {
   mParamsIO->handleUnnecessaryParameter("weightInitType", std::string(""));
   return nullptr;
}

BaseWeightUpdater *IndexWeightConn::createWeightUpdater() {
   return new IndexWeightUpdater(mParamsIO, mCommunicator);
}

Response::Status
IndexWeightConn::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   auto *weightUpdater = getComponentByType<IndexWeightUpdater>();
   pvAssert(weightUpdater);
   weightUpdater->respond(message);
   return Response::SUCCESS;
}

} // end of namespace PV block
