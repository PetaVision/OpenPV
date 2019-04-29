/*
 * IndexWeightConn.cpp
 *
 *  Created on: Mar 2, 2017
 *      Author: pschultz
 */

#include "IndexWeightConn.hpp"
#include "IndexWeightUpdater.hpp"

namespace PV {

IndexWeightConn::IndexWeightConn(const char *name, PVParams *params, Communicator const *comm)
      : HyPerConn() {
   initialize(name, params, comm);
}

IndexWeightConn::~IndexWeightConn() {}

void IndexWeightConn::initialize(const char *name, PVParams *params, Communicator const *comm) {
   HyPerConn::initialize(name, params, comm);
}

InitWeights *IndexWeightConn::createWeightInitializer() {
   parameters()->handleUnnecessaryStringParameter(name, "weightInitType", nullptr);
   return nullptr;
}

BaseWeightUpdater *IndexWeightConn::createWeightUpdater() {
   return new IndexWeightUpdater(name, parameters(), mCommunicator);
}

Response::Status
IndexWeightConn::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   auto *weightUpdater = getComponentByType<IndexWeightUpdater>();
   pvAssert(weightUpdater);
   weightUpdater->respond(message);
   return Response::SUCCESS;
}

} // end of namespace PV block
