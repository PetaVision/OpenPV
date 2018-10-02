/*
 * IndexWeightConn.cpp
 *
 *  Created on: Mar 2, 2017
 *      Author: pschultz
 */

#include "IndexWeightConn.hpp"
#include "IndexWeightUpdater.hpp"

namespace PV {

IndexWeightConn::IndexWeightConn(const char *name, HyPerCol *hc) : HyPerConn() {
   initialize(name, hc);
}

IndexWeightConn::~IndexWeightConn() {}

int IndexWeightConn::initialize(const char *name, HyPerCol *hc) {
   return HyPerConn::initialize(name, hc);
}

InitWeights *IndexWeightConn::createWeightInitializer() {
   parameters()->handleUnnecessaryStringParameter(name, "weightInitType", nullptr);
   return nullptr;
}

BaseWeightUpdater *IndexWeightConn::createWeightUpdater() {
   return new IndexWeightUpdater(name, parent);
}

Response::Status
IndexWeightConn::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   auto *weightUpdater = mObserverTable.lookupByType<IndexWeightUpdater>();
   pvAssert(weightUpdater);
   weightUpdater->initializeWeights();
   return Response::SUCCESS;
}

} // end of namespace PV block
