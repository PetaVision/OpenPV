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
   parent->parameters()->handleUnnecessaryStringParameter(name, "weightInitType", nullptr);
   return nullptr;
}

BaseWeightUpdater *IndexWeightConn::createWeightUpdater() {
   return new IndexWeightUpdater(name, parent);
}

int IndexWeightConn::initializeState() {
   auto *weightUpdater =
         mapLookupByType<IndexWeightUpdater>(mComponentTable.getObjectMap(), getDescription());
   weightUpdater->initializeWeights();
   return PV_SUCCESS;
}

} // end of namespace PV block
