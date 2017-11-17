/*
 * IndexWeightConn.cpp
 *
 *  Created on: Mar 2, 2017
 *      Author: pschultz
 */

#include "IndexWeightConn.hpp"

namespace PV {

IndexWeightConn::IndexWeightConn(const char *name, HyPerCol *hc) : HyPerConn() {
   initialize(name, hc);
}

IndexWeightConn::~IndexWeightConn() {}

int IndexWeightConn::initialize(const char *name, HyPerCol *hc) {
   return HyPerConn::initialize(name, hc);
}

void IndexWeightConn::createWeightInitializer() {
   parent->parameters()->handleUnnecessaryStringParameter(name, "weightInitType", NULL);
   weightInitializer = nullptr;
}

int IndexWeightConn::setInitialValues() {
   for (int arbor = 0; arbor < numberOfAxonalArborLists(); arbor++) {
      updateWeights(arbor);
   }
   return PV_SUCCESS;
}

int IndexWeightConn::updateWeights(int axonId) {
   int nPatch = fPatchSize() * xPatchSize() * yPatchSize();
   for (int patchIndex = 0; patchIndex < getNumDataPatches(); patchIndex++) {
      float *Wdata = getWeightsDataHead(axonId, patchIndex);
      for (int kPatch = 0; kPatch < nPatch; kPatch++) {
         Wdata[kPatch] = patchIndex * nPatch + kPatch + parent->simulationTime();
      }
   }
   return PV_SUCCESS;
}

} // end of namespace PV block
