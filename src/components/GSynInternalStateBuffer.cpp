/*
 * GSynInternalStateBuffer.cpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 */

#include "GSynInternalStateBuffer.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

GSynInternalStateBuffer::GSynInternalStateBuffer(char const *name, HyPerCol *hc) {
   initialize(name, hc);
}

GSynInternalStateBuffer::~GSynInternalStateBuffer() {}

int GSynInternalStateBuffer::initialize(char const *name, HyPerCol *hc) {
   int status = InternalStateBuffer::initialize(name, hc);
   return status;
}

void GSynInternalStateBuffer::setObjectType() { mObjectType = "GSynInternalStateBuffer"; }

Response::Status GSynInternalStateBuffer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   Response::Status status = InternalStateBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   int const maxIterations = 1; // Limits the depth of recursion when searching for dependencies.
   mLayerInput = message->mHierarchy->lookupByTypeRecursive<LayerInputBuffer>(maxIterations);
   FatalIf(
         mLayerInput == nullptr,
         "%s could not find a LayerInputBuffer component.\n",
         getDescription_c());
   requireInputChannels();

   return Response::SUCCESS;
}

void GSynInternalStateBuffer::requireInputChannels() {}

} // namespace PV
