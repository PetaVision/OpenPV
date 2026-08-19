/*
 * HyPerLayer.cpp
 *
 *  Created on: Jul 29, 2008
 *
 *  The top of the hierarchy for layer classes.
*/

#include "HyPerLayer.hpp"
#include "components/HyPerActivityBuffer.hpp"
#include "components/HyPerActivityComponent.hpp"
#include "components/HyPerInternalStateBuffer.hpp"
#include "components/PublisherComponent.hpp"
#include "observerpattern/ObserverTable.hpp"
#include "utils/BufferUtilsMPI.hpp"

namespace PV {

// Default constructor. Any derived classes' constructors should call this constructor, not
// the constructor with arguments. See the comments for the default constructor in BaseLayer.cpp
// for details.
HyPerLayer::HyPerLayer() {}

HyPerLayer::HyPerLayer(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

HyPerLayer::~HyPerLayer() {}

void HyPerLayer::initialize(const char *name, PVParams *params, Communicator const *comm) {
   BaseLayer::initialize(name, params, comm);
}

/******************************************************************
 * Create components
 *****************************************************************/
void HyPerLayer::fillComponentTable() {
   BaseLayer::fillComponentTable();
   mLayerInput = createLayerInput();
   if (mLayerInput) {
      addUniqueComponent(mLayerInput);
   }
}

LayerInputBuffer *HyPerLayer::createLayerInput() {
   return new LayerInputBuffer(getName(), parameters(), mCommunicator);
}

ActivityComponent *HyPerLayer::createActivityComponent() {
   return new HyPerActivityComponent<HyPerInternalStateBuffer, HyPerActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

Response::Status
HyPerLayer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   Response::Status status = BaseLayer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

#ifdef PV_USE_CUDA
   // Here, the connection tells all participating recv layers to allocate memory on gpu
   // if receive from gpu is set. These buffers should be set in allocate
   if (mActivityComponent and mActivityComponent->getUpdateGpu()) {
      if (mLayerInput) {
         mLayerInput->useCuda();
      }
   }
#endif

   return Response::SUCCESS;
}

#ifdef PV_USE_CUDA
Response::Status
HyPerLayer::layerCopyFromGpu(std::shared_ptr<LayerCopyFromGpuMessage const> message) {
   if (mActivityComponent and mActivityComponent->isUsingGPU()) {
      mActivityComponent->copyFromCuda();
   }
   if (mLayerInput and mLayerInput->isUsingGPU()) {
      mLayerInput->respond(message);
   }
   Response::Status status = Response::SUCCESS;
   return status;
}
#endif // PV_USE_CUDA

} // end of PV namespace
