/*
 * LayerInputBuffer.cpp
 *
 *  Created on: Sep 13, 2018
 *      Author: Pete Schultz
 */

#include "LayerInputBuffer.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

LayerInputBuffer::LayerInputBuffer(char const *name, HyPerCol *hc) { initialize(name, hc); }

LayerInputBuffer::~LayerInputBuffer() {
   delete mReceiveInputTimer;
   delete mReceiveInputCudaTimer;
}

int LayerInputBuffer::initialize(char const *name, HyPerCol *hc) {
   int status    = BufferComponent::initialize(name, hc);
   mExtendedFlag = false;
   mBufferLabel  = ""; // GSyn doesn't get checkpointed
   return status;
}

void LayerInputBuffer::initMessageActionMap() {
   BufferComponent::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<LayerClearProgressFlagsMessage const>(msgptr);
      return respondLayerClearProgressFlags(castMessage);
   };
   mMessageActionMap.emplace("LayerClearProgressFlags", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<LayerRecvSynapticInputMessage const>(msgptr);
      return respondLayerRecvSynapticInput(castMessage);
   };
   mMessageActionMap.emplace("LayerRecvSynapticInput", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<LayerCopyFromGpuMessage const>(msgptr);
      return respondLayerCopyFromGpu(castMessage);
   };
   mMessageActionMap.emplace("LayerCopyFromGpu", action);
}

void LayerInputBuffer::setObjectType() { mObjectType = "LayerInputBuffer"; }

int LayerInputBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) { return PV_SUCCESS; }

Response::Status
LayerInputBuffer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   Response::Status status = BufferComponent::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   return Response::SUCCESS;
}

void LayerInputBuffer::requireChannel(int channelNeeded) {
   if (channelNeeded >= mNumChannels) {
      mNumChannels = channelNeeded + 1;
   }
}

void LayerInputBuffer::addDeliverySource(LayerInputDelivery *deliverySource) {
   // Don't add the same source twice.
   for (auto &d : mDeliverySources) {
      if (d == deliverySource) {
         return;
      }
   }
#ifdef PV_USE_CUDA
   // CPU connections must run first to avoid race conditions
   if (!deliverySource->getReceiveGpu()) {
      mDeliverySources.insert(mDeliverySources.begin(), deliverySource);
   }
   // Otherwise, add to the back. If no gpus at all, just add to back
   else {
      mDeliverySources.push_back(deliverySource);
      // If it is receiving from gpu, set layer flag as such
      useCuda();
   }
#else
   mDeliverySources.push_back(deliverySource);
#endif
}

Response::Status LayerInputBuffer::allocateDataStructures() {
   auto status = BufferComponent::allocateDataStructures();
   if (mNumChannels >= 0) {
      mChannelTimeConstants.resize(mNumChannels);
   }
   initChannelTimeConstants();
   return status;
}

void LayerInputBuffer::initChannelTimeConstants() {
   for (auto &c : mChannelTimeConstants) {
      c = 0.0f;
   }
}

Response::Status
LayerInputBuffer::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = BufferComponent::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }

   auto *checkpointer = message->mDataRegistry;
   mReceiveInputTimer = new Timer(getName(), "layer", "recvsyn");
   checkpointer->registerTimer(mReceiveInputTimer);
#ifdef PV_USE_CUDA
   auto cudaDevice = mCudaDevice;
   if (cudaDevice) {
      mReceiveInputCudaTimer = new PVCuda::CudaTimer(getName(), "layer", "gpurecvsyn");
      mReceiveInputCudaTimer->setStream(cudaDevice->getStream());
      checkpointer->registerTimer(mReceiveInputCudaTimer);
   }
#endif // PV_USE_CUDA
   return Response::SUCCESS;
}

Response::Status LayerInputBuffer::respondLayerClearProgressFlags(
      std::shared_ptr<LayerClearProgressFlagsMessage const> message) {
   mHasReceived = false;
   return Response::SUCCESS;
}

Response::Status LayerInputBuffer::respondLayerRecvSynapticInput(
      std::shared_ptr<LayerRecvSynapticInputMessage const> message) {
   Response::Status status = Response::SUCCESS;
// Calling HyPerLayer has checked phase, so we don't need to do it here.
#ifdef PV_USE_CUDA
   if (message->mRecvOnGpuFlag != isUsingGPU()) {
      return status;
   }
#endif // PV_USE_CUDA
   if (mHasReceived) {
      return status;
   }
   if (*(message->mSomeLayerHasActed) or !isAllInputReady()) {
      *(message->mSomeLayerIsPending) = true;
      return status;
   }
   updateBuffer(message->mTime, message->mDeltaT);
   mHasReceived                   = true;
   *(message->mSomeLayerHasActed) = true;
   return status;
}

bool LayerInputBuffer::isAllInputReady() {
   bool isReady = true;
   for (auto &d : mDeliverySources) {
      pvAssert(d);
      isReady &= d->isAllInputReady();
   }
   return isReady;
}

void LayerInputBuffer::resetGSynBuffers(double simulationTime, double deltaTime) {
   int const sizeAcrossChannels = getBufferSizeAcrossBatch() * getNumChannels();
   float *bufferData            = mBufferData.data();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel
#endif
   for (int k = 0; k < sizeAcrossChannels; k++) {
      bufferData[k] = 0.0f;
   }
}

void LayerInputBuffer::recvAllSynapticInput(double simulationTime, double deltaTime) {
   bool switchGpu = false;
   // Start CPU timer here
   mReceiveInputTimer->start();

   for (auto &d : mDeliverySources) {
      pvAssert(d != nullptr);
#ifdef PV_USE_CUDA
      // Check if it's done with cpu connections
      if (!switchGpu && d->getReceiveGpu()) {
         // Copy GSyn over to GPU
         copyToCuda();
         // Start gpu timer
         mReceiveInputCudaTimer->start();
         switchGpu = true;
      }
#endif
      std::ptrdiff_t channelOffset = getChannelData(d->getChannelCode()) - getBufferData();
      float *channelBuffer         = &mBufferData[channelOffset];
      d->deliver(channelBuffer);
   }
#ifdef PV_USE_CUDA
   if (switchGpu) {
      // Stop timer
      mReceiveInputCudaTimer->stop();
   }
#endif
   mReceiveInputTimer->stop();
}

Response::Status
LayerInputBuffer::respondLayerCopyFromGpu(std::shared_ptr<LayerCopyFromGpuMessage const> message) {
   copyFromCuda();
   mReceiveInputCudaTimer->accumulateTime();
   return Response::SUCCESS;
}

void LayerInputBuffer::updateBuffer(double simTime, double deltaTime) {
   resetGSynBuffers(simTime, deltaTime);
   recvAllSynapticInput(simTime, deltaTime);
}

} // namespace PV
