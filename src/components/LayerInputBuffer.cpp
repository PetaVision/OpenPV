/*
 * LayerInputBuffer.cpp
 *
 *  Created on: Sep 13, 2018 from the original HyPerLayer
 *      Author: Pete Schultz
 */

#include "LayerInputBuffer.hpp"

namespace PV {

LayerInputBuffer::LayerInputBuffer(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

LayerInputBuffer::~LayerInputBuffer() {
   delete mReceiveInputTimer;
#ifdef PV_USE_CUDA
   delete mReceiveInputCudaTimer;
#endif // PV_USE_CUDA
}

void LayerInputBuffer::initialize(char const *name, PVParams *params, Communicator const *comm) {
   ComponentBuffer::initialize(name, params, comm);
   mExtendedFlag = false;
   setBufferLabel("GSyn");
   mCheckpointFlag = false; // GSyn doesn't get checkpointed
}

void LayerInputBuffer::initMessageActionMap() {
   ComponentBuffer::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<LayerRecvSynapticInputMessage const>(msgptr);
      return respondLayerRecvSynapticInput(castMessage);
   };
   mMessageActionMap.emplace("LayerRecvSynapticInput", action);

#ifdef PV_USE_CUDA
   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<LayerCopyFromGpuMessage const>(msgptr);
      return respondLayerCopyFromGpu(castMessage);
   };
   mMessageActionMap.emplace("LayerCopyFromGpu", action);
#endif // PV_USE_CUDA
}

void LayerInputBuffer::setObjectType() { mObjectType = "LayerInputBuffer"; }

int LayerInputBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) { return PV_SUCCESS; }

Response::Status
LayerInputBuffer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   Response::Status status = ComponentBuffer::communicateInitInfo(message);
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
   mDeliverySources.insert(mDeliverySources.begin(), deliverySource);
   // insert() is done for backward compatibility; push_back should work just as well.
}

Response::Status LayerInputBuffer::allocateDataStructures() {
   auto status = ComponentBuffer::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }

#ifdef PV_USE_CUDA
   // Separating GPU and CPU delivery sources has to wait until allocateDataStructures
   // because it depends on the delivery source's ReceiveGpu flag, which might not get
   // set until communicate, and postponing could create a postponement loop.
   auto iter = mDeliverySources.end();
   while (iter != mDeliverySources.begin()) {
      iter--;
      auto *deliverySource = *iter;
      if (deliverySource->getReceiveGpu()) {
         mGPUDeliverySources.insert(mGPUDeliverySources.begin(), deliverySource);
         mDeliverySources.erase(iter);
         useCuda();
      }
   }
#endif // PV_USE_CUDA

   return Response::SUCCESS;
}

Response::Status
LayerInputBuffer::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = ComponentBuffer::registerData(message);
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

Response::Status LayerInputBuffer::respondLayerRecvSynapticInput(
      std::shared_ptr<LayerRecvSynapticInputMessage const> message) {
// Calling LayerUpdateController has checked phase, so we don't need to do it here.
#ifdef PV_USE_CUDA
   if (message->mRecvOnGpuFlag != isUsingGPU()) {
      return Response::NO_ACTION;
   }
#endif // PV_USE_CUDA
   if (!isAllInputReady()) {
      *(message->mSomeLayerIsPending) = true;
      return Response::NO_ACTION;
   }

   updateBuffer(message->mTime, message->mDeltaT);
   return Response::SUCCESS;
}

bool LayerInputBuffer::isAllInputReady() {
   bool isReady = true;
   for (auto &d : mDeliverySources) {
      pvAssert(d);
      isReady &= d->isAllInputReady();
   }
#ifdef PV_USE_CUDA
   for (auto &d : mGPUDeliverySources) {
      pvAssert(d);
      isReady &= d->isAllInputReady();
   }
#endif // PV_USE_CUDA
   return isReady;
}

void LayerInputBuffer::updateBufferCPU(double simTime, double deltaTime) {
   resetGSynBuffers(simTime, deltaTime);
   recvAllSynapticInput(simTime, deltaTime);
   mHasReceived = true;
}

#ifdef PV_USE_CUDA
void LayerInputBuffer::updateBufferGPU(double simTime, double deltaTime) {
   // The difference between GPU and CPU is handled inside the recvAllSynapticInput() method;
   // updateBufferGPU and updateBufferCPU both need to call the same functions.
   updateBufferCPU(simTime, deltaTime);
}

Response::Status
LayerInputBuffer::respondLayerCopyFromGpu(std::shared_ptr<LayerCopyFromGpuMessage const> message) {
   copyFromCuda();
   mReceiveInputCudaTimer->accumulateTime();
   return Response::SUCCESS;
}
#endif // PV_USE_CUDA

void LayerInputBuffer::resetGSynBuffers(double simTime, double deltaTime) {
   int const sizeAcrossChannels = getBufferSizeAcrossBatch() * getNumChannels();
   float *bufferData            = mBufferData.data();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel
#endif
   for (int k = 0; k < sizeAcrossChannels; k++) {
      bufferData[k] = 0.0f;
   }
}

void LayerInputBuffer::recvAllSynapticInput(double simTime, double deltaTime) {
   // Start CPU timer here
   mReceiveInputTimer->start();

   // non-GPU sources must go before GPU sources to avoid a race condition.
   for (auto &d : mDeliverySources) {
      pvAssert(d != nullptr);
      std::ptrdiff_t channelOffset = getChannelData(d->getChannelCode()) - getBufferData();
      float *channelBuffer         = &mBufferData[channelOffset];
      d->deliver(channelBuffer);
   }

#ifdef PV_USE_CUDA
   if (isUsingGPU()) {
      copyToCuda();
      mReceiveInputCudaTimer->start();

      for (auto &d : mGPUDeliverySources) {
         pvAssert(d != nullptr);
         std::ptrdiff_t channelOffset = getChannelData(d->getChannelCode()) - getBufferData();
         float *channelBuffer         = &mBufferData[channelOffset];
         d->deliver(channelBuffer);
      }

      mReceiveInputCudaTimer->stop();
   }
#endif // PV_USE_CUDA

   mReceiveInputTimer->stop();
}

void LayerInputBuffer::recvUnitInput(float *recvBuffer, int channelCode) {
   for (auto &d : mDeliverySources) {
      if (d != nullptr and d->getChannelCode() == channelCode) {
         d->deliverUnitInput(recvBuffer);
      }
   }
}

} // namespace PV
