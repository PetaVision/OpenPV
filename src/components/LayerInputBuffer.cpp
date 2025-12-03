/*
 * LayerInputBuffer.cpp
 *
 *  Created on: Sep 13, 2018 from the original HyPerLayer
 *      Author: Pete Schultz
 */

#include "LayerInputBuffer.hpp"
#include <vector>

namespace PV {

LayerInputBuffer::LayerInputBuffer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

LayerInputBuffer::~LayerInputBuffer() {
   delete mBroadcastReduceTimer;
   delete mReceiveInputTimer;
#ifdef PV_USE_CUDA
   delete mCopyFromCudaTimer;
   delete mReceiveInputCudaTimer;
#endif // PV_USE_CUDA
}

void LayerInputBuffer::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   ComponentBuffer::initialize(paramsIO, comm);
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

int LayerInputBuffer::ioParamsFillGroup(ParamsIOSwitch ioSwitch) { return PV_SUCCESS; }

Response::Status
LayerInputBuffer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   Response::Status status = ComponentBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   mLayerGeometry = message->mObjectTable->findObject<LayerGeometry>(getName());
   FatalIf(
         mLayerGeometry == nullptr,
         "%s could not find a LayerInputBuffer component.\n",
         getDescription_c());

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

MPI_Op LayerInputBuffer::setReductionOp() {
   if (!mDeliverySources.empty()) {
      mMPIReductionOp = mDeliverySources[0]->getMPIReductionOp();
      for (auto *d : mDeliverySources) {
         FatalIf(
               d->getMPIReductionOp() != mMPIReductionOp,
               "Connections \"%s\" and \"%s\" both connect to layer \"%s\", "
               "but they have different reduction types.\n",
               mDeliverySources[0]->getName(), d->getName(), getName());
      }
   }
   return mMPIReductionOp;
}

Response::Status LayerInputBuffer::allocateDataStructures() {
   auto status = ComponentBuffer::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }

   // Checking that all delivery sources have the same reduction type (max vs. sum vs. ?)
   // waits until allocateDataStructures because the delivery source might not set its reduction
   // type until communicate, and this is the easiest way to avoid a postponement loop.
   setReductionOp();

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

   auto *checkpointer    = message->mDataRegistry;
   mBroadcastReduceTimer = new Timer(getName(), "layer", "broadcast");
   mReceiveInputTimer    = new Timer(getName(), "layer", "recvsyn");
   checkpointer->registerTimer(mBroadcastReduceTimer);
   checkpointer->registerTimer(mReceiveInputTimer);
#ifdef PV_USE_CUDA
   auto cudaDevice = mCudaDevice;
   if (cudaDevice) {
      mReceiveInputCudaTimer = new PVCuda::CudaTimer(getName(), "layer", "gpurecvsyn");
      mReceiveInputCudaTimer->setStream(cudaDevice->getStream());
      checkpointer->registerTimer(mReceiveInputCudaTimer);
      mCopyFromCudaTimer = new Timer(getName(), "layer", "InputFrGPU");
      checkpointer->registerTimer(mCopyFromCudaTimer);
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
   mCopyFromCudaTimer->start();
   copyFromCuda();
   mReceiveInputCudaTimer->accumulateTime();
   mCopyFromCudaTimer->stop();
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
      auto channelOffset = getChannelData(d->getChannelCode()) - getBufferData();
      float *channelBuffer         = &mBufferData[channelOffset];
      float reductionMultiplier = d->getReductionMultiplier();
      if (reductionMultiplier == 1.0f) {
         d->deliver(channelBuffer);
      }
      else {
         std::vector<float> buffer(getBufferSizeAcrossBatch());
         d->deliver(buffer.data());
         for (int k = 0; k < getBufferSizeAcrossBatch(); ++k) {
            channelBuffer[k] += reductionMultiplier * buffer[k];
         }
      }
   }

#ifdef PV_USE_CUDA
   if (isUsingGPU()) {
      copyToCuda();
      mReceiveInputCudaTimer->start();

      for (auto &d : mGPUDeliverySources) {
         pvAssert(d != nullptr);
         auto channelOffset = getChannelData(d->getChannelCode()) - getBufferData();
         float *channelBuffer         = &mBufferData[channelOffset];
         float reductionMultiplier = d->getReductionMultiplier();
         if (reductionMultiplier == 1.0f) {
            d->deliver(channelBuffer);
         }
         else {
            std::vector<float> buffer(getBufferSizeAcrossBatch());
            d->deliver(buffer.data());
            for (int k = 0; k < getBufferSizeAcrossBatch(); ++k) {
               channelBuffer[k] += reductionMultiplier * buffer[k];
            }
         }
      }

      mReceiveInputCudaTimer->stop();
   }
#endif // PV_USE_CUDA

   mReceiveInputTimer->stop();

   if (mLayerGeometry->getBroadcastFlag() and !mDeliverySources.empty()) {
      mBroadcastReduceTimer->start();
      MPI_Allreduce(
            MPI_IN_PLACE, 
            mBufferData.data(), 
            mBufferData.size(),
            MPI_FLOAT,
            mMPIReductionOp,
            getCommunicator()->communicator());
      mBroadcastReduceTimer->stop();
   }
}

void LayerInputBuffer::recvUnitInput(float *recvBuffer, int channelCode) {
   for (auto &d : mDeliverySources) {
      if (d != nullptr and d->getChannelCode() == channelCode) {
         d->deliverUnitInput(recvBuffer);
      }
   }
}

} // namespace PV
