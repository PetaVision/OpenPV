/*
 * GSynAccumulator.cpp
 *
 *  Created on: Nov 14, 2018
 *      Author: Pete Schultz
 */

#include "GSynAccumulator.hpp"

#undef PV_RUN_ON_GPU
#include "GSynAccumulator.kpp"

namespace PV {

GSynAccumulator::GSynAccumulator(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

GSynAccumulator::~GSynAccumulator() {}

void GSynAccumulator::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   RestrictedBuffer::initialize(params, defaults, comm);
   setBufferLabel("GSyn");
   mCheckpointFlag = false; // Only used internally; not checkpointed
   initializeChannelCoefficients();
}

void GSynAccumulator::setObjectType() { mObjectType = "GSynAccumulator"; }

void GSynAccumulator::initializeChannelCoefficients() { mChannelCoefficients = {1.0f, -1.0f}; }

int GSynAccumulator::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   ioParam_channelIndices(ioSwitch);
   ioParam_channelCoefficients(ioSwitch);
   return PV_SUCCESS;
}

void GSynAccumulator::ioParam_channelIndices(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, std::string("channelIndices"), &mChannelIndicesParams);
}

void GSynAccumulator::ioParam_channelCoefficients(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, std::string("channelCoefficients"), &mChannelCoefficientsParams);
}

Response::Status
GSynAccumulator::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   Response::Status status = RestrictedBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   mLayerInput = message->mObjectTable->findObject<LayerInputBuffer>(getName());
   FatalIf(
         mLayerInput == nullptr,
         "%s could not find a LayerInputBuffer component.\n",
         getDescription_c());

   std::size_t numChannelIndices = mChannelIndicesParams.size();
   FatalIf(
         mChannelCoefficientsParams.size() != numChannelIndices,
         "%s has different array lengths for ChannelIndices and ChannelCoefficients "
         "(%zu versus %zu).\n",
         getDescription_c(),
         numChannelIndices,
         mChannelCoefficientsParams.size());
   for (std::size_t n = 0; n < numChannelIndices; n++) {
      int channelIndex = mChannelIndicesParams[n];
      if (channelIndex < 0) {
         continue;
      } // Should there be a warning here? A fatal error?
      if (channelIndex >= (int)mChannelCoefficients.size()) {
         mChannelCoefficients.resize(channelIndex + 1, 0.0f);
      }
      mChannelCoefficients[channelIndex] = mChannelCoefficientsParams[n];
   }
   return Response::SUCCESS;
}

Response::Status GSynAccumulator::allocateDataStructures() {
   mNumInputChannels = (int)mChannelCoefficients.size();
   if (mLayerInput->getNumChannels() < mNumInputChannels) {
      mNumInputChannels = mLayerInput->getNumChannels();
   }
   return RestrictedBuffer::allocateDataStructures();
}

void GSynAccumulator::updateBufferCPU(double simTime, double deltaTime) {
   int const numNeuronsAcrossBatch = getBufferSizeAcrossBatch();
   float const *channelCoeffs      = mChannelCoefficients.data();
   float const *layerInput         = mLayerInput->getBufferData();
   float *bufferData               = mBufferData.data();
   updateGSynAccumulatorOnCPU(
         numNeuronsAcrossBatch, mNumInputChannels, channelCoeffs, layerInput, bufferData);
}

#ifdef PV_USE_CUDA
void GSynAccumulator::allocateUpdateKernel() {
   PVCuda::CudaDevice *device = mCudaDevice;

   size_t size              = mChannelCoefficients.size() * sizeof(*mChannelCoefficients.data());
   mCudaChannelCoefficients = device->createBuffer(size, &getDescription());
}

Response::Status GSynAccumulator::copyInitialStateToGPU() {
   Response::Status status = RestrictedBuffer::copyInitialStateToGPU();
   if (!Response::completed(status)) {
      return status;
   }
   if (!isUsingGPU()) {
      return status;
   }

   mCudaChannelCoefficients->copyToDevice(mChannelCoefficients.data());
   return Response::SUCCESS;
}

void GSynAccumulator::updateBufferGPU(double simTime, double deltaTime) {
   pvAssert(isUsingGPU()); // or should be in updateBufferCPU() method.
   if (!mLayerInput->isUsingGPU()) {
      mLayerInput->copyToCuda();
   }

   runKernel();
}
#endif // PV_USE_CUDA

} // namespace PV
