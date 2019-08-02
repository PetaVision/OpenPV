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

GSynAccumulator::GSynAccumulator(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

GSynAccumulator::~GSynAccumulator() {}

void GSynAccumulator::initialize(char const *name, PVParams *params, Communicator const *comm) {
   RestrictedBuffer::initialize(name, params, comm);
   setBufferLabel("GSyn");
   mCheckpointFlag = false; // Only used internally; not checkpointed
   initializeChannelCoefficients();
}

void GSynAccumulator::setObjectType() { mObjectType = "GSynAccumulator"; }

void GSynAccumulator::initializeChannelCoefficients() { mChannelCoefficients = {1.0f, -1.0f}; }

int GSynAccumulator::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_channelIndices(ioFlag);
   ioParam_channelCoefficients(ioFlag);
   return PV_SUCCESS;
}

void GSynAccumulator::ioParam_channelIndices(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamArray(
         ioFlag, getName(), "channelIndices", &mChannelIndicesParams, &mNumChannelIndices);
}

void GSynAccumulator::ioParam_channelCoefficients(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamArray(
         ioFlag,
         getName(),
         "channelCoefficients",
         &mChannelCoefficientsParams,
         &mNumChannelCoefficients);
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

   FatalIf(
         mNumChannelCoefficients != mNumChannelIndices,
         "%s has different array lengths for ChannelIndices and ChannelCoefficients (%d versus "
         "%d).\n",
         getDescription_c(),
         mNumChannelIndices,
         mNumChannelCoefficients);
   for (int i = 0; i < mNumChannelIndices; i++) {
      int channelIndex = mChannelIndicesParams[i];
      if (channelIndex < 0) {
         continue;
      } // Should there be a warning here? A fatal error?
      if (channelIndex >= (int)mChannelCoefficients.size()) {
         mChannelCoefficients.resize(channelIndex + 1, 0.0f);
      }
      mChannelCoefficients[channelIndex] = mChannelCoefficientsParams[i];
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
