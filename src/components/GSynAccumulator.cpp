/*
 * GSynAccumulator.cpp
 *
 *  Created on: Nov 14, 2018
 *      Author: Pete Schultz
 */

#include "GSynAccumulator.hpp"

namespace PV {

GSynAccumulator::GSynAccumulator(char const *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

GSynAccumulator::~GSynAccumulator() {}

void GSynAccumulator::initialize(char const *name, PVParams *params, Communicator *comm) {
   RestrictedBuffer::initialize(name, params, comm);
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
   int const maxIterations = 1; // Limits the depth of recursion when searching for dependencies.
   mLayerInput = message->mHierarchy->lookupByTypeRecursive<LayerInputBuffer>(maxIterations);
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

   for (std::size_t ch = (std::size_t)0; ch < mChannelCoefficients.size(); ch++) {
      if (mChannelCoefficients[ch] != 0.0f) {
         mLayerInput->requireChannel(ch);
      }
   }

   return Response::SUCCESS;
}

void GSynAccumulator::updateBufferCPU(double simTime, double deltaTime) {
   const PVLayerLoc *loc = getLayerLoc();
   float *bufferData     = mBufferData.data();
   int numNeurons        = getBufferSizeAcrossBatch();
   int numChannels       = (int)mChannelCoefficients.size();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (int kIndex = 0; kIndex < numNeurons; kIndex++) {
      bufferData[kIndex] = 0.0f;
      for (int ch = 0; ch < numChannels; ch++) {
         bufferData[kIndex] += mChannelCoefficients[ch] * mLayerInput->getChannelData(ch)[kIndex];
      }
   }
}

#ifdef PV_USE_CUDA
void GSynAccumulator::allocateUpdateKernel() {
   PVCuda::CudaDevice *device = mCudaDevice;

   size_t size              = mChannelCoefficients.size() * sizeof(*mChannelCoefficients.data());
   mCudaChannelCoefficients = device->createBuffer(size, &getDescription());

   mCudaUpdateKernel = new PVCuda::CudaUpdateGSynAccumulator(mCudaDevice);
}

Response::Status GSynAccumulator::copyInitialStateToGPU() {
   Response::Status status = RestrictedBuffer::copyInitialStateToGPU();
   if (!Response::completed(status)) {
      return status;
   }
   if (!isUsingGPU()) {
      return status;
   }

   // Set arguments of update kernel
   const PVLayerLoc *loc = getLayerLoc();
   int const numNeurons  = getBufferSize();
   int const nbatch      = loc->nbatch;
   pvAssert(getCudaBuffer());
   PVCuda::CudaBuffer *layerInputCudaBuffer = mLayerInput->getCudaBuffer();
   pvAssert(layerInputCudaBuffer);

   pvAssert(mCudaUpdateKernel);

   mCudaChannelCoefficients->copyToDevice(mChannelCoefficients.data());

   // Set arguments to kernel
   auto *cudaUpdateKernel = dynamic_cast<PVCuda::CudaUpdateGSynAccumulator *>(mCudaUpdateKernel);
   pvAssert(cudaUpdateKernel);
   cudaUpdateKernel->setArgs(
         nbatch,
         numNeurons,
         mNumChannels,
         mCudaChannelCoefficients,
         layerInputCudaBuffer,
         getCudaBuffer());
   return Response::SUCCESS;
}

void GSynAccumulator::updateBufferGPU(double simTime, double deltaTime) {
   pvAssert(isUsingGPU()); // or should be in updateBufferCPU() method.
   if (!mLayerInput->isUsingGPU()) {
      mLayerInput->copyToCuda();
   }

   // Sync all buffers before running
   mCudaDevice->syncDevice();

   // Run kernel
   mCudaUpdateKernel->run();
}
#endif // PV_USE_CUDA

} // namespace PV
