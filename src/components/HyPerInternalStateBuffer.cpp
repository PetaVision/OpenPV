/*
 * HyPerInternalStateBuffer.cpp
 *
 *  Created on: Oct 12, 2018 from the original HyPerLayer
 *      Author: Pete Schultz
 */

#include <algorithm>
#include "HyPerInternalStateBuffer.hpp"

#undef PV_RUN_ON_GPU
#include "HyPerInternalStateBuffer.kpp"

namespace PV {

HyPerInternalStateBuffer::HyPerInternalStateBuffer(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

HyPerInternalStateBuffer::~HyPerInternalStateBuffer() {}

void HyPerInternalStateBuffer::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   InternalStateBuffer::initialize(name, params, comm);
}

void HyPerInternalStateBuffer::setObjectType() { mObjectType = "HyPerInternalStateBuffer"; }

int HyPerInternalStateBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InternalStateBuffer::ioParamsFillGroup(ioFlag);
   if (status != PV_SUCCESS) { return status; }
   ioParam_channelIndices(ioFlag);
   ioParam_channelCoefficients(ioFlag);
   return PV_SUCCESS;
}

void HyPerInternalStateBuffer::ioParam_channelIndices(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamArray(
         ioFlag,
         getName(),
         "channelIndices",
         &mChannelIndicesParams,
         &mNumChannelIndicesParams);
}

void HyPerInternalStateBuffer::ioParam_channelCoefficients(enum ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(getName(), "channelIndices"));
   int numChannelCoefficients = mNumChannelIndicesParams;
   parameters()->ioParamArray(
         ioFlag,
         getName(),
         "channelCoefficients",
         &mChannelCoefficientsParams,
         &numChannelCoefficients);
   FatalIf(
         numChannelCoefficients != mNumChannelIndicesParams,
         "Layer \"%s\" has different array lengths for ChannelIndices and ChannelCoefficients "
         "(%d versus %d).\n",
         getName(),
         mNumChannelIndicesParams,
         numChannelCoefficients);
}

Response::Status HyPerInternalStateBuffer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   Response::Status status = InternalStateBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   mGSyn = message->mObjectTable->findObject<LayerInputBuffer>(getName());
   FatalIf(
         mGSyn == nullptr,
         "%s could not find a LayerInputBuffer (GSyn) component.\n",
         getDescription_c());
   return Response::SUCCESS;
}

Response::Status HyPerInternalStateBuffer::allocateDataStructures() {
   mChannelIndices = {0, 1};
   mChannelCoefficients = {1.0f, -1.0f};
   if (mGSyn->getNumChannels() < 2) {
      mChannelIndices.resize(mGSyn->getNumChannels());
      mChannelCoefficients.resize(mGSyn->getNumChannels());
   }
   int numDefaultChannels = static_cast<int>(mChannelIndices.size());

   for (int ch = 0; ch < mNumChannelIndicesParams; ++ch) {
      int channelNumber = mChannelIndicesParams[ch];
      float coeff       = mChannelCoefficientsParams[ch];
      if (channelNumber < numDefaultChannels) {
         mChannelCoefficients[channelNumber] = coeff;
      }
      else if (channelNumber >= mGSyn->getNumChannels()) {
         WarnLog().printf(
               "Layer %s channel index %d, coefficient %f is in params, but is not used.\n",
               getName(), mChannelIndicesParams[ch], (double)mChannelCoefficientsParams[ch]);
      }
      else {
         auto location = find(mChannelIndices.begin(), mChannelIndices.end(), channelNumber);
         if (location == mChannelIndices.end()) {
            mChannelIndices.push_back(mChannelIndicesParams[ch]);
            mChannelCoefficients.push_back(mChannelCoefficientsParams[ch]);
         }
         else {
            auto offset = location - mChannelIndices.begin();
            assert(mChannelIndices[offset] == channelNumber);
            mChannelCoefficients[offset] = coeff;
         }
      }
   }

   assert(mChannelCoefficients.size() == mChannelIndices.size());
   mNumChannelIndices = static_cast<int>(mChannelIndices.size());
   return InternalStateBuffer::allocateDataStructures();
}

void HyPerInternalStateBuffer::updateBufferCPU(double simTime, double deltaTime) {
   updateHyPerInternalStateBufferOnCPU(
         getBufferSizeAcrossBatch(),
         mNumChannelIndices,
         mChannelIndices.data(),
         mChannelCoefficients.data(),
         mGSyn->getBufferData(),
         mBufferData.data());
}

#ifdef PV_USE_CUDA
void HyPerInternalStateBuffer::allocateUpdateKernel() {
   PVCuda::CudaDevice *device = mCudaDevice;

   std::size_t size         = (std::size_t)mNumChannelIndices * sizeof(mChannelIndices[0]);
   std::string description  = getDescription() + " indices";
   mCudaChannelIndices      = device->createBuffer(size, &description);
   size                     = (std::size_t)mNumChannelIndices * sizeof(mChannelCoefficients[0]);
   description              = getDescription() + " coefficients";
   mCudaChannelCoefficients = device->createBuffer(size, &description);
}

Response::Status HyPerInternalStateBuffer::copyInitialStateToGPU() {
   Response::Status status = RestrictedBuffer::copyInitialStateToGPU();
   if (!Response::completed(status)) {
      return status;
   }
   if (!isUsingGPU()) {
      return status;
   }

   mCudaChannelIndices->copyToDevice(mChannelIndices.data());
   mCudaChannelCoefficients->copyToDevice(mChannelCoefficients.data());
   return Response::SUCCESS;
}

void HyPerInternalStateBuffer::updateBufferGPU(double simTime, double deltaTime) {
   pvAssert(isUsingGPU()); // or should be in updateBufferCPU() method.
   if (!mGSyn->isUsingGPU()) {
      mGSyn->copyToCuda();
   }

   runKernel();
}
#endif // PV_USE_CUDA

} // namespace PV
