/*
 * OccludingGSynAccumulator.cpp
 *
 *  Created on: Jul 18, 2019
 *      Author: Jacob Springer
 */

#include "OccludingGSynAccumulator.hpp"

#undef PV_RUN_ON_GPU
#include "OccludingGSynAccumulator.kpp"

namespace PV {

OccludingGSynAccumulator::OccludingGSynAccumulator(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

OccludingGSynAccumulator::~OccludingGSynAccumulator() {}

void OccludingGSynAccumulator::initialize(char const *name, PVParams *params, Communicator const *comm) {
   RestrictedBuffer::initialize(name, params, comm);
   setBufferLabel("GSyn");
   mCheckpointFlag = false; // Only used internally; not checkpointed
}

void OccludingGSynAccumulator::setObjectType() { mObjectType = "OccludingGSynAccumulator"; }

void OccludingGSynAccumulator::ioParam_opaqueMagnitude(enum ParamsIOFlag ioFlag) {
   this->parameters()->ioParamValue(
           ioFlag, this->getName(), "mOpaqueMagnitude", &mOpaqueMagnitude, mOpaqueMagnitude, true);
}

int OccludingGSynAccumulator::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_opaqueMagnitude(ioFlag);
   return PV_SUCCESS;
}

Response::Status
OccludingGSynAccumulator::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   Response::Status status = RestrictedBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   mLayerInput = message->mObjectTable->findObject<LayerInputBuffer>(getName());
   FatalIf(
         mLayerInput == nullptr,
         "%s could not find a LayerInputBuffer component.\n",
         getDescription_c());

   return Response::SUCCESS;
}

Response::Status OccludingGSynAccumulator::allocateDataStructures() {
   PVLayerLoc const *loc           = getLayerLoc();
   mNumChannels = mLayerInput->getNumChannels();
   mContribData.resize(loc->nx * loc->ny * loc->nbatch * mNumChannels, 1.0); 
   
   return RestrictedBuffer::allocateDataStructures();
}

void OccludingGSynAccumulator::updateBufferCPU(double simTime, double deltaTime) {
   PVLayerLoc const *loc           = getLayerLoc();
   int const numNeuronsAcrossBatch = getBufferSizeAcrossBatch();
   int const numPixelsAcrossBatch  = loc->nbatch * loc->nx * loc->ny;
   float const *layerInput         = mLayerInput->getBufferData();
   float *bufferData               = mBufferData.data();
   float *contribData              = mContribData.data();
   updateOccludingGSynAccumulatorOnCPU(
         loc->nbatch, loc->nx, loc->ny, loc->nf, mNumChannels, mOpaqueMagnitude, layerInput, bufferData, contribData);
}

float const* OccludingGSynAccumulator::retrieveContribData() {
   if (isUsingGPU()) {
      mCudaContribData->copyFromDevice(mContribData.data());
   }
   return mContribData.data();
}

#ifdef PV_USE_CUDA
void OccludingGSynAccumulator::allocateUpdateKernel() {
   PVCuda::CudaDevice *device = mCudaDevice;
   size_t size = mContribData.size() * sizeof(*mContribData.data());
   mCudaContribData = device->createBuffer(size, &getDescription());
}

Response::Status OccludingGSynAccumulator::copyInitialStateToGPU() {
   Response::Status status = RestrictedBuffer::copyInitialStateToGPU();
   if (!Response::completed(status)) {
      return status;
   }
   if (!isUsingGPU()) {
      return status;
   }

   mCudaContribData->copyToDevice(mContribData.data());
   return Response::SUCCESS;
}

void OccludingGSynAccumulator::updateBufferGPU(double simTime, double deltaTime) {
   pvAssert(isUsingGPU()); // or should be in updateBufferCPU() method.
   if (!mLayerInput->isUsingGPU()) {
      mLayerInput->copyToCuda();
   }

   runKernel();
}
#endif // PV_USE_CUDA

} // namespace PV
