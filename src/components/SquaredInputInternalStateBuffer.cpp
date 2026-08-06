/*
 * SquaredInputInternalStateBuffer.cpp
 */

#include "SquaredInputInternalStateBuffer.hpp"

namespace PV {

SquaredInputInternalStateBuffer::SquaredInputInternalStateBuffer(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

SquaredInputInternalStateBuffer::~SquaredInputInternalStateBuffer() {}

void SquaredInputInternalStateBuffer::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   InternalStateBuffer::initialize(name, params, comm);
}

void SquaredInputInternalStateBuffer::setObjectType() {
   mObjectType = "SquaredInputInternalStateBuffer";
}

Response::Status SquaredInputInternalStateBuffer::communicateInitInfo(
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

void SquaredInputInternalStateBuffer::updateBufferCPU(double simTime, double deltaTime) {
   float const *gSynExc = mGSyn->getChannelData(CHANNEL_EXC);
   float *bufferData    = mBufferData.data();
   long numNeurons      = getBufferSizeAcrossBatch();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (long neuron = 0; neuron < numNeurons; neuron++) {
      float gSyn         = gSynExc[neuron];
      bufferData[neuron] = gSyn * gSyn;
   }
}

} // namespace PV
