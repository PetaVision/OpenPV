/*
 * PtwiseProductInternalStateBuffer.cpp
 */

#include "PtwiseProductInternalStateBuffer.hpp"

namespace PV {

PtwiseProductInternalStateBuffer::PtwiseProductInternalStateBuffer(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

PtwiseProductInternalStateBuffer::~PtwiseProductInternalStateBuffer() {}

void PtwiseProductInternalStateBuffer::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   InternalStateBuffer::initialize(name, params, comm);
}

void PtwiseProductInternalStateBuffer::setObjectType() {
   mObjectType = "PtwiseProductInternalStateBuffer";
}

Response::Status PtwiseProductInternalStateBuffer::communicateInitInfo(
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

void PtwiseProductInternalStateBuffer::updateBufferCPU(double simTime, double deltaTime) {
   float const *gSynExc = mGSyn->getChannelData(CHANNEL_EXC);
   float const *gSynInh = mGSyn->getChannelData(CHANNEL_INH);
   float *bufferData    = mBufferData.data();
   long numNeurons      = getBufferSizeAcrossBatch();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (long index = 0; index < numNeurons; index++) {
      bufferData[index] = gSynExc[index] * gSynInh[index];
   }
}

} // namespace PV
