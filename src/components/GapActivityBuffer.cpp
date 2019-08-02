/*
 * GapActivityBuffer.cpp
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#include "GapActivityBuffer.hpp"
#include "components/OriginalLayerNameParam.hpp"

// GapActivityBuffer can be used to implement gap junctions
namespace PV {
GapActivityBuffer::GapActivityBuffer() {}

GapActivityBuffer::GapActivityBuffer(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

GapActivityBuffer::~GapActivityBuffer() {}

void GapActivityBuffer::initialize(const char *name, PVParams *params, Communicator const *comm) {
   HyPerActivityBuffer::initialize(name, params, comm);
}

int GapActivityBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerActivityBuffer::ioParamsFillGroup(ioFlag);
   ioParam_ampSpikelet(ioFlag);
   return status;
}

void GapActivityBuffer::ioParam_ampSpikelet(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "ampSpikelet", &mAmpSpikelet, mAmpSpikelet);
}

Response::Status
GapActivityBuffer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = HyPerActivityBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   if (mOriginalActivity == nullptr) {
      auto *objectTable            = message->mObjectTable;
      auto *originalLayerNameParam = objectTable->findObject<OriginalLayerNameParam>(getName());
      if (!originalLayerNameParam->getInitInfoCommunicatedFlag()) {
         return Response::POSTPONE;
      }

      // Retrieve original layer's ActivityBuffer
      char const *linkedObjectName = originalLayerNameParam->getLinkedObjectName();
      mOriginalActivity            = objectTable->findObject<ActivityBuffer>(linkedObjectName);
      FatalIf(
            mOriginalActivity == nullptr,
            "%s could not find an InternalStateBuffer within %s.\n",
            getDescription_c(),
            linkedObjectName);
   }

   if (!mOriginalActivity->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   checkDimensionsEqual(mOriginalActivity, this);

   return Response::SUCCESS;
}

void GapActivityBuffer::updateBufferCPU(double simTime, double deltaTime) {
   PVLayerLoc const *loc           = getLayerLoc();
   int nx                          = loc->nx;
   int ny                          = loc->ny;
   int nf                          = loc->nf;
   PVHalo const &origHalo          = mOriginalActivity->getLayerLoc()->halo;
   int lt                          = loc->halo.lt;
   int rt                          = loc->halo.rt;
   int dn                          = loc->halo.dn;
   int up                          = loc->halo.up;
   int orig_lt                     = origHalo.lt;
   int orig_rt                     = origHalo.rt;
   int orig_dn                     = origHalo.dn;
   int orig_up                     = origHalo.up;
   float ampSpikelet               = mAmpSpikelet;
   int const numNeurons            = nx * ny * nf;
   int const numNeuronsAcrossBatch = numNeurons * loc->nbatch;
   float const *V                  = mInternalState->getBufferData();
   float const *checkActive        = mOriginalActivity->getBufferData();
   float *A                        = mBufferData.data();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (int kbatch = 0; kbatch < numNeuronsAcrossBatch; kbatch++) {
      int b                         = kbatch / numNeurons;
      int k                         = kbatch % numNeurons;
      float *ABatch                 = A + b * ((nx + lt + rt) * (ny + up + dn) * nf);
      float const *VBatch           = V + b * numNeurons;
      float const *checkActiveBatch = checkActive + b * numNeurons;
      int kex                       = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      int kexorig = kIndexExtended(k, nx, ny, nf, orig_lt, orig_rt, orig_dn, orig_up);
      ABatch[kex] = VBatch[k];
      if (checkActiveBatch[kexorig] > 0.0f) {
         ABatch[kex] += ampSpikelet;
      }
   }
}

} // end namespace PV
