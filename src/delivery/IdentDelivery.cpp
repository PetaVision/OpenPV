/*
 * IdentDelivery.cpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#include "IdentDelivery.hpp"
#include <cstring>

namespace PV {

IdentDelivery::IdentDelivery(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

void IdentDelivery::initialize(char const *name, PVParams *params, Communicator const *comm) {
   BaseDelivery::initialize(name, params, comm);
}

void IdentDelivery::setObjectType() { mObjectType = "IdentDelivery"; }

void IdentDelivery::ioParam_receiveGpu(enum ParamsIOFlag ioFlag) {
   // Never receive from gpu
   mReceiveGpu = false;
   if (ioFlag == PARAMS_IO_READ) {
      parameters()->handleUnnecessaryParameter(name, "receiveGpu", false /*correctValue*/);
   }
}

Response::Status
IdentDelivery::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = BaseDelivery::communicateInitInfo(message);
   if (status != Response::SUCCESS) {
      return status;
   }

   mSingleArbor = message->mObjectTable->findObject<SingleArbor>(getName());
   pvAssert(mSingleArbor);

   checkPreAndPostDimensions();
   return Response::SUCCESS;
}

void IdentDelivery::checkPreAndPostDimensions() {
   int status = PV_SUCCESS;
   pvAssert(mPreData and mPostGSyn); // Only call this after BaseDelivery::communicateInitInfo().
   PVLayerLoc const *preLoc  = mPreData->getLayerLoc();
   PVLayerLoc const *postLoc = mPostGSyn->getLayerLoc();
   if (preLoc->nx != postLoc->nx) {
      ErrorLog().printf(
            "%s requires pre and post nx be equal (%d versus %d).\n",
            getDescription_c(),
            preLoc->nx,
            postLoc->nx);
      status = PV_FAILURE;
   }
   if (preLoc->ny != postLoc->ny) {
      ErrorLog().printf(
            "%s requires pre and post ny be equal (%d versus %d).\n",
            getDescription_c(),
            preLoc->ny,
            postLoc->ny);
      status = PV_FAILURE;
   }
   if (preLoc->nf != postLoc->nf) {
      ErrorLog().printf(
            "%s requires pre and post nf be equal (%d versus %d).\n",
            getDescription_c(),
            preLoc->nf,
            postLoc->nf);
      status = PV_FAILURE;
   }
   if (preLoc->nbatch != postLoc->nbatch) {
      ErrorLog().printf(
            "%s requires pre and post nbatch be equal (%d versus %d).\n",
            getDescription_c(),
            preLoc->nbatch,
            postLoc->nbatch);
      status = PV_FAILURE;
   }
   FatalIf(
         status != PV_SUCCESS,
         "IdentDelivery \"%s\" Error: %s and %s do not have the same dimensions.\n Dims: "
         "%dx%dx%d vs. %dx%dx%d\n",
         name,
         mPreData->getName(),
         mPostGSyn->getName(),
         preLoc->nx,
         preLoc->ny,
         preLoc->nf,
         postLoc->nx,
         postLoc->ny,
         postLoc->nf);
}

void IdentDelivery::deliver(float *destBuffer) {
   if (mChannelCode == CHANNEL_NOUPDATE) {
      return;
   }

   int delay                         = mSingleArbor->getDelay(0);
   PVLayerCube const preActivityCube = mPreData->getPublisher()->createCube(delay);
   PVLayerLoc const &preLoc          = preActivityCube.loc;
   PVLayerLoc const &postLoc         = *mPostGSyn->getLayerLoc();

   int const nx       = preLoc.nx;
   int const ny       = preLoc.ny;
   int const nf       = preLoc.nf;
   int nxPreExtended  = nx + preLoc.halo.lt + preLoc.halo.rt;
   int nyPreExtended  = ny + preLoc.halo.dn + preLoc.halo.up;
   int numPreExtended = nxPreExtended * nyPreExtended * nf;
   pvAssert(numPreExtended * preLoc.nbatch == preActivityCube.numItems);
   int numPostRestricted = nx * ny * nf;

   float *postChannel = destBuffer;
   int const nbatch   = preLoc.nbatch;
   FatalIf(
         postLoc.nbatch != nbatch,
         "%s has different presynaptic and postsynaptic batch sizes.\n",
         getDescription_c());
   for (int b = 0; b < nbatch; b++) {
      float const *preActivityBuffer = preActivityCube.data + b * numPreExtended;
      float *postGSynBuffer          = postChannel + b * numPostRestricted;
      if (preActivityCube.isSparse) {
         SparseList<float>::Entry const *activeIndices =
               (SparseList<float>::Entry *)preActivityCube.activeIndices + b * numPreExtended;
         int numActive = preActivityCube.numActive[b];
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int loopIndex = 0; loopIndex < numActive; loopIndex++) {
            int kPre = activeIndices[loopIndex].index;
            int kx   = kxPos(kPre, nxPreExtended, nyPreExtended, nf) - preLoc.halo.lt;
            int ky   = kyPos(kPre, nxPreExtended, nyPreExtended, nf) - preLoc.halo.up;
            if (kx < 0 or kx >= nx or ky < 0 or ky >= ny) {
               continue;
            }
            int kf    = featureIndex(kPre, nxPreExtended, nyPreExtended, nf);
            int kPost = kIndex(kx, ky, kf, nx, ny, nf);
            pvAssert(kPost >= 0 and kPost < numPostRestricted);
            float a = activeIndices[loopIndex].value;
            postGSynBuffer[kPost] += a;
         }
      }
      else {
         int const nk = postLoc.nx * postLoc.nf;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int y = 0; y < ny; y++) {
            int preLineIndex =
                  kIndex(preLoc.halo.lt, y + preLoc.halo.up, 0, nxPreExtended, nyPreExtended, nf);

            float const *preActivityLine = &preActivityBuffer[preLineIndex];
            int postLineIndex            = kIndex(0, y, 0, postLoc.nx, ny, postLoc.nf);
            float *postGSynLine          = &postGSynBuffer[postLineIndex];
            for (int k = 0; k < nk; k++) {
               postGSynLine[k] += preActivityLine[k];
            }
         }
      }
   }
}

void IdentDelivery::deliverUnitInput(float *recvBuffer) {
   const int numNeuronsPost = mPostGSyn->getBufferSizeAcrossBatch();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
   for (int k = 0; k < numNeuronsPost; k++) {
      recvBuffer[k] += 1.0f;
   }
}

bool IdentDelivery::isAllInputReady() const {
   bool isReady;
   if (getChannelCode() == CHANNEL_NOUPDATE) {
      isReady = true;
   }
   else {
      isReady = mPreData->isExchangeFinished(mSingleArbor->getDelay(0));
   }
   return isReady;
}

} // end namespace PV
