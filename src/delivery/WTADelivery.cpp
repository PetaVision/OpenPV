/*
 * WTADelivery.cpp
 *
 *  Created on: Aug 15, 2018
 *      Author: Pete Schultz
 */

#include "WTADelivery.hpp"
#include <cstring>

namespace PV {

WTADelivery::WTADelivery(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

void WTADelivery::initialize(char const *name, PVParams *params, Communicator const *comm) {
   BaseDelivery::initialize(name, params, comm);
}

void WTADelivery::setObjectType() { mObjectType = "WTADelivery"; }

void WTADelivery::ioParam_receiveGpu(enum ParamsIOFlag ioFlag) {
   // Never receive from gpu
   mReceiveGpu = false;
   if (ioFlag == PARAMS_IO_READ) {
      parameters()->handleUnnecessaryParameter(name, "receiveGpu", false /*correctValue*/);
   }
}

Response::Status
WTADelivery::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = BaseDelivery::communicateInitInfo(message);
   if (status != Response::SUCCESS) {
      return status;
   }

   auto *singleArbor = message->mObjectTable->findObject<SingleArbor>(getName());
   FatalIf(!singleArbor, "%s requires a SingleArbor component.\n", getDescription_c());
   if (!singleArbor->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }

   checkPreAndPostDimensions();
   return Response::SUCCESS;
}

void WTADelivery::checkPreAndPostDimensions() {
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
   if (postLoc->nf != 1) {
      ErrorLog().printf(
            "%s requires post nf be equal to 1 (observed value is %d).\n",
            getDescription_c(),
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
         "WTADelivery \"%s\" Error: %s and %s do not have the same dimensions.\n Dims: "
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

void WTADelivery::deliver(float *destBuffer) {
   if (mChannelCode == CHANNEL_NOUPDATE) {
      return;
   }

   PVLayerCube const preActivityCube = mPreData->getPublisher()->createCube(mDelay);
   PVLayerLoc const &preLoc          = preActivityCube.loc;

   int const nx       = preLoc.nx;
   int const ny       = preLoc.ny;
   int const nf       = preLoc.nf;
   PVHalo const &halo = preLoc.halo;
   int nxPreExtended  = nx + halo.lt + halo.rt;
   int nyPreExtended  = ny + halo.dn + halo.up;
   int numPreExtended = nxPreExtended * nyPreExtended * nf;
   pvAssert(numPreExtended * preLoc.nbatch == preActivityCube.numItems);
   int numPostRestricted = nx * ny * nf;

   float *postChannel = destBuffer;
   int const nbatch   = preLoc.nbatch;
   pvAssert(nbatch == mPostGSyn->getLayerLoc()->nbatch);
   for (int b = 0; b < nbatch; b++) {
      float const *preActivityBuffer = preActivityCube.data + b * numPreExtended;
      float *postGSynBuffer          = postChannel + b * numPostRestricted;
      // If preActivityCube.isSparse is true, we could use the list of activeIndices.
      // However, we have to make sure that two indices that correspond to the same location but
      // different features are collapsed properly, and that if all the nonzero values at a
      // location are negative but not all values at the location are nonzero, that the maximum is
      // zero, not the greatest nonzero value.
      // For now, at least, use the nonsparse method even if preActivityCube.isSparse is true.
      int const numLocations = nx * ny;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (int k = 0; k < numLocations; k++) {
         int const kPreExtended =
               kIndexExtended(k, nx, ny, 1, halo.lt, halo.rt, halo.dn, halo.up) * nf;
         float maxValue = -FLT_MAX;
         for (int f = 0; f < nf; f++) {
            float value = preActivityBuffer[kPreExtended + f];
            maxValue    = value > maxValue ? value : maxValue;
         }
         postGSynBuffer[k] += maxValue;
      }
   }
}

void WTADelivery::deliverUnitInput(float *recvBuffer) {
   const int numNeuronsPost = mPostGSyn->getBufferSizeAcrossBatch();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
   for (int k = 0; k < numNeuronsPost; k++) {
      recvBuffer[k] += 1.0f;
   }
}

bool WTADelivery::isAllInputReady() const {
   bool isReady;
   if (getChannelCode() == CHANNEL_NOUPDATE) {
      isReady = true;
   }
   else {
      isReady = mPreData->isExchangeFinished(mDelay);
   }
   return isReady;
}

} // end namespace PV
