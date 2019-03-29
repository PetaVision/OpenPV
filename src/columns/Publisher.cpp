/*
 * Publisher.cpp
 *
 *  Created on: Jul 19, 2016
 *      Author: pschultz
 */

#include "Publisher.hpp"
#include "checkpointing/CheckpointEntryDataStore.hpp"
#include "include/pv_common.h"
#include "utils/PVAssert.hpp"

namespace PV {

Publisher::Publisher(
      MPIBlock const &mpiBlock,
      float const *data,
      PVLayerLoc const *loc,
      int numLevels,
      bool isSparse) {
   mLayerCube = (PVLayerCube *)malloc(sizeof(PVLayerCube));

   int const nxExt           = loc->nx + loc->halo.lt + loc->halo.rt;
   int const nyExt           = loc->ny + loc->halo.dn + loc->halo.up;
   int const numItemsInBatch = nxExt * nyExt * loc->nf; // number of items in one batch element.

   mLayerCube->numItems      = numItemsInBatch * loc->nbatch; // number of items across the batch.
   mLayerCube->data          = data;
   mLayerCube->loc           = *loc;
   mLayerCube->isSparse      = isSparse;
   mLayerCube->numActive     = nullptr;
   mLayerCube->activeIndices = nullptr;

   int const numBuffers = loc->nbatch; // DataStore buffers correspond to batch elements

   store = new DataStore(numBuffers, numItemsInBatch, numLevels, isSparse);

   mBorderExchanger = new BorderExchange(mpiBlock, *loc);

   mpiRequestsBuffer = new RingBuffer<std::vector<MPI_Request>>(numLevels, 1);
   for (int l = 0; l < numLevels; l++) {
      auto *v = mpiRequestsBuffer->getBuffer(l, 0);
      v->clear();
      v->reserve((NUM_NEIGHBORHOOD - 1) * numBuffers);
   }
}

Publisher::~Publisher() {
   for (int l = 0; l < mpiRequestsBuffer->getNumLevels(); l++) {
      wait(l);
   }
   delete mpiRequestsBuffer;
   delete store;
   delete mBorderExchanger;
}

void Publisher::checkpointDataStore(
      Checkpointer *checkpointer,
      char const *objectName,
      char const *bufferName) {
   bool registerSucceeded = checkpointer->registerCheckpointEntry(
         std::make_shared<CheckpointEntryDataStore>(
               objectName, bufferName, checkpointer->getMPIBlock(), store, &mLayerCube->loc),
         false /*not constant*/);
   FatalIf(
         !registerSucceeded,
         "%s failed to register %s for checkpointing.\n",
         objectName,
         bufferName);
}

void Publisher::updateAllActiveIndices() {
   if (store->isSparse()) {
      for (int l = 0; l < store->getNumLevels(); l++) {
         updateActiveIndices(l);
      }
   }
}

PVLayerCube Publisher::createCube(int delay) {
   wait(delay);
   return store->createCube(mLayerCube->loc, delay);
}

void Publisher::updateActiveIndices(int delay) {
   if (store->isSparse()) {
      for (int b = 0; b < store->getNumBuffers(); b++) {
         // Active indicies stored as local extended values
         if (*store->numActiveBuffer(b, delay) < 0L) {
            store->updateActiveIndices(b, delay);
         }
         pvAssert(*store->numActiveBuffer(b, delay) >= 0L);
      }
   }
}

int Publisher::publish(double lastUpdateTime) {
   //
   // Everyone publishes border region to neighbors even if no subscribers.
   // This means that everyone should wait as well.
   //

   size_t dataSize = mLayerCube->numItems * sizeof(float);

   float const *sendBuf = mLayerCube->data;
   float *recvBuf       = recvBuffer(0); // Grab all of the buffer, allocated continuously

   memcpy(recvBuf, sendBuf, dataSize);
   exchangeBorders(&mLayerCube->loc, 0);
   store->setLastUpdateTime(0 /*bufferId*/, lastUpdateTime);

   for (int b = 0; b < store->getNumBuffers(); b++) {
      store->markActiveIndicesOutOfSync(b, 0);
   }
   // Updating active indices is done after MPI wait in HyPerCol
   // to avoid race condition because exchangeBorders mpi is async

   return PV_SUCCESS;
}

void Publisher::copyForward(double lastUpdateTime) {
   if (store->getNumLevels() > 1) {
      float *recvBuf  = recvBuffer(0); // Grab all of the buffer, allocated continuously
      size_t dataSize = mLayerCube->numItems * sizeof(float);
      memcpy(recvBuf, recvBuffer(0 /*bufferId*/, 1), dataSize);
      store->setLastUpdateTime(0 /*bufferId*/, lastUpdateTime);
      updateActiveIndices(0); // alternately, could copy active indices forward as well.
   }
}

int Publisher::exchangeBorders(const PVLayerLoc *loc, int delay /*default 0*/) {
   PVHalo const *halo = &loc->halo;
   if (halo->lt == 0 && halo->rt == 0 && halo->dn == 0 && halo->up == 0) {
      return PV_SUCCESS;
   }
   int status = PV_SUCCESS;

#ifdef PV_USE_MPI
   auto *requestsVector = mpiRequestsBuffer->getBuffer(delay, 0);
   pvAssert(requestsVector->empty());

   // Loop through batch.
   // The loop over batch elements probably belongs inside
   // BorderExchange::exchange(), but for this to happen, exchange() would need
   // to know how its data argument is organized with respect to batching.
   int exchangeVectorSize = 2 * (mBorderExchanger->getNumNeighbors() - 1);
   for (int b = 0; b < loc->nbatch; b++) {
      // don't send interior
      pvAssert(requestsVector->size() == (std::size_t)(b * exchangeVectorSize));

      float *data = recvBuffer(b, delay);
      std::vector<MPI_Request> batchElementMPIRequest{};
      mBorderExchanger->exchange(data, batchElementMPIRequest);
      pvAssert(batchElementMPIRequest.size() == (std::size_t)exchangeVectorSize);
      requestsVector->insert(
            requestsVector->end(), batchElementMPIRequest.begin(), batchElementMPIRequest.end());
      pvAssert(requestsVector->size() == (std::size_t)((b + 1) * exchangeVectorSize));
   }

#endif // PV_USE_MPI

   return status;
}

int Publisher::isExchangeFinished(int delay /* default 0*/) {
   bool isReady;
   auto *requestsVector = mpiRequestsBuffer->getBuffer(delay, 0);
   if (requestsVector->empty()) {
      isReady = true;
   }
   else {
      int test;
      MPI_Testall((int)requestsVector->size(), requestsVector->data(), &test, MPI_STATUSES_IGNORE);
      if (test) {
         requestsVector->clear();
         updateActiveIndices(delay);
      }
      isReady = (bool)test;
   }
   return isReady;
}

/**
 * wait until all outstanding published messages have arrived
 */
int Publisher::wait(int delay /*default 0*/) {
#ifdef DEBUG_OUTPUT
   InfoLog().printf(
         "[%2d]: waiting for data, num_requests==%d\n", mBorderExchanger->getRank(), numRemote);
   InfoLog().flush();
#endif // DEBUG_OUTPUT

   auto *requestsVector = mpiRequestsBuffer->getBuffer(delay, 0);
   if (!requestsVector->empty()) {
      mBorderExchanger->wait(*requestsVector);
      pvAssert(requestsVector->empty());
   }
   updateActiveIndices(delay);

   return 0;
}

void Publisher::increaseTimeLevel() {
   wait(mpiRequestsBuffer->getNumLevels() - 1);
   mpiRequestsBuffer->newLevel();
   store->newLevelIndex();
}

} /* namespace PV */
