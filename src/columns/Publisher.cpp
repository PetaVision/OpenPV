/*
 * Publisher.cpp
 *
 *  Created on: Jul 19, 2016
 *      Author: pschultz
 */

#include "Publisher.hpp"
#include "utils/PVAssert.hpp"
#include "include/pv_common.h"

namespace PV {

Publisher::Publisher(Communicator * comm, PVLayerCube * cube, int numLevels, bool isSparse)
{
   this->mLayerCube = cube;
   this->mComm  = comm;

   int const numBuffers = cube->loc.nbatch;
   int const numItems = cube->numItems/numBuffers; // number of items in one batch element.

   store = new DataStore(numBuffers, numItems, numLevels, isSparse);

   this->neighborDatatypes = Communicator::newDatatypes(&cube->loc);

   requests.clear();
   requests.reserve((NUM_NEIGHBORHOOD-1) * numBuffers);
}

Publisher::~Publisher()
{
   delete store;
   Communicator::freeDatatypes(neighborDatatypes); neighborDatatypes = nullptr;
}


int Publisher::updateAllActiveIndices() {
   if(store->isSparse()) return calcAllActiveIndices(); else return PV_SUCCESS;
}

int Publisher::updateActiveIndices() {
   if(store->isSparse()) return calcActiveIndices(); else return PV_SUCCESS;
}

int Publisher::calcAllActiveIndices() {
   for(int l = 0; l < store->getNumLevels(); l++){
      for(int b = 0; b < store->getNumBuffers(); b++){
         //Active indicies stored as local ext values
         int numActive = 0;
         pvdata_t * activity = store->buffer(b, l);;
         unsigned int * activeIndices = store->activeIndicesBuffer(b, l);
         long * numActiveBuf = store->numActiveBuffer(b, l);

         for (int kex = 0; kex < store->getNumItems(); kex++) {
            if (activity[kex] != 0.0) {
               activeIndices[numActive] = kex;
               numActive++;
            }
         }
         *numActiveBuf = numActive;
      }
   }

   return PV_SUCCESS;
}

int Publisher::calcActiveIndices() {
   for(int b = 0; b < store->getNumBuffers(); b++){
      //Active indicies stored as local ext values
      int numActive = 0;
      pvdata_t * activity = store->buffer(b);;
      unsigned int * activeIndices = store->activeIndicesBuffer(b);
      long * numActiveBuf = store->numActiveBuffer(b);
      for (int kex = 0; kex < store->getNumItems(); kex++) {
         if (activity[kex] != 0.0) {
            activeIndices[numActive] = kex;
            numActive++;
         }
      }
      *numActiveBuf = numActive;
   }

   return PV_SUCCESS;
}

int Publisher::publish(double currentTime, double lastUpdateTime)
{
   //
   // Everyone publishes border region to neighbors even if no subscribers.
   // This means that everyone should wait as well.
   //

   size_t dataSize = mLayerCube->numItems * sizeof(pvdata_t);

   pvdata_t const * sendBuf = mLayerCube->data;
   pvdata_t * recvBuf = recvBuffer(0); //Grab all of the buffer, allocated continuously

   if (lastUpdateTime >= currentTime) {
      // copy entire layer and let neighbors overwrite
      //Only need to exchange borders if layer was updated this timestep
      memcpy(recvBuf, sendBuf, dataSize);
      exchangeBorders(&mLayerCube->loc, 0);
      store->setLastUpdateTime(LOCAL/*bufferId*/, lastUpdateTime);

      //Updating active indices is done after MPI wait in HyPerCol
      //to avoid race condition because exchangeBorders mpi is async
   }
   else if (store->getNumLevels()>1){
      // If there are delays, copy last level's data to this level.
      // TODO: we could use pointer indirection to cut down on the number of memcpy calls required, if this turns out to be an expensive step
      memcpy(recvBuf, recvBuffer(LOCAL/*bufferId*/,1), dataSize);
      store->setLastUpdateTime(LOCAL/*bufferId*/, lastUpdateTime);
   }

   return PV_SUCCESS;
}

int Publisher::exchangeBorders(const PVLayerLoc * loc, int delay/*default 0*/) {
   PVHalo const * halo = &loc->halo;
   if (halo->lt==0 && halo->rt==0 && halo->dn==0 && halo->up==0) { return PV_SUCCESS; }
   int status = PV_SUCCESS;

#ifdef PV_USE_MPI
   pvAssert(requests.empty());
   //Using local ranks and communicators for border exchange
   int icRank = mComm->commRank();
   MPI_Comm mpiComm = mComm->communicator();

   //Loop through batch.
   //The loop over batch elements probably belongs inside
   //Communicator::exchange(), but for this to happen, exchange() would need
   //to know how its data argument is organized with respect to batching.
   for(int b = 0; b < loc->nbatch; b++){
      // don't send interior
      pvAssert(requests.size() == b * (mComm->numberOfNeighbors()-1));

      pvdata_t * data = recvBuffer(b, delay);
      std::vector<MPI_Request> batchElementMPIRequest{};
      mComm->exchange(data, neighborDatatypes, loc, batchElementMPIRequest);
      pvAssert(batchElementMPIRequest.size()==mComm->numberOfNeighbors()-1);
      requests.insert(requests.end(), batchElementMPIRequest.begin(), batchElementMPIRequest.end());
      pvAssert(requests.size() == (b+1) * (mComm->numberOfNeighbors()-1));
   }
   pvAssert(requests.size() == loc->nbatch * (mComm->numberOfNeighbors()-1));

#endif // PV_USE_MPI

   return status;
}

/**
 * wait until all outstanding published messages have arrived
 */
int Publisher::wait()
{
#ifdef PV_USE_MPI
# ifdef DEBUG_OUTPUT
   pvInfo().printf("[%2d]: waiting for data, num_requests==%d\n", mComm->commRank(), numRemote);
   pvInfo().flush();
# endif // DEBUG_OUTPUT

   if (!requests.empty()) {
      mComm->wait(requests);
   }
#endif // PV_USE_MPI

   return 0;
}

} /* namespace PV */
