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

Publisher::Publisher(Communicator * comm, int numItems, PVLayerLoc loc, int numLevels, bool isSparse)
{
   //size_t dataSize  = numItems * sizeof(float);
   size_t dataSize  = sizeof(float);

   this->mComm  = comm;

   cube.data = nullptr;
   cube.loc = loc;
   cube.numItems = numItems;

   const int numBuffers = loc.nbatch;

   // not really inplace but ok as is only used to deliver
   // to provide cube information for data from store
   cube.size = numBuffers * numItems * dataSize + sizeof(PVLayerCube);

   store = new DataStore(numBuffers, numItems, dataSize, numLevels, isSparse);

   //DONE: check for memory leak here, method flagged by valgrind
   this->neighborDatatypes = Communicator::newDatatypes(&loc);

   numRequests = 0;
   requests = (MPI_Request *) calloc((NUM_NEIGHBORHOOD-1) * loc.nbatch, sizeof(MPI_Request));
   pvAssert(requests);
}

Publisher::~Publisher()
{
   delete store;
   Communicator::freeDatatypes(neighborDatatypes); neighborDatatypes = nullptr;
   free(requests);
}


int Publisher::updateAllActiveIndices() {
   if(store->isSparse()) return calcAllActiveIndices(); else return PV_SUCCESS;
}

int Publisher::updateActiveIndices() {
   if(store->isSparse()) return calcActiveIndices(); else return PV_SUCCESS;
}

int Publisher::calcAllActiveIndices() {
   for(int l = 0; l < store->numberOfLevels(); l++){
      for(int b = 0; b < store->numberOfBuffers(); b++){
         //Active indicies stored as local ext values
         int numActive = 0;
         pvdata_t * activity = (pvdata_t*) store->buffer(b, l);;
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
   for(int b = 0; b < store->numberOfBuffers(); b++){
      //Active indicies stored as local ext values
      int numActive = 0;
      pvdata_t * activity = (pvdata_t*) store->buffer(b);;
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

int Publisher::publish(double currentTime, double lastUpdateTime, int neighbors[], int numNeighbors,
                       PVLayerCube* cube)
{
   //
   // Everyone publishes border region to neighbors even if no subscribers.
   // This means that everyone should wait as well.
   //

   size_t dataSize = cube->numItems * sizeof(pvdata_t);
   pvAssert(dataSize == (store->size() * store->numberOfBuffers()));

   pvdata_t * sendBuf = cube->data;
   pvdata_t * recvBuf = recvBuffer(0); //Grab all of the buffer, allocated continuously

   bool isSparse = store->isSparse();

   if (lastUpdateTime >= currentTime) {
      // copy entire layer and let neighbors overwrite
      //Only need to exchange borders if layer was updated this timestep
      memcpy(recvBuf, sendBuf, dataSize);
      exchangeBorders(neighbors, numNeighbors, &cube->loc, 0);
      store->setLastUpdateTime(LOCAL/*bufferId*/, lastUpdateTime);

      //Updating active indices is done after MPI wait in HyPerCol
      //to avoid race condition because exchangeBorders mpi is async
   }
   else if (store->numberOfLevels()>1){
      // If there are delays, copy last level's data to this level.
      // TODO: we could use pointer indirection to cut down on the number of memcpy calls required, if this turns out to be an expensive step
      memcpy(recvBuf, recvBuffer(LOCAL/*bufferId*/,1), dataSize);
      store->setLastUpdateTime(LOCAL/*bufferId*/, lastUpdateTime);
   }

   return PV_SUCCESS;
}

int Publisher::exchangeBorders(int neighbors[], int numNeighbors, const PVLayerLoc * loc, int delay/*default 0*/) {
   // Code duplication with Communicator::exchange.  Consolidate?
   PVHalo const * halo = &loc->halo;
   if (halo->lt==0 && halo->rt==0 && halo->dn==0 && halo->up==0) { return PV_SUCCESS; }
   int status = PV_SUCCESS;

#ifdef PV_USE_MPI
   //Using local ranks and communicators for border exchange
   int icRank = mComm->commRank();
   MPI_Comm mpiComm = mComm->communicator();

   //Loop through batches
   for(int b = 0; b < loc->nbatch; b++){
      // don't send interior
      pvAssert(numRequests == b * (mComm->numberOfNeighbors()-1));
      for (int n = 1; n < NUM_NEIGHBORHOOD; n++) {
         if (neighbors[n] == icRank) continue;  // don't send interior to self
         pvdata_t * recvBuf = recvBuffer(b, delay) + mComm->recvOffset(n, loc);
         // sendBuf = cube->data + Communicator::sendOffset(n, &cube->loc);
         pvdata_t * sendBuf = recvBuffer(b, delay) + mComm->sendOffset(n, loc);


#ifdef DEBUG_OUTPUT
         size_t recvOff = mComm->recvOffset(n, &cube.loc);
         size_t sendOff = mComm->sendOffset(n, &cube.loc);
         if( cube.loc.nb > 0 ) {
            pvInfo().printf("[%2d]: recv,send to %d, n=%d, delay=%d, recvOffset==%ld, sendOffset==%ld, numitems=%d, send[0]==%f\n", mComm->commRank(), neighbors[n], n, delay, recvOff, sendOff, cube.numItems, sendBuf[0]);
         }
         else {
            pvInfo().printf("[%2d]: recv,send to %d, n=%d, delay=%d, recvOffset==%ld, sendOffset==%ld, numitems=%d\n", mComm->commRank(), neighbors[n], n, delay, recvOff, sendOff, cube.numItems);
         }
         pvInfo().flush();
#endif //DEBUG_OUTPUT

         MPI_Irecv(recvBuf, 1, neighborDatatypes[n], neighbors[n], mComm->getReverseTag(n), mpiComm,
                   &requests[numRequests++]);
         int status = MPI_Send( sendBuf, 1, neighborDatatypes[n], neighbors[n], mComm->getTag(n), mpiComm);
         pvAssert(status==0);

      }
      pvAssert(numRequests == (b+1) * (mComm->numberOfNeighbors()-1));
   }

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

   if (numRequests != 0) {
      MPI_Waitall(numRequests, requests, MPI_STATUSES_IGNORE);
      numRequests = 0;
   }
#endif // PV_USE_MPI

   return 0;
}

} /* namespace PV */
