/*
 * InterColComm.cpp
 *
 *  Created on: Aug 28, 2008
 *      Author: rasmussn
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "InterColComm.hpp"
#include "HyPerCol.hpp"

namespace PV {

InterColComm::InterColComm(int* argc, char*** argv) : Communicator(argc, argv)
{
   numPublishers = 0;
   publisherArraySize = INITIAL_PUBLISHER_ARRAY_SIZE;
   publishers = (Publisher **) malloc( publisherArraySize * sizeof(Publisher *) );
   for (int i = 0; i < publisherArraySize; i++) {
      publishers[i] = NULL;
   }
}

InterColComm::~InterColComm()
{
   clearPublishers();
   free(publishers); publishers = NULL;
}

int InterColComm::addPublisher(HyPerLayer* pub, int numItems, int numLevels)
{
   int pubId = pub->getLayerId();
   if( pubId >= publisherArraySize) {
      int status = resizePublishersArray(pubId+1);
      assert(status == EXIT_SUCCESS);
   }

//#ifdef PV_USE_OPENCL
//   bool copydstoreflag=pub->getCopyDataStoreFlag();
//   publishers[pubId] = new Publisher(pubId, pub->getParent(), numItems, pub->getCLayer()->loc, numLevels, copydstoreflag);
//#else
   publishers[pubId] = new Publisher(pubId, pub->getParent(), numItems, pub->clayer->loc, numLevels);
//#endif
   numPublishers += 1;

   return pubId;
}

int InterColComm::clearPublishers() {
   for (int i=0; i<numPublishers; i++) {
      delete publishers[i]; publishers[i] = NULL;
   }
   numPublishers = 0;
   return PV_SUCCESS;
}

int InterColComm::resizePublishersArray(int newSize) {
   /* If newSize is greater than the existing size publisherArraySize,
    * create a new array of size newSize, and copy over the existing
    * publishers.  publisherArraySize is updated, to equal newSize.
    * If newSize <= publisherArraySize, do nothing
    * Returns PV_SUCCESS if resizing was successful
    * (or not needed; i.e. if newSize<=publisherArraySize)
    * Returns PV_FAILURE if unable to allocate a new array; in this
    * (unlikely) case, publishers and publisherArraySize are unchanged.
    */
   if( newSize > publisherArraySize ) {
      Publisher ** newPublishers = (Publisher **) malloc( newSize * sizeof(Publisher *) );
      if( newPublishers == NULL) return PV_FAILURE;
      for( int k=0; k< publisherArraySize; k++ ) {
         newPublishers[k] = publishers[k];
      }
      for( int k=publisherArraySize; k<newSize; k++) {
          newPublishers[k] = NULL;
      }
      free(publishers);
      publishers = newPublishers;
      publisherArraySize = newSize;
   }
   return PV_SUCCESS;
}

int InterColComm::subscribe(HyPerConn* conn)
{
   int pubId = conn->preSynapticLayer()->getLayerId();
   assert( pubId < publisherArraySize && pubId >= 0);
   return publishers[pubId]->subscribe(conn);
}

int InterColComm::publish(HyPerLayer* pub, PVLayerCube* cube)
{
   int pubId = pub->getLayerId();
   return publishers[pubId]->publish(pub, neighbors, numNeighbors, borders, numBorders, cube);
}

int InterColComm::exchangeBorders(int pubId, const PVLayerLoc * loc, int delay/*default=0*/) {
   int status = publishers[pubId]->exchangeBorders(neighbors, numNeighbors, loc, delay);
   return status;
}

/**
 * wait until all outstanding published messages have arrived
 */
int InterColComm::wait(int pubId)
{
   return publishers[pubId]->wait();
}

//#ifdef PV_USE_OPENCL
//Publisher::Publisher(int pubId, HyPerCol * hc, int numItems, PVLayerLoc loc, int numLevels, bool copydstoreflag)
//#else
Publisher::Publisher(int pubId, HyPerCol * hc, int numItems, PVLayerLoc loc, int numLevels)
//#endif
{
   size_t dataSize  = numItems * sizeof(float);

   this->pubId = pubId;
   this->comm  = hc->icCommunicator();
   this->numSubscribers = 0;

   cube.data = NULL;
   cube.loc = loc;
   cube.numItems = numItems;

   // not really inplace but ok as is only used to deliver
   // to provide cube information for data from store
   cube.size = dataSize + sizeof(PVLayerCube);

   const int numBuffers = 1;
//#ifdef PV_USE_OPENCL
//   store = new DataStore(hc, numBuffers, dataSize, numLevels, copydstoreflag);
//#else
   store = new DataStore(hc, numBuffers, dataSize, numLevels);
//#endif

   //DONE: check for memory leak here, method flagged by valgrind
   this->neighborDatatypes = Communicator::newDatatypes(&loc);

   this->subscriberArraySize = INITIAL_SUBSCRIBER_ARRAY_SIZE;
   this->connection = (HyPerConn **) malloc( subscriberArraySize * sizeof(HyPerConn *) );
   assert(this->connection);
   for (int i = 0; i < subscriberArraySize; i++) {
      this->connection[i] = NULL;
   }
   numRequests = 0;
}

Publisher::~Publisher()
{
   delete store;
   Communicator::freeDatatypes(neighborDatatypes); neighborDatatypes = NULL;
   free(connection);
}

int Publisher::publish(HyPerLayer* pub,
                       int neighbors[], int numNeighbors,
                       int borders[], int numBorders,
                       PVLayerCube* cube, int delay/*default=0*/)
{
   //
   // Everyone publishes border region to neighbors even if no subscribers.
   // This means that everyone should wait as well.
   //

   size_t dataSize = cube->numItems * sizeof(pvdata_t);
   assert(dataSize == store->size());

   pvdata_t * sendBuf = cube->data;
   pvdata_t * recvBuf = recvBuffer(LOCAL);  // only LOCAL buffer, neighbors copy into LOCAL extended buffer

   if (pub->getLastUpdateTime() >= pub->getParent()->simulationTime()) {
      // copy entire layer and let neighbors overwrite
      // TODO - have layers use the data store directly then no need for extra copy
      //Only memcopy if layer needs an update
      memcpy(recvBuf, sendBuf, dataSize);
      exchangeBorders(neighbors, numNeighbors, &cube->loc, 0);
   }

   return PV_SUCCESS;
}

int Publisher::exchangeBorders(int neighbors[], int numNeighbors, const PVLayerLoc * loc, int delay/*default 0*/) {
   // Code duplication with Communicator::exchange.  Consolidate?
   PVHalo const * halo = &loc->halo;
   if (halo->lt==0 && halo->rt==0 && halo->dn==0 && halo->up==0) { return PV_SUCCESS; }
   int status = PV_SUCCESS;

#ifdef PV_USE_MPI
   int icRank = comm->commRank();
   MPI_Comm mpiComm = comm->communicator();

   // don't send interior
   assert(numRequests == 0);
   for (int n = 1; n < NUM_NEIGHBORHOOD; n++) {
      if (neighbors[n] == icRank) continue;  // don't send interior to self
      pvdata_t * recvBuf = recvBuffer(LOCAL, delay) + comm->recvOffset(n, loc);
      // sendBuf = cube->data + Communicator::sendOffset(n, &cube->loc);
      pvdata_t * sendBuf = recvBuffer(LOCAL, delay) + comm->sendOffset(n, loc);


#ifdef DEBUG_OUTPUT
      size_t recvOff = comm->recvOffset(n, &cube.loc);
      size_t sendOff = comm->sendOffset(n, &cube.loc);
      if( cube.loc.nb > 0 ) {
         fprintf(stderr, "[%2d]: recv,send to %d, n=%d, delay=%d, recvOffset==%ld, sendOffset==%ld, numitems=%d, send[0]==%f\n", comm->commRank(), neighbors[n], n, delay, recvOff, sendOff, cube.numItems, sendBuf[0]);
      }
      else {
         fprintf(stderr, "[%2d]: recv,send to %d, n=%d, delay=%d, recvOffset==%ld, sendOffset==%ld, numitems=%d\n", comm->commRank(), neighbors[n], n, delay, recvOff, sendOff, cube.numItems);
      }
      fflush(stdout);
#endif //DEBUG_OUTPUT
      MPI_Irecv(recvBuf, 1, neighborDatatypes[n], neighbors[n], comm->getReverseTag(n), mpiComm,
                &requests[numRequests++]);
      int status = MPI_Send( sendBuf, 1, neighborDatatypes[n], neighbors[n], comm->getTag(n), mpiComm);
      assert(status==0);

   }
   assert(numRequests == comm->numberOfNeighbors() - 1);
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
   fprintf(stderr, "[%2d]: waiting for data, num_requests==%d\n", comm->commRank(), numRemote); fflush(stdout);
# endif // DEBUG_OUTPUT

   if (numRequests != 0) {
      MPI_Waitall(numRequests, requests, MPI_STATUSES_IGNORE);
      numRequests = 0;
   }
#endif // PV_USE_MPI

   return 0;
}

int Publisher::readData(int delay) {
   if (delay > 0) {
      cube.data = recvBuffer(LOCAL, delay);
   }
   else {
      cube.data = recvBuffer(LOCAL);
   }
   return 0;
}

int Publisher::subscribe(HyPerConn* conn)
{
   assert(numSubscribers <= subscriberArraySize);
   if( numSubscribers == subscriberArraySize ) {
      subscriberArraySize += RESIZE_ARRAY_INCR;
      HyPerConn ** newConnection = (HyPerConn **) malloc( subscriberArraySize * sizeof(HyPerConn *) );
      assert(newConnection);
      for( int k=0; k<numSubscribers; k++ ) {
         newConnection[k] = connection[k];
      }
      free(connection);
      connection = newConnection;
   }
   connection[numSubscribers] = conn;
   numSubscribers += 1;

   return 0;
}

} // end namespace PV
