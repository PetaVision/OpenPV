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
   for (int i = 0; i < numPublishers; i++) {
      if (publishers[i] != NULL) {
         delete publishers[i];
      }
   }
}

int InterColComm::addPublisher(HyPerLayer* pub, int numItems, int numLevels)
{
   int pubId = pub->getLayerId();
   if( pubId >= publisherArraySize) {
      int status = resizePublishersArray(pubId+1);
      assert(status == EXIT_SUCCESS);
   }

   publishers[pubId] = new Publisher(pubId, this, numItems, pub->clayer->loc, numLevels);
   numPublishers += 1;

   return pubId;
}

int InterColComm::resizePublishersArray(int newSize) {
   /* If newSize is greater than the existing size publisherArraySize,
    * create a new array of size newSize, and copy over the existing
    * publishers.  publisherArraySize is updated, to equal newSize.
    * If newSize <= publisherArraySize, do nothing
    * Returns EXIT_SUCCESS if resizing was successful
    * (or not needed; i.e. if newSize<=publisherArraySize)
    * Returns EXIT_FAILURE if unable to allocate a new array; in this
    * (unlikely) case, publishers and publisherArraySize are unchanged.
    */
   if( newSize > publisherArraySize ) {
      Publisher ** newPublishers = (Publisher **) malloc( newSize * sizeof(Publisher *) );
      if( newPublishers == NULL) return EXIT_FAILURE;
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
   return EXIT_SUCCESS;
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

/**
 * wait until all outstanding published messages have arrived
 */
int InterColComm::wait(int pubId)
{
   const int numRemote = numNeighbors - 1;  // numNeighbors includes LOCAL
   return publishers[pubId]->wait(numRemote);
}

/**
 * deliver all outstanding published messages
 */
int InterColComm::deliver(HyPerCol* hc, int pubId)
{
#ifdef DEBUG_OUTPUT
   printf("[%d]: InterColComm::deliver: pubId=%d\n", commRank(), pubId);  fflush(stdout);
#endif
   return publishers[pubId]->deliver(hc, numNeighbors, numBorders);
}

#ifdef OBSOLETE
// deprecated constructor that separates borders from the layer data structure
Publisher::Publisher(int pubId, int numType1, size_t size1, int numType2, size_t size2, int numLevels)
{
   size_t maxSize = (size1 > size2) ? size1 : size2;
   this->pubId = pubId;
   this->comm  = NULL;
   this->numSubscribers = 0;
   this->store = new DataStore(numType1+numType2, maxSize, numLevels);
   for (int i = 0; i < MAX_SUBSCRIBERS; i++) {
      this->connection[i] = NULL;
   }
}
#endif

Publisher::Publisher(int pubId, Communicator * comm, int numItems, PVLayerLoc loc, int numLevels)
{
   size_t dataSize  = numItems * sizeof(float);

   this->pubId = pubId;
   this->comm  = comm;
   this->numSubscribers = 0;

   cube.data = NULL;
   cube.loc = loc;
   cube.numItems = numItems;

   // not really inplace but ok as is only used to deliver
   // to provide cube information for data from store
   cube.size = dataSize + sizeof(PVLayerCube);

   const int numBuffers = 1;
   this->store = new DataStore(numBuffers, dataSize, numLevels);

   this->neighborDatatypes = Communicator::newDatatypes(&loc);

   this->subscriberArraySize = INITIAL_SUBSCRIBER_ARRAY_SIZE;
   this->connection = (HyPerConn **) malloc( subscriberArraySize * sizeof(HyPerConn *) );
   assert(this->connection);
   for (int i = 0; i < subscriberArraySize; i++) {
      this->connection[i] = NULL;
   }
}

Publisher::~Publisher()
{
   delete store;
}

int Publisher::publish(HyPerLayer* pub,
                       int neighbors[], int numNeighbors,
                       int borders[], int numBorders,
                       PVLayerCube* cube)
{
   //
   // Everyone publishes border region to neighbors even if no subscribers.
   // This means that everyone should wait as well.
   //

   size_t dataSize = cube->numItems * sizeof(pvdata_t);
   assert(dataSize == store->size());

   pvdata_t * sendBuf = cube->data;
   pvdata_t * recvBuf = recvBuffer(LOCAL);  // only LOCAL buffer, neighbors copy into LOCAL extended buffer

   // copy entire layer and let neighbors overwrite
   // TODO - have layers use the data store directly then no need for extra copy
   //
   memcpy(recvBuf, sendBuf, dataSize);

#ifdef PV_USE_MPI
   int icRank = comm->commRank();
   MPI_Comm mpiComm = comm->communicator();

   // don't send interior
   int nreq = 0;
   for (int n = 1; n < NUM_NEIGHBORHOOD; n++) {
      if (neighbors[n] == icRank) continue;  // don't send interior to self
      recvBuf = cube->data + Communicator::recvOffset(n, &cube->loc);
      sendBuf = cube->data + Communicator::sendOffset(n, &cube->loc);
#ifdef DEBUG_OUTPUT
      size_t recvOff = Communicator::recvOffset(n, &cube->loc);
      size_t sendOff = Communicator::recvOffset(n, &cube->loc);
      fprintf(stderr, "[%2d]: recv,send to %d, n=%d recvOffset==%ld sendOffset==%ld send[0]==%f\n", comm->commRank(), neighbors[n], n, recvOff, sendOff, sendBuf[0]); fflush(stdout);
#endif
      MPI_Irecv(recvBuf, 1, neighborDatatypes[n], neighbors[n], 33, mpiComm,
                &requests[nreq++]);
      MPI_Send( sendBuf, 1, neighborDatatypes[n], neighbors[n], 33, mpiComm);
   }
   assert(nreq == comm->numberOfNeighbors() - 1);

#endif // PV_USE_MPI

   return 0;
}

/**
 * wait until all outstanding published messages have arrived
 */
int Publisher::wait(int numRemote)
{
   //
   // Everyone publishes border region to neighbors even if no subscribers.
   // This means that everyone should wait as well.
   //
#ifdef PV_USE_MPI
# ifdef DEBUG_OUTPUT
   fprintf(stderr, "[%2d]: waiting for data, num_requests==%d\n", comm->commRank(), numRemote); fflush(stdout);
# endif

   MPI_Waitall(numRemote, requests, MPI_STATUSES_IGNORE);
#endif // PV_USE_MPI

   return 0;
}

/**
 * deliver published messages
 */
int Publisher::deliver(HyPerCol* hc, int numNeighbors, int numBorders)
{
   //
   // Waiting for data to arrive has been separated from delivery.
   // This method now assumes that wait has already been called
   // and that the data have all arrived from remote neighbors.
   //

   for (int ic = 0; ic < numSubscribers; ic++) {
      HyPerConn* conn = connection[ic];
//      int delay = conn->getDelay(0);
//      if (delay > 0) {
//         cube.data = recvBuffer(LOCAL, delay);
//      }
//      else {
//         cube.data = recvBuffer(LOCAL);
//      }
#ifdef DEBUG_OUTPUT
      printf("[%d]: Publisher::deliver: buf=%p\n", comm->commRank(), cube.data);
      fflush(stdout);
#endif
      conn->deliver(this, &cube, LOCAL);
   }

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
