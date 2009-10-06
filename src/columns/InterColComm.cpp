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
#include "../connections/PVConnection.h"

namespace PV {

InterColComm::InterColComm(int* argc, char*** argv) : Communicator(argc, argv)
{
   numPublishers = 0;

   for (int i = 0; i < MAX_PUBLISHERS; i++) {
      publishers[i] = NULL;
   }
}

InterColComm::~InterColComm()
{
   for (int i = 0; i < MAX_PUBLISHERS; i++) {
      if (publishers[i] != NULL) {
         delete publishers[i];
      }
   }
}

int InterColComm::addPublisher(HyPerLayer* pub, int numItems, int numLevels)
{
   int pubId = pub->getLayerId();

#ifdef EXTEND_BORDER_INDEX
   publishers[pubId] = new Publisher(pubId, this, numItems, pub->clayer->loc, numLevels);
#else
   publishers[pubId] = new Publisher(pubId, numNeighbors, size1, numBorders, size2, numLevels);
   publishers[pubId]->setCommunicator(communicator());
#endif

   numPublishers += 1;

   DataStore* store = publishers[pubId]->dataStore();

#ifndef EXTEND_BORDER_INDEX
   for (int i = 0; i < numBorders; i++) {
      for (int delay = 0; delay < numLevels; delay++) {
         int borderIndex = Publisher::borderStoreIndex(i, numNeighbors);
         PVLayerCube* border = (PVLayerCube*) store->buffer(borderIndex, delay);
         pvcube_setAddr(border);
         pub->initBorder(border, borders[i]);
      }
   }
#endif

   return pubId;
}

int InterColComm::subscribe(HyPerConn* conn)
{
   int pubId = conn->preSynapticLayer()->getLayerId();
   assert(pubId < MAX_PUBLISHERS && pubId >= 0);
   return publishers[pubId]->subscribe(conn);
}

int InterColComm::publish(HyPerLayer* pub, PVLayerCube* cube)
{
   int pubId = pub->getLayerId();
   return publishers[pubId]->publish(pub, remoteNeighbors, numNeighbors, borders, numBorders, cube);
}

/**
 * deliver all outstanding published messages
 */
int InterColComm::deliver(HyPerCol* hc, int pubId)
{
#ifdef DEBUG_OUTPUT
   printf("[%d]: InterColComm::deliver: pubId=%d\n", icRank, pubId);  fflush(stdout);
#endif
   return publishers[pubId]->deliver(hc, numNeighbors, numBorders);
}

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

Publisher::Publisher(int pubId, Communicator * comm, int numItems, LayerLoc loc, int numLevels)
{
   size_t dataSize  = numItems * sizeof(float);

   this->pubId = pubId;
   this->comm  = comm;
   this->numSubscribers = 0;

   cube.data = NULL;
   cube.loc = loc;
   cube.numItems = numItems;
   cube.size = dataSize + sizeof(PVLayerCube);  // not really inplace but ok

   const int numBuffers = 1;
   this->store = new DataStore(numBuffers, dataSize, numLevels);

   this->neighborDatatypes = Communicator::newDatatypes(&loc);

   for (int i = 0; i < MAX_SUBSCRIBERS; i++) {
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
   size_t dataSize = cube->numItems * sizeof(pvdata_t);
   assert(dataSize == store->size());

   if (numSubscribers < 1) {
      return 0;  // no one to deliver to
   }

#ifdef PV_USE_MPI
   int icRank = comm->commRank();
   MPI_Comm mpiComm = comm->communicator();

   // don't send interior
   int nreq = 0;
   for (int n = 1; n < NUM_NEIGHBORHOOD; n++) {
      if (neighbors[n] == icRank) continue;  // don't send interior to self
      pvdata_t * recvBuf = cube->data + Communicator::recvOffset(n, &cube->loc);
      pvdata_t * sendBuf = cube->data + Communicator::sendOffset(n, &cube->loc);
#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: recv,send to %d, n=%d recvOffset==%ld sendOffset==%ld send[0]==%f\n", icRank, neighbors[n], n, recvOffset(n,loc), sendOffset(n,loc), sendBuf[0]); fflush(stdout);
#endif
      MPI_Irecv(recvBuf, 1, neighborDatatypes[n], neighbors[n], 33, mpiComm,
                &requests[nreq++]);
      MPI_Send( sendBuf, 1, neighborDatatypes[n], neighbors[n], 33, mpiComm);
   }
   assert(nreq == comm->numberOfNeighbors() - 1);

#endif // PV_USE_MPI

   pvdata_t * sendBuf = cube->data + Communicator::sendOffset(LOCAL, &cube->loc);

   // copy layer interior (no margins)
   memcpy(recvBuffer(LOCAL), sendBuf, dataSize);

#ifdef PV_OLD_MPI
   // send/recv to/from neighbors
   for (int i = 0; i < numNeighbors; i++) {
      // Note - cube->data addr need not be correct as it will be wrong copied in from MPI
      void * recvBuf = recvBuffer(i);
      MPI_Irecv(recvBuf, size, MPI_CHAR, neighbors[i], pubId, comm,
            &request[i]);
      MPI_Send(cube, size, MPI_CHAR, neighbors[i], pubId, comm);
#ifdef DEBUG_OUTPUT
      printf("[%d]: Publisher::publish: neighbor=%d pubId=%d sendbuf=%p recvbuf=%p\n",
             icRank, neighbors[i], i, cube, recvBuf);
      fflush(stdout);
#endif
   }
#endif // PV_OLD_MPI

   //
   // transform cube (and copy) for boundary conditions of neighbor slots that
   // don't exist in processor topology (i.e., a hypercolumn at edge of image)
   //
//   for (int i = 0; i < numBorders; i++) {
//      int borderIndex = Publisher::borderStoreIndex(i, numNeighbors);
//      PVLayerCube* border = (PVLayerCube*) recvBuffer(borderIndex);
//      pub->copyToBorder(borders[i], cube, border);
//   }

   return 0;
}

/**
 * deliver all outstanding published messages
 */
int Publisher::deliver(HyPerCol* hc, int numNeighbors, int numBorders)
{
   if (numSubscribers < 1) {
      return 0;  // no one to deliver to
   }

   // deliver delayed information first
   for (int ic = 0; ic < numSubscribers; ic++) {
      HyPerConn* conn = connection[ic];
      int delay = conn->getDelay();
      if (delay > 0) {
         cube.data = (pvdata_t *) store->buffer(LOCAL, delay);
#ifdef DEBUG_OUTPUT
         printf("[%d]: Publisher::deliver: buf=%p\n", icRank, cube.data);
         fflush(stdout);
#endif
         conn->deliver(&cube, LOCAL);
      }
   }

   // deliver current (no delay) information last

#ifdef PV_USE_MPI
   MPI_Waitall(numNeighbors-1, requests, MPI_STATUSES_IGNORE);
#ifdef DEBUG_OUTPUT
   fprintf(stderr, "[%2d]: waiting for data, count==%d\n", icRank, count); fflush(stdout);
#endif
#endif // PV_USE_MPI

   for (int ic = 0; ic < numSubscribers; ic++) {
      HyPerConn* conn = this->connection[ic];
      if (conn->getDelay() == 0) {
         cube.data = (pvdata_t *) store->buffer(LOCAL);
#ifdef DEBUG_OUTPUT
         printf("[%d]: Publisher::deliver: buf=%p\n", icRank, cube.data);
         fflush(stdout);
#endif
         conn->deliver(&cube, LOCAL);
      }
   }

   return 0;
}

int Publisher::subscribe(HyPerConn* conn)
{
   connection[numSubscribers] = conn;
   numSubscribers += 1;

   return 0;
}

} // end namespace PV
