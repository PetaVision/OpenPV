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

int InterColComm::addPublisher(HyPerLayer* pub, size_t size1, size_t size2, int numLevels)
{
   int pubId = pub->getLayerId();

#ifdef EXTEND_BORDER_INDEX
   publishers[pubId] = new Publisher(pubId, this, pub->clayer->loc, numLevels);
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
   this->comm  = MPI_COMM_WORLD;
   this->numSubscribers = 0;
   this->store = new DataStore(numType1+numType2, maxSize, numLevels);
   for (int i = 0; i < MAX_SUBSCRIBERS; i++) {
      this->connection[i] = NULL;
   }
}

Publisher::Publisher(int pubId, Communicator * comm, LayerLoc loc, int numLevels)
{
   size_t size = (loc.nx + 2*loc.nPad) * (loc.ny + 2*loc.nPad);
   this->pubId = pubId;
   this->comm  = comm->communicator();
   this->numSubscribers = 0;
   this->store = new DataStore(1, size, numLevels);
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
   size_t size = cube->size;
   assert(size == store->size());

   if (numSubscribers < 1) {
      // no one to deliver to
      return 0;
   }

#ifdef PV_USE_MPI
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
#else // PV_USE_MPI
   memcpy(recvBuffer(0), cube, size);
#endif // PV_USE_MPI

   //
   // transform cube (and copy) for boundary conditions of neighbor slots that
   // don't exist in processor topology (i.e., a hypercolumn at edge of image)
   //
   for (int i = 0; i < numBorders; i++) {
      int borderIndex = Publisher::borderStoreIndex(i, numNeighbors);
      PVLayerCube* border = (PVLayerCube*) recvBuffer(borderIndex);
      pub->copyToBorder(borders[i], cube, border);
   }

   return 0;
}

/**
 * deliver all outstanding published messages
 */
int Publisher::deliver(HyPerCol* hc, int numNeighbors, int numBorders)
{
   if (numSubscribers < 1) {
      // no one to deliver to
      return 0;
   }

   // deliver delayed information first
   for (int ic = 0; ic < numSubscribers; ic++) {
      HyPerConn* conn = connection[ic];
      int delay = conn->getDelay();
      if (delay > 0) {
         for (int n = 0; n < numNeighbors; n++) {
            PVLayerCube* cube = (PVLayerCube*) store->buffer(n, delay);
            pvcube_setAddr(cube);  // fix data address arriving from MPI
#ifdef DEBUG_OUTPUT
            printf("[%d]: Publisher::deliver: neighbor=%d buf=%p\n", icRank, n, cube);
            fflush(stdout);
#endif
            conn->deliver(cube, n);
         }
      }
   }

   // deliver current (no delay) information last
   for (int n = 0; n < numNeighbors; n++) {
      int neighborId = n; /* WARNING - this must be initialized to n to work with PV_MPI */

#ifdef PV_USE_MPI
      MPI_Waitany(numNeighbors, request, &neighborId, MPI_STATUS_IGNORE);
#endif // PV_USE_MPI

      for (int ic = 0; ic < numSubscribers; ic++) {
         HyPerConn* conn = this->connection[ic];
         if (conn->getDelay() == 0) {
            PVLayerCube* cube = (PVLayerCube*) store->buffer(neighborId, 0);
            pvcube_setAddr(cube);  // fix data address arriving from MPI
#ifdef DEBUG_OUTPUT
            printf("[%d]: Publisher::deliver: neighbor=%d buf=%p\n", icRank, neighborId, cube);
            fflush(stdout);
#endif
            conn->deliver(cube, neighborId);
         }
      }
   }

   // deliver border regions
   for (int i = 0; i < numBorders; i++) {
      int borderIndex = Publisher::borderStoreIndex(i, numNeighbors);

      for (int ic = 0; ic < numSubscribers; ic++) {
         HyPerConn* conn = this->connection[ic];
         int delay = conn->getDelay();
         PVLayerCube* border = (PVLayerCube*) store->buffer(borderIndex, delay);
         pvcube_setAddr(border);  // fix data address arriving from MPI
         conn->deliver(border, borderIndex);
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
