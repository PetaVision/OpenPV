/*
 * InterColComm.cpp
 *
 *  Created on: Aug 28, 2008
 *      Author: rasmussn
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "InterColComm.hpp"
#include "HyPerCol.hpp"
#include "../connections/PVConnection.h"

namespace PV {

InterColComm::InterColComm(int* argc, char*** argv)
{
   float r;

   commInit(argc, argv);

   r = sqrt(worldSize);
   numRows = (int) r;
   numCols = (int) worldSize / numRows;
   numHyPerCols = numRows * numCols;
   assert (numHyPerCols <= worldSize);

   int exclsize = worldSize - numHyPerCols;

   if (exclsize == 0) {
      MPI_Comm_dup(MPI_COMM_WORLD, &icComm);
   }
   else {
      MPI_Group worldGroup, newGroup;
      MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);

      int * ranks = (int *) malloc(exclsize * sizeof(int));
      assert(ranks != NULL);

      for (int i = 0; i < exclsize; i++) {
         ranks[i] = i + numHyPerCols;
      }

      MPI_Group_excl(worldGroup, exclsize, ranks, &newGroup);
      MPI_Comm_create(MPI_COMM_WORLD, newGroup, &icComm);

#ifdef DEBUG_OUTPUT
      printf("[%2d]: Formed resized communicator, size=%d cols=%d rows=%d\n", icRank, icSize, numCols, numRows);
#endif

      free(ranks);
   }

   if (worldRank < numHyPerCols) {
      MPI_Comm_rank(icComm, &icRank);
      MPI_Comm_size(icComm, &icSize);
   }
   else {
      icSize = 0;
      icRank = -worldRank;
   }

   if (icSize > 0) {
      neighborInit();

      numPublishers = 0;

      for (int i = 0; i < MAX_PUBLISHERS; i++) {
         publishers[i] = NULL;
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);
}

InterColComm::~InterColComm()
{
   MPI_Barrier(MPI_COMM_WORLD);

   commFinalize();

   if (icSize > 0) {
      for (int i = 0; i < MAX_PUBLISHERS; i++) {
         if (publishers[i] != NULL) {
            delete publishers[i];
         }
      }
   }
}

int InterColComm::commInit(int* argc, char*** argv)
{
   MPI_Init(argc, argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
   MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

#ifdef DEBUG_OUTPUT
   printf("[%d]: InterColComm::commInit: size=%d\n", icRank, icSize);  fflush(stdout);
#endif

   return 0;
}

int InterColComm::commFinalize()
{
   MPI_Finalize();
   return 0;
}

int InterColComm::addPublisher(HyPerLayer* pub, size_t size1, size_t size2, int numLevels)
{
   int pubId = pub->getLayerId();
   publishers[pubId] = new Publisher(pubId, numNeighbors, size1, numBorders, size2, numLevels);
   numPublishers += 1;

   DataStore* store = publishers[pubId]->dataStore();

   for (int i = 0; i < numBorders; i++) {
      for (int delay = 0; delay < numLevels; delay++) {
         int borderIndex = Publisher::borderStoreIndex(i, numNeighbors);
         PVLayerCube* border = (PVLayerCube*) store->buffer(borderIndex, delay);
         pvcube_setAddr(border);
         pub->initBorder(border, borders[i]);
      }
   }

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
   return publishers[pubId]->publish(pub, neighbors, numNeighbors, borders, numBorders, cube);
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

/**
 * Initialize the communication neighbors
 */
int InterColComm::neighborInit()
{
   int num_neighbors = 0;
   int num_borders   = 0;

   // initialize neighbor and border lists
   // (local borders and remote neighbors form the complete neighborhood)

   this->numNeighbors = numberNeighbors();
   this->numBorders   = 1 + MAX_NEIGHBORS - this->numNeighbors;

   for (int i = 0; i < 1 + MAX_NEIGHBORS; i++) {
      neighbors[i] = 0;
      int n = neighborIndex(icRank, i);
      if (n >= 0) {
         neighbors[num_neighbors++] = n;
#ifdef DEBUG_OUTPUT
         printf("[%d]: neighborInit: neighbor[%d] of %d is %d, i = %d\n",
                icRank, num_neighbors - 1, this->numNeighbors, n, i);
         fflush(stdout);
#endif
      } else {
         borders[num_borders++] = -n;
      }
   }
   assert(this->numNeighbors == num_neighbors);
   assert(this->numBorders   == num_borders);

   return 0;
}

/**
 * Returns the communication row id for the given communication id
 */
int InterColComm::commRow(int commId)
{
   return (int) commId / this->numCols;
}

/**
 * Returns the communication column id for the given communication id
 */
int InterColComm::commColumn(int commId)
{
   return (commId - this->numCols * commRow(commId));
}

/**
 * Returns true if the given commId has a western neighbor
 * (false otherwise)
 */
bool InterColComm::hasWesternNeighbor(int commId)
{
   return commId % this->numHyPerCols;
}

/**
 * Returns true if the given commId has an eastern neighbor
 * (false otherwise)
 */
bool InterColComm::hasEasternNeighbor(int commId)
{
   return (commId + 1) % this->numHyPerCols;
}

/**
 * Returns true if the given commId has a northern neighbor
 * (false otherwise)
 */
bool InterColComm::hasNorthernNeighbor(int commId)
{
   return ((commId + this->numHyPerCols) > (this->icSize - 1)) ? 0 : 1;
}

/**
 * Returns true if the given commId has a southern neighbor
 * (false otherwise)
 */
bool InterColComm::hasSouthernNeighbor(int commId)
{
   return ((commId - this->numHyPerCols) < 0) ? 0 : 1;
}

/**
 * Returns the number in communication neighborhood (local included)
 */
int InterColComm::numberNeighbors()
{
   int n = 1;

   int hasWest = hasWesternNeighbor(icRank);
   int hasEast = hasEasternNeighbor(icRank);
   int hasNorth = hasNorthernNeighbor(icRank);
   int hasSouth = hasSouthernNeighbor(icRank);

   if (hasNorth > 0) n += 1;
   if (hasSouth > 0) n += 1;

   if (hasWest > 0) {
      n += 1;
      if (hasNorth > 0) n += 1;
      if (hasSouth > 0) n += 1;
   }

   if (hasEast > 0) {
      n += 1;
      if (hasNorth > 0) n += 1;
      if (hasSouth > 0) n += 1;
   }

   return n;
}

/**
 * Returns the communication id of the northwestern HyperColumn
 */
int InterColComm::northwest(int commId)
{
   if (hasNorthernNeighbor(commId) == 0) return -NORTHWEST;
   return west(commId + this->numHyPerCols);
}

/**
 * Returns the communication id of the northern HyperColumn
 */
int InterColComm::north(int commId)
{
   if (hasNorthernNeighbor(commId) == 0) return -NORTH;
   return (commId + this->numHyPerCols);
}

/**
 * Returns the communication id of the northeastern HyperColumn
 */
int InterColComm::northeast(int commId)
{
   if (hasNorthernNeighbor(commId) == 0) return -NORTHEAST;
   return east(commId + this->numHyPerCols);
}

/**
 * Returns the communication id of the western HyperColumn
 */
int InterColComm::west(int commId)
{
   if (hasWesternNeighbor(commId) == 0) return -WEST;
   return (commRow(commId) * numHyPerCols + ((commId - 1) % numHyPerCols));
}

/**
 * Returns the communication id of the eastern HyperColumn
 */
int InterColComm::east(int commId)
{
   if (hasEasternNeighbor(commId) == 0) return -EAST;
   return (commRow(commId) * numHyPerCols + ((commId + 1) % numHyPerCols));
}

/**
 * Returns the communication id of the southwestern HyperColumn
 */
int InterColComm::southwest(int commId)
{
   if (hasSouthernNeighbor(commId) == 0) return -SOUTHWEST;
   return west(commId - this->numHyPerCols);
}

/**
 * Returns the communication id of the southern HyperColumn
 */
int InterColComm::south(int commId)
{
   if (hasSouthernNeighbor(commId) == 0) return -SOUTH;
   return (commId - this->numHyPerCols);
}

/**
 * Returns the communication id of the southeastern HyperColumn
 */
int InterColComm::southeast(int commId)
{
   if (hasSouthernNeighbor(commId) == 0) return -SOUTHEAST;
   return east(commId - this->numHyPerCols);
}

/**
 * Returns the sender rank for the given connection index
 */
int InterColComm::neighborIndex(int commId, int index)
{
   switch (index) {
   case LOCAL: /* local */
      return commId;
   case NORTHWEST : /* northwest */
      return northwest(commId);
   case NORTH     : /* north */
      return north(commId);
   case NORTHEAST : /* northeast */
      return northeast(commId);
   case WEST      : /* west */
      return west(commId);
   case EAST      : /* east */
      return east(commId);
   case SOUTHWEST : /* southwest */
      return southwest(commId);
   case SOUTH     : /* south */
      return south(commId);
   case SOUTHEAST : /* southeast */
      return southeast(commId);
   default:
      fprintf(stderr, "ERROR:neighborIndex: bad index\n");
   }
   return -1;
}

Publisher::Publisher(int pubId, int numType1, size_t size1, int numType2, size_t size2, int numLevels)
{
   size_t maxSize = (size1 > size2) ? size1 : size2;
   this->pubId = pubId;
   this->numSubscribers = 0;
   this->store = new DataStore(numType1+numType2, maxSize, numLevels);
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

   // send/recv to/from neighbors
   for (int i = 0; i < numNeighbors; i++) {
      // Note - cube->data addr need not be correct as it will be wrong copied in from MPI
      void* recvBuf = recvBuffer(i);
      MPI_Irecv(recvBuf, size, MPI_CHAR, neighbors[i], pubId, MPI_COMM_WORLD,
            &request[i]);
      MPI_Send(cube, size, MPI_CHAR, neighbors[i], pubId, MPI_COMM_WORLD);
#ifdef DEBUG_OUTPUT
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      printf("[%d]: Publisher::publish: neighbor=%d pubId=%d sendbuf=%p recvbuf=%p\n",
             rank, neighbors[i], i, cube, recvBuf);
      fflush(stdout);
#endif
   }

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
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            printf("[%d]: Publisher::deliver: neighbor=%d buf=%p\n", rank, n, cube);
            fflush(stdout);
#endif
            conn->deliver(cube, n);
         }
      }
   }

   // deliver current (no delay) information last
   for (int n = 0; n < numNeighbors; n++) {
      int neighborId = n; /* WARNING - this must be initialized to n to work with PV_MPI */
      MPI_Waitany(numNeighbors, request, &neighborId, MPI_STATUS_IGNORE);

      for (int ic = 0; ic < numSubscribers; ic++) {
         HyPerConn* conn = this->connection[ic];
         if (conn->getDelay() == 0) {
            PVLayerCube* cube = (PVLayerCube*) store->buffer(neighborId, 0);
            pvcube_setAddr(cube);  // fix data address arriving from MPI
#ifdef DEBUG_OUTPUT
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            printf("[%d]: Publisher::deliver: neighbor=%d buf=%p\n", rank, neighborId, cube);
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
