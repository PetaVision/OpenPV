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

      int * ranks = new int [exclsize];

      for (int i = 0; i < exclsize; i++) {
         ranks[i] = i + numHyPerCols;
      }

      MPI_Group_excl(worldGroup, exclsize, ranks, &newGroup);
      MPI_Comm_create(MPI_COMM_WORLD, newGroup, &icComm);

#ifdef DEBUG_OUTPUT
      printf("[%2d]: Formed resized communicator, size=%d cols=%d rows=%d\n", icRank, icSize, numCols, numRows);
#endif

      delete ranks;
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
   publishers[pubId]->setCommunicator(icComm);
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

/**
 * Initialize the communication neighborhood
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
      int n = neighborIndex(icRank, i);
      neighbors[i] = icRank;   // default neighbor is self
      remoteNeighbors[i] = 0;
      if (n >= 0) {
         neighbors[i] = n;
         remoteNeighbors[num_neighbors++] = n;
#ifdef DEBUG_OUTPUT
         printf("[%d]: neighborInit: remote[%d] of %d is %d, i=%d, neighbor=%d\n",
                icRank, num_neighbors - 1, this->numNeighbors, n, i, neighbors[i]);
         fflush(stdout);
#endif
      } else {
         borders[num_borders++] = -n;
#ifdef DEBUG_OUTPUT
         printf("[%d]: neighborInit: i=%d, neighbor=%d\n", icRank, i, neighbors[i]);
         fflush(stdout);
#endif
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
 * Returns true if the given neighbor is present
 * (false otherwise)
 */
bool InterColComm::hasNeighbor(int neighbor)
{
   switch (neighbor) {
   case LOCAL: /* local */
      return true;
   case NORTHWEST : /* northwest */
      return hasNorthwesternNeighbor(icRank);
   case NORTH     : /* north */
      return hasNorthernNeighbor(icRank);
   case NORTHEAST : /* northeast */
      return hasNortheasternNeighbor(icRank);
   case WEST      : /* west */
      return hasWesternNeighbor(icRank);
   case EAST      : /* east */
      return hasEasternNeighbor(icRank);
   case SOUTHWEST : /* southwest */
      return hasSouthwesternNeighbor(icRank);
   case SOUTH     : /* south */
      return hasSouthernNeighbor(icRank);
   case SOUTHEAST : /* southeast */
      return hasSoutheasternNeighbor(icRank);
   default:
      fprintf(stderr, "ERROR:hasNeighbor: bad index\n");
   }
   return false;
}

/**
 * Returns true if the given commId has a northwestern neighbor
 * (false otherwise)
 */
bool InterColComm::hasNorthwesternNeighbor(int commId)
{
   return (hasNorthernNeighbor(commId) && hasWesternNeighbor(commId));
}

/**
 * Returns true if the given commId has a northern neighbor
 * (false otherwise)
 */
bool InterColComm::hasNorthernNeighbor(int commId)
{
   return ((commId - numCommColumns()) < 0) ? 0 : 1;
}

/**
 * Returns true if the given commId has a northeastern neighbor
 * (false otherwise)
 */
bool InterColComm::hasNortheasternNeighbor(int commId)
{
   return (hasNorthernNeighbor(commId) && hasEasternNeighbor(commId));
}

/**
 * Returns true if the given commId has a western neighbor
 * (false otherwise)
 */
bool InterColComm::hasWesternNeighbor(int commId)
{
   return commId % numCommColumns();
}

/**
 * Returns true if the given commId has an eastern neighbor
 * (false otherwise)
 */
bool InterColComm::hasEasternNeighbor(int commId)
{
   return (commId + 1) % numCommColumns();
}

/**
 * Returns true if the given commId has a southwestern neighbor
 * (false otherwise)
 */
bool InterColComm::hasSouthwesternNeighbor(int commId)
{
   return (hasSouthernNeighbor(commId) && hasWesternNeighbor(commId));
}

/**
 * Returns true if the given commId has a southern neighbor
 * (false otherwise)
 */
bool InterColComm::hasSouthernNeighbor(int commId)
{

}

/**
 * Returns true if the given commId has a southeastern neighbor
 * (false otherwise)
 */
bool InterColComm::hasSoutheasternNeighbor(int commId)
{
   return (hasSouthernNeighbor(commId) && hasEasternNeighbor(commId));
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
   int id = west(commId + numCommColumns());
   return (id < 0) ? -SOUTHWEST : id;
}

/**
 * Returns the communication id of the southern HyperColumn
 */
int InterColComm::south(int commId)
{
   if (hasSouthernNeighbor(commId) == 0) return -SOUTH;
   return (commId + numCommColumns());
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

/**
 * Returns the recv data offset for the given neighbor
 *  - recv into borders
 */
size_t InterColComm::recvOffset(int n, const PVLayerLoc * loc)
{
   const size_t nx = loc->nx;
   const size_t ny = loc->nx;
   const size_t nxBorder = loc->nxBorder;
   const size_t nyBorder = loc->nyBorder;

   const size_t sy = 2 * nxBorder + nx;

   switch (n) {
   case LOCAL:
      return (nxBorder      + sy * nyBorder);
   case NORTHWEST:
      return ((size_t) 0                   );
   case NORTH:
      return (nxBorder                     );
   case NORTHEAST:
      return (nxBorder + nx                );
   case WEST:
      return (                sy * nyBorder);
   case EAST:
      return (nxBorder + nx + sy * nyBorder);
   case SOUTHWEST:
      return (              + sy * (nyBorder + ny));
   case SOUTH:
      return (nxBorder      + sy * (nyBorder + ny));
   case SOUTHEAST:
      return (nxBorder + nx + sy * (nyBorder + ny));
   default:
      fprintf(stderr, "ERROR:recvOffset: bad neighbor index\n");
   }
   return 0;
}

/**
 * Returns the send data offset for the given neighbor
 *  - send from interior
 */
size_t InterColComm::sendOffset(int n, const PVLayerLoc * loc)
{
   const size_t nx = loc->nx;
   const size_t ny = loc->nx;
   const size_t nxBorder = loc->nxBorder;
   const size_t nyBorder = loc->nyBorder;

   const size_t sy = 2 * nxBorder + nx;

   switch (n) {
   case LOCAL:
      return (nxBorder + sy * nyBorder);
   case NORTHWEST:
      return (nxBorder + sy * nyBorder);
   case NORTH:
      return (nxBorder + sy * nyBorder);
   case NORTHEAST:
      return (nx       + sy * nyBorder);
   case WEST:
      return (nxBorder + sy * nyBorder);
   case EAST:
      return (nx       + sy * nyBorder);
   case SOUTHWEST:
      return (nxBorder + sy * ny);
   case SOUTH:
      return (nxBorder + sy * ny);
   case SOUTHEAST:
      return (nx       + sy * ny);
   default:
      fprintf(stderr, "ERROR:sendOffset: bad neighbor index\n");
   }
   return 0;
}

/**
 * Create a set of data types for inter-neighbor communication
 *   - caller must free the data-type array
 */
MPI_Datatype * InterColComm::newDatatypes(const PVLayerLoc * loc)
{
   int count, blocklength, stride;

   MPI_Datatype * comms = new MPI_Datatype [MAX_NEIGHBORS+1];

   count       = loc->ny;
   blocklength = loc->nx;
   stride      = 2*loc->nxBorder + loc->nx;

   /* local interior */
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[LOCAL]);
   MPI_Type_commit(&comms[LOCAL]);

   count = loc->nyBorder;

   /* northwest */
   blocklength = loc->nxBorder;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[NORTHWEST]);
   MPI_Type_commit(&comms[NORTHWEST]);

   /* north */
   blocklength = loc->nx;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[NORTH]);
   MPI_Type_commit(&comms[NORTH]);

   /* northeast */
   blocklength = loc->nxBorder;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[NORTHEAST]);
   MPI_Type_commit(&comms[NORTHEAST]);

   count       = loc->ny;
   blocklength = loc->nxBorder;

   /* west */
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[WEST]);
   MPI_Type_commit(&comms[WEST]);

   /* east */
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[EAST]);
   MPI_Type_commit(&comms[EAST]);

   count = loc->nyBorder;

   /* southwest */
   blocklength = loc->nxBorder;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[SOUTHWEST]);
   MPI_Type_commit(&comms[SOUTHWEST]);

   /* south */
   blocklength = loc->nx;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[SOUTH]);
   MPI_Type_commit(&comms[SOUTH]);

   /* southeast */
   blocklength = loc->nxBorder;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[SOUTHEAST]);
   MPI_Type_commit(&comms[SOUTHEAST]);

   return comms;
}

/**
 * Recv data from neighbors
 *   - wait for delivery as recv has already been posted
 *   - the data regions to be sent are described by the datatypes
 */
int InterColComm::recv(pvdata_t * data, const MPI_Datatype neighborDatatypes [],
                       const PVLayerLoc * loc)
{
   // don't recv interior
   int count = numberNeighbors() - 1;
   //   printf("[%d]: waiting for data, count==%d\n", icRank, count); fflush(stdout);
   MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);

   return 0;
}

/**
 * Send data to neighbors
 *   - the data regions to be sent are described by the datatypes
 *   - do irecv first so there is a location for send data to be received
 */
int InterColComm::send(pvdata_t * data, const MPI_Datatype neighborDatatypes [],
                       const PVLayerLoc * loc)
{
   // don't send interior
   int nreq = 0;
   for (int n = 1; n < MAX_NEIGHBORS+1; n++) {
      if (neighbors[n] == icRank) continue;  // don't send to self
      pvdata_t * recvBuf = data + recvOffset(n, loc);
      pvdata_t * sendBuf = data + sendOffset(n, loc);
      printf("[%d]: recv,send to %d, n=%d recvOffset==%d sendOffset==%d send[0]==%f\n", icRank, neighbors[n], n, recvOffset(n,loc), sendOffset(n,loc), sendBuf[0]); fflush(stdout);
      MPI_Irecv(recvBuf, 1, neighborDatatypes[n], neighbors[n], 33, icComm,
                &requests[nreq++]);
      MPI_Send( sendBuf, 1, neighborDatatypes[n], neighbors[n], 33, icComm);
   }

   return 0;
}

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
      MPI_Irecv(recvBuf, size, MPI_CHAR, neighbors[i], pubId, comm,
            &request[i]);
      MPI_Send(cube, size, MPI_CHAR, neighbors[i], pubId, comm);
#ifdef DEBUG_OUTPUT
      printf("[%d]: Publisher::publish: neighbor=%d pubId=%d sendbuf=%p recvbuf=%p\n",
             icRank, neighbors[i], i, cube, recvBuf);
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
      MPI_Waitany(numNeighbors, request, &neighborId, MPI_STATUS_IGNORE);

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
