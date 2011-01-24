/*
 * Communicator.cpp
 */

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "Communicator.hpp"
#include "../utils/conversions.h"

namespace PV {

Communicator::Communicator(int* argc, char*** argv)
{
   float r;

   commInit(argc, argv);

   sprintf(commName, "[%2d]: ", icRank);

   r = sqrtf(worldSize);
   numRows = (int) r;
   numCols = (int) worldSize / numRows;

   int commSize = numRows * numCols;

#ifdef PV_USE_MPI
   int exclsize = worldSize - commSize;

   if (exclsize == 0) {
      MPI_Comm_dup(MPI_COMM_WORLD, &icComm);
   }
   else {
      MPI_Group worldGroup, newGroup;
      MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);

      int * ranks = new int [exclsize];

      for (int i = 0; i < exclsize; i++) {
         ranks[i] = i + commSize;
      }

      MPI_Group_excl(worldGroup, exclsize, ranks, &newGroup);
      MPI_Comm_create(MPI_COMM_WORLD, newGroup, &icComm);

      delete ranks;
   }
#endif // PV_USE_MPI

#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: Formed resized communicator, size==%d cols==%d rows==%d\n", icRank, icSize, numCols, numRows);
#endif

   // some ranks are excluded if they don't fit in the processor quilt
   if (worldRank < commSize) {
#ifdef PV_USE_MPI
      MPI_Comm_rank(icComm, &icRank);
      MPI_Comm_size(icComm, &icSize);
#else // PV_USE_MPI
      icRank = 0;
      icSize = 1;
#endif // PV_USE_MPI
   }
   else {
      icSize = 0;
      icRank = -worldRank;
   }

   commName[0] = '\0';
   if (icSize > 1) {
      sprintf(commName, "[%2d]: ", icRank);
   }

   if (icSize > 0) {
      neighborInit();
   }

#ifdef PV_USE_MPI
   MPI_Barrier(MPI_COMM_WORLD);
#endif // PV_USE_MPI
}

Communicator::~Communicator()
{
#ifdef PV_USE_MPI
   MPI_Barrier(MPI_COMM_WORLD);
#endif // PV_USE_MPI

   commFinalize();
}

int Communicator::commInit(int* argc, char*** argv)
{
#ifdef PV_USE_MPI
   MPI_Init(argc, argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
   MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
#else // PV_USE_MPI
   worldRank = 0;
   worldSize = 1;
#endif // PV_USE_MPI

#ifdef DEBUG_OUTPUT
   fprintf(stderr, "[%2d]: Communicator::commInit: world_size==%d\n", worldRank, worldSize);
#endif

   return 0;
}

int Communicator::commFinalize()
{
#ifdef PV_USE_MPI
   MPI_Finalize();
#endif // PV_USE_MPI
   return 0;
}

/**
 * Initialize the communication neighborhood
 */
int Communicator::neighborInit()
{
   int num_neighbors = 0;
   int num_borders   = 0;

   // initialize neighbor and border lists
   // (local borders and remote neighbors form the complete neighborhood)

   this->numNeighbors = numberOfNeighbors();
   this->numBorders   = NUM_NEIGHBORHOOD - this->numNeighbors;

   for (int i = 0; i < NUM_NEIGHBORHOOD; i++) {
      int n = neighborIndex(icRank, i);
      neighbors[i] = icRank;   // default neighbor is self
      remoteNeighbors[i] = 0;
      if (n >= 0) {
         neighbors[i] = n;
         remoteNeighbors[num_neighbors++] = n;
#ifdef DEBUG_OUTPUT
         fprintf(stderr, "[%2d]: neighborInit: remote[%d] of %d is %d, i=%d, neighbor=%d\n",
                icRank, num_neighbors - 1, this->numNeighbors, n, i, neighbors[i]);
#endif
      } else {
         borders[num_borders++] = -n;
#ifdef DEBUG_OUTPUT
         fprintf(stderr, "[%2d]: neighborInit: i=%d, neighbor=%d\n", icRank, i, neighbors[i]);
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
int Communicator::commRow(int commId)
{
   return (int) commId / this->numCols;
}

/**
 * Returns the communication column id for the given communication id
 */
int Communicator::commColumn(int commId)
{
   return (commId - this->numCols * commRow(commId));
}

/**
 * Returns true if the given neighbor is present
 * (false otherwise)
 */
bool Communicator::hasNeighbor(int neighbor)
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
bool Communicator::hasNorthwesternNeighbor(int commId)
{
   return (hasNorthernNeighbor(commId) && hasWesternNeighbor(commId));
}

/**
 * Returns true if the given commId has a northern neighbor
 * (false otherwise)
 */
bool Communicator::hasNorthernNeighbor(int commId)
{
   return ((commId - numCommColumns()) < 0) ? false : true;
}

/**
 * Returns true if the given commId has a northeastern neighbor
 * (false otherwise)
 */
bool Communicator::hasNortheasternNeighbor(int commId)
{
   return (hasNorthernNeighbor(commId) && hasEasternNeighbor(commId));
}

/**
 * Returns true if the given commId has a western neighbor
 * (false otherwise)
 */
bool Communicator::hasWesternNeighbor(int commId)
{
   return (commId % numCommColumns());
}

/**
 * Returns true if the given commId has an eastern neighbor
 * (false otherwise)
 */
bool Communicator::hasEasternNeighbor(int commId)
{
   return ((commId + 1) % numCommColumns());
}

/**
 * Returns true if the given commId has a southwestern neighbor
 * (false otherwise)
 */
bool Communicator::hasSouthwesternNeighbor(int commId)
{
   return (hasSouthernNeighbor(commId) && hasWesternNeighbor(commId));
}

/**
 * Returns true if the given commId has a southern neighbor
 * (false otherwise)
 */
bool Communicator::hasSouthernNeighbor(int commId)
{
   return ((commId + numCommColumns()) < icSize) ? true : false;
}

/**
 * Returns true if the given commId has a southeastern neighbor
 * (false otherwise)
 */
bool Communicator::hasSoutheasternNeighbor(int commId)
{
   return (hasSouthernNeighbor(commId) && hasEasternNeighbor(commId));
}

/**
 * Returns the number in communication neighborhood (local included)
 */
int Communicator::numberOfNeighbors()
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
int Communicator::northwest(int commId)
{
   if (not hasNorthwesternNeighbor(commId)) return -NORTHWEST;
   return (commId - 1 - numCommColumns());
}

/**
 * Returns the communication id of the northern HyperColumn
 */
int Communicator::north(int commId)
{
   if (not hasNorthernNeighbor(commId)) return -NORTH;
   return (commId - numCommColumns());
}

/**
 * Returns the communication id of the northeastern HyperColumn
 */
int Communicator::northeast(int commId)
{
   if (not hasNortheasternNeighbor(commId)) return -NORTHEAST;
   return (commId + 1 - numCommColumns());
}

/**
 * Returns the communication id of the western HyperColumn
 */
int Communicator::west(int commId)
{
   if (not hasWesternNeighbor(commId)) return -WEST;
   return (commId - 1);
}

/**
 * Returns the communication id of the eastern HyperColumn
 */
int Communicator::east(int commId)
{
   if (not hasEasternNeighbor(commId)) return -EAST;
   return (commId + 1);
}

/**
 * Returns the communication id of the southwestern HyperColumn
 */
int Communicator::southwest(int commId)
{
   if (not hasSouthwesternNeighbor(commId)) return -SOUTHWEST;
   return (commId - 1 + numCommColumns());
}

/**
 * Returns the communication id of the southern HyperColumn
 */
int Communicator::south(int commId)
{
   if (not hasSouthernNeighbor(commId)) return -SOUTH;
   return (commId + numCommColumns());
}

/**
 * Returns the communication id of the southeastern HyperColumn
 */
int Communicator::southeast(int commId)
{
   if (not hasSoutheasternNeighbor(commId)) return -SOUTHEAST;
   return (commId + 1 + numCommColumns());
}

/**
 * Returns the sender rank for the given connection index
 */
int Communicator::neighborIndex(int commId, int index)
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
size_t Communicator::recvOffset(int n, const PVLayerLoc * loc)
{
   const size_t nx = loc->nx;
   const size_t ny = loc->ny;
   const size_t nf = loc->nf;
   const size_t nxBorder = loc->nb;
   const size_t nyBorder = loc->nb;

   const size_t sx = strideXExtended(loc);
   const size_t sy = strideYExtended(loc);

   switch (n) {
   case LOCAL:
      return (sx*nxBorder         + sy * nyBorder);
   case NORTHWEST:
      return ((size_t) 0                         );
   case NORTH:
      return (sx*nxBorder                        );
   case NORTHEAST:
      return (sx*nxBorder + sx*nx                );
   case WEST:
      return (                      sy * nyBorder);
   case EAST:
      return (sx*nxBorder + sx*nx + sy * nyBorder);
   case SOUTHWEST:
      return (                    + sy * (nyBorder + ny));
   case SOUTH:
      return (sx*nxBorder         + sy * (nyBorder + ny));
   case SOUTHEAST:
      return (sx*nxBorder + sx*nx + sy * (nyBorder + ny));
   default:
      fprintf(stderr, "ERROR:recvOffset: bad neighbor index\n");
   }
   return 0;
}

/**
 * Returns the send data offset for the given neighbor
 *  - send from interior
 */
size_t Communicator::sendOffset(int n, const PVLayerLoc * loc)
{
   const size_t nx = loc->nx;
   const size_t ny = loc->ny;
   const size_t nf = loc->nf;
   const size_t nxBorder = loc->nb;
   const size_t nyBorder = loc->nb;

   const size_t sx = strideXExtended(loc);
   const size_t sy = strideYExtended(loc);

   switch (n) {
   case LOCAL:
      return (sx*nxBorder + sy * nyBorder);
   case NORTHWEST:
      return (sx*nxBorder + sy * nyBorder);
   case NORTH:
      return (sx*nxBorder + sy * nyBorder);
   case NORTHEAST:
      return (sx*nx       + sy * nyBorder);
   case WEST:
      return (sx*nxBorder + sy * nyBorder);
   case EAST:
      return (sx*nx       + sy * nyBorder);
   case SOUTHWEST:
      return (sx*nxBorder + sy * ny);
   case SOUTH:
      return (sx*nxBorder + sy * ny);
   case SOUTHEAST:
      return (sx*nx       + sy * ny);
   default:
      fprintf(stderr, "ERROR:sendOffset: bad neighbor index\n");
   }
   return 0;
}

/**
 * Create a set of data types for inter-neighbor communication
 *   - caller must delete the data-type array
 */
MPI_Datatype * Communicator::newDatatypes(const PVLayerLoc * loc)
{
#ifdef PV_USE_MPI
   int count, blocklength, stride;

   MPI_Datatype * comms = new MPI_Datatype [NUM_NEIGHBORHOOD];
   
   const int nxBorder = loc->nPad;
   const int nyBorder = loc->nPad;

   // TODO - is this numFeatures
   const int nf = loc->nBands;

   count       = loc->ny;
   blocklength = nf*loc->nx;
   stride      = nf*(2*nxBorder + loc->nx);

   /* local interior */
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[LOCAL]);
   MPI_Type_commit(&comms[LOCAL]);

   count = nyBorder;

   /* northwest */
   blocklength = nf*nxBorder;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[NORTHWEST]);
   MPI_Type_commit(&comms[NORTHWEST]);

   /* north */
   blocklength = nf*loc->nx;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[NORTH]);
   MPI_Type_commit(&comms[NORTH]);

   /* northeast */
   blocklength = nf*nxBorder;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[NORTHEAST]);
   MPI_Type_commit(&comms[NORTHEAST]);

   count       = loc->ny;
   blocklength = nf*nxBorder;

   /* west */
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[WEST]);
   MPI_Type_commit(&comms[WEST]);

   /* east */
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[EAST]);
   MPI_Type_commit(&comms[EAST]);

   count = nyBorder;

   /* southwest */
   blocklength = nf*nxBorder;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[SOUTHWEST]);
   MPI_Type_commit(&comms[SOUTHWEST]);

   /* south */
   blocklength = nf*loc->nx;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[SOUTH]);
   MPI_Type_commit(&comms[SOUTH]);

   /* southeast */
   blocklength = nf*nxBorder;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[SOUTHEAST]);
   MPI_Type_commit(&comms[SOUTHEAST]);

   return comms;
#else // PV_USE_MPI
   return NULL;
#endif // PV_USE_MPI
}

/**
 * Exchange data with neighbors
 *   - the data regions to be sent are described by the datatypes
 *   - do irecv first so there is a location for send data to be received
 */
int Communicator::exchange(pvdata_t * data,
                           const MPI_Datatype neighborDatatypes [],
                           const PVLayerLoc * loc)
{
#ifdef PV_USE_MPI

   // don't send interior
   int nreq = 0;
   for (int n = 1; n < NUM_NEIGHBORHOOD; n++) {
      if (neighbors[n] == icRank) continue;  // don't send interior/self
      pvdata_t * recvBuf = data + recvOffset(n, loc);
      pvdata_t * sendBuf = data + sendOffset(n, loc);
#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: recv,send to %d, n=%d recvOffset==%ld sendOffset==%ld send[0]==%f\n", icRank, neighbors[n], n, recvOffset(n,loc), sendOffset(n,loc), sendBuf[0]); fflush(stdout);
#endif
      MPI_Irecv(recvBuf, 1, neighborDatatypes[n], neighbors[n], 33, icComm,
                &requests[nreq++]);
      MPI_Send( sendBuf, 1, neighborDatatypes[n], neighbors[n], 33, icComm);
   }

   // don't recv interior
   int count = numberOfNeighbors() - 1;
#ifdef DEBUG_OUTPUT
   fprintf(stderr, "[%2d]: waiting for data, count==%d\n", icRank, count); fflush(stdout);
#endif
   MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);

#endif // PV_USE_MPI

   return 0;
}

} // end namespace PV
