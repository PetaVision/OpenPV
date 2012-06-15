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
#include "../io/io.h"

namespace PV {

Communicator::Communicator(int* argc, char*** argv)
{
   float r;

   commInit(argc, argv);

   sprintf(commName, "[%2d]: ", icRank);

   bool rowsDefined = pv_getopt_int(*argc,  *argv, "-rows", &numRows)==0;
   bool colsDefined = pv_getopt_int(*argc, *argv, "-columns", &numCols)==0;

   if( rowsDefined && !colsDefined ) {
      numCols = (int) worldSize / numRows;
   }
   if( !rowsDefined && colsDefined ) {
      numRows = (int) worldSize / numCols;
   }
   if( !rowsDefined  && !colsDefined ) {
      r = sqrtf(worldSize);
      numRows = (int) r;
      numCols = (int) worldSize / numRows;
   }

   int commSize = numRows * numCols;

#ifdef PV_USE_MPI
   int exclsize = worldSize - commSize;

   if( exclsize < 0 ) {
      fprintf(stderr, "Error: %d rows and %d columns specified but only %d processes are available.\n", numRows, numCols, worldSize);
      exit(EXIT_FAILURE);
   }
   else if (exclsize == 0) {
      MPI_Comm_dup(MPI_COMM_WORLD, &icComm);
   }
   else {
      fprintf(stderr, "Error: %d rows and %d columns specified but %d processes available.  Excess processes not yet supported.  Exiting.\n", numRows, numCols, worldSize);
      exit(EXIT_FAILURE);
// Currently, excess processes cause problems because all processes, whether in the icComm group or not, call all the MPI commands.
// The excluded processes should be prevented from calling commands in the communicator.  It isn't desirable to have the excess
// processes simply exit, because there may be additional HyPerColumn simulations to run.
/*
      MPI_Group worldGroup, newGroup;
      MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);

      int * ranks = new int [exclsize];

      for (int i = 0; i < exclsize; i++) {
         ranks[i] = i + commSize;
      }

      MPI_Group_excl(worldGroup, exclsize, ranks, &newGroup);
      MPI_Comm_create(MPI_COMM_WORLD, newGroup, &icComm);

      delete ranks;
 */
   }
#endif // PV_USE_MPI

#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: Formed resized communicator, size==%d cols==%d rows==%d\n", icRank, icSize, numCols, numRows);
#endif // DEBUG_OUTPUT

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
   // If MPI wasn't initialized, initialize it.
   // Remember if it was initialized on entry; the destructor will only finalize if the constructor init'ed.
   // This way, you can do several simulations sequentially by initializing MPI before creating
   // the first HyPerCol; after running the first simulation the MPI environment will still exist and you
   // can run the second simulation, etc.
   MPI_Initialized(&mpi_initialized_on_entry);
   if( !mpi_initialized_on_entry ) MPI_Init(argc, argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
   MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
#else // PV_USE_MPI
   worldRank = 0;
   worldSize = 1;
#endif // PV_USE_MPI

#ifdef DEBUG_OUTPUT
   fprintf(stderr, "[%2d]: Communicator::commInit: world_size==%d\n", worldRank, worldSize);
#endif // DEBUG_OUTPUT

   return 0;
}

int Communicator::commFinalize()
{
#ifdef PV_USE_MPI
   if( !mpi_initialized_on_entry ) MPI_Finalize();
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
   int tags[9] = {0, 33, 34, 33, 34, 34, 33, 34, 33};
   // Corners (NW, NE, SW, SE) have tag 33 and edges (N, W, E, S) have tag 34.
   // In the top row of processes in the hypercolumn, a process is both the
   // northeast and east neighbor of the process to its left.  The difference
   // in tags ensures that the MPI_Send/MPI_Irecv calls can be distinguished.

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
#endif // DEBUG_OUTPUT
      } else {
         borders[num_borders++] = -n;
#ifdef DEBUG_OUTPUT
         fprintf(stderr, "[%2d]: neighborInit: i=%d, neighbor=%d\n", icRank, i, neighbors[i]);
#endif // DEBUG_OUTPUT
      }
      this->tags[i] = tags[i];
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
   return rowFromRank(commId, numRows, numCols);
}

/**
 * Returns the communication column id for the given communication id
 */
int Communicator::commColumn(int commId)
{
   return columnFromRank(commId, numRows, numCols);
}

/**
 * Returns the communication id for a given row and column
 */
int Communicator::commIdFromRowColumn(int commRow, int commColumn) {
   return rankFromRowAndColumn(commRow, commColumn, numRows, numCols);
}

/**
 * Returns true if the given neighbor is present
 * (false otherwise)
 */
bool Communicator::hasNeighbor(int neighbor)
{
   int nbrIdx = neighborIndex(icRank, neighbor);
   return nbrIdx >= 0;

//   switch (neighbor) {
//   case LOCAL: /* local */
//      return true;
//   case NORTHWEST : /* northwest */
//      return hasNorthwesternNeighbor(commRow(), commColumn());
//   case NORTH     : /* north */
//      return hasNorthernNeighbor(commRow(), commColumn());
//   case NORTHEAST : /* northeast */
//      return hasNortheasternNeighbor(commRow(), commColumn());
//   case WEST      : /* west */
//      return hasWesternNeighbor(commRow(), commColumn());
//   case EAST      : /* east */
//      return hasEasternNeighbor(commRow(), commColumn());
//   case SOUTHWEST : /* southwest */
//      return hasSouthwesternNeighbor(commRow(), commColumn());
//   case SOUTH     : /* south */
//      return hasSouthernNeighbor(commRow(), commColumn());
//   case SOUTHEAST : /* southeast */
//      return hasSoutheasternNeighbor(commRow(), commColumn());
//   default:
//      fprintf(stderr, "ERROR:hasNeighbor: bad index\n");
//      return false;
//   }
}

/**
 * Returns true if the given commId has a northwestern neighbor
 * (false otherwise)
 */
bool Communicator::hasNorthwesternNeighbor(int row, int column)
{
   return (hasNorthernNeighbor(row, column) || hasWesternNeighbor(row, column));
}

/**
 * Returns true if the given commId has a northern neighbor
 * (false otherwise)
 */
bool Communicator::hasNorthernNeighbor(int row, int column)
{
   return row > 0;
}

/**
 * Returns true if the given commId has a northeastern neighbor
 * (false otherwise)
 */
bool Communicator::hasNortheasternNeighbor(int row, int column)
{
   return (hasNorthernNeighbor(row, column) || hasEasternNeighbor(row, column));
}

/**
 * Returns true if the given commId has a western neighbor
 * (false otherwise)
 */
bool Communicator::hasWesternNeighbor(int row, int column)
{
   return column > 0;
}

/**
 * Returns true if the given commId has an eastern neighbor
 * (false otherwise)
 */
bool Communicator::hasEasternNeighbor(int row, int column)
{
   return column < numCommColumns() - 1;
}

/**
 * Returns true if the given commId has a southwestern neighbor
 * (false otherwise)
 */
bool Communicator::hasSouthwesternNeighbor(int row, int column)
{
   return (hasSouthernNeighbor(row, column) || hasWesternNeighbor(row, column));
}

/**
 * Returns true if the given commId has a southern neighbor
 * (false otherwise)
 */
bool Communicator::hasSouthernNeighbor(int row, int column)
{
   return row < numCommRows() - 1;
}

/**
 * Returns true if the given commId has a southeastern neighbor
 * (false otherwise)
 */
bool Communicator::hasSoutheasternNeighbor(int row, int column)
{
   return (hasSouthernNeighbor(row, column) || hasEasternNeighbor(row, column));
}

/**
 * Returns the number in communication neighborhood (local included)
 */
int Communicator::numberOfNeighbors()
{
   int n = 1 +
         hasNorthwesternNeighbor(commRow(), commColumn()) +
         hasNorthernNeighbor(commRow(), commColumn()) +
         hasNortheasternNeighbor(commRow(), commColumn()) +
         hasWesternNeighbor(commRow(), commColumn()) +
         hasEasternNeighbor(commRow(), commColumn()) +
         hasSouthwesternNeighbor(commRow(), commColumn()) +
         hasSouthernNeighbor(commRow(), commColumn()) +
         hasSoutheasternNeighbor(commRow(), commColumn());

//   int hasWest = hasWesternNeighbor(commRow(), commColumn());
//   int hasEast = hasEasternNeighbor(commRow(), commColumn());
//   int hasNorth = hasNorthernNeighbor(commRow(), commColumn());
//   int hasSouth = hasSouthernNeighbor(commRow(), commColumn());
//
//   if (hasNorth > 0) n += 1;
//   if (hasSouth > 0) n += 1;
//
//   if (hasWest > 0) {
//      n += 1;
//      if (hasNorth > 0) n += 1;
//      if (hasSouth > 0) n += 1;
//   }
//
//   if (hasEast > 0) {
//      n += 1;
//      if (hasNorth > 0) n += 1;
//      if (hasSouth > 0) n += 1;
//   }

   return n;
}

/**
 * Returns the communication id of the northwestern HyperColumn
 */
int Communicator::northwest(int commRow, int commColumn)
{
   int nbr_id = -NORTHWEST;
   if( hasNorthwesternNeighbor(commRow, commColumn) ) {
      int nbr_row = commRow - (commRow > 0);
      int nbr_column = commColumn - (commColumn > 0);
      nbr_id = commIdFromRowColumn(nbr_row, nbr_column);
   }
   return nbr_id;
}

/**
 * Returns the communication id of the northern HyperColumn
 */
int Communicator::north(int commRow, int commColumn)
{
   int nbr_id = -NORTH;
   if( hasNorthernNeighbor(commRow, commColumn) ) {
      nbr_id = commIdFromRowColumn(commRow-1, commColumn);
   }
   return nbr_id;
}

/**
 * Returns the communication id of the northeastern HyperColumn
 */
int Communicator::northeast(int commRow, int commColumn)
{
   int nbr_id = -NORTHEAST;
   if( hasNortheasternNeighbor(commRow, commColumn) ) {
      int nbr_row = commRow - (commRow > 0);
      int nbr_column = commColumn + (commColumn < numCommColumns()-1);
      nbr_id = commIdFromRowColumn(nbr_row, nbr_column);
   }
   return nbr_id;
}

/**
 * Returns the communication id of the western HyperColumn
 */
int Communicator::west(int commRow, int commColumn)
{
   int nbr_id = -WEST;
   if( hasWesternNeighbor(commRow, commColumn) ) {
      nbr_id = commIdFromRowColumn(commRow, commColumn-1);
   }
   return nbr_id;
}

/**
 * Returns the communication id of the eastern HyperColumn
 */
int Communicator::east(int commRow, int commColumn)
{
   int nbr_id = -EAST;
   if( hasEasternNeighbor(commRow, commColumn) ) {
      nbr_id = commIdFromRowColumn(commRow, commColumn+1);
   }
   return nbr_id;
}

/**
 * Returns the communication id of the southwestern HyperColumn
 */
int Communicator::southwest(int commRow, int commColumn)
{
   int nbr_id = -SOUTHWEST;
   if( hasSouthwesternNeighbor( commRow, commColumn) ) {
      int nbr_row = commRow + (commRow < numCommRows()-1);
      int nbr_column = commColumn - (commColumn > 0);
      nbr_id = commIdFromRowColumn(nbr_row, nbr_column);
   }
   return nbr_id;
}

/**
 * Returns the communication id of the southern HyperColumn
 */
int Communicator::south(int commRow, int commColumn)
{
   int nbr_id = -SOUTH;
   if( hasSouthernNeighbor(commRow, commColumn) ) {
      nbr_id = commIdFromRowColumn(commRow+1, commColumn);
   }
   return nbr_id;
}

/**
 * Returns the communication id of the southeastern HyperColumn
 */
int Communicator::southeast(int commRow, int commColumn)
{
    int nbr_id = -SOUTHEAST;
    if( hasSoutheasternNeighbor( commRow, commColumn) ) {
       int nbr_row = commRow + (commRow < numCommRows()-1);
       int nbr_column = commColumn + (commColumn < numCommColumns()-1);
       nbr_id = commIdFromRowColumn(nbr_row, nbr_column);
    }
    return nbr_id;
}

/**
 * Returns the intercolumn rank of the neighbor in the given direction
 * If there is no neighbor, returns a negative value
 */
int Communicator::neighborIndex(int commId, int index)
{
   int row = commRow(commId);
   int column = commColumn(commId);
   switch (index) {
   case LOCAL: /* local */
      return commId;
   case NORTHWEST : /* northwest */
      return northwest(row, column);
   case NORTH     : /* north */
      return north(row, column);
   case NORTHEAST : /* northeast */
      return northeast(row, column);
   case WEST      : /* west */
      return west(row, column);
   case EAST      : /* east */
      return east(row, column);
   case SOUTHWEST : /* southwest */
      return southwest(row, column);
   case SOUTH     : /* south */
      return south(row, column);
   case SOUTHEAST : /* southeast */
      return southeast(row, column);
   default:
      fprintf(stderr, "ERROR:neighborIndex: bad index\n");
      return -1;
   }
}

/**
 * Returns the recv data offset for the given neighbor
 *  - recv into borders
 */
size_t Communicator::recvOffset(int n, const PVLayerLoc * loc)
{
   const size_t nx = loc->nx;
   const size_t ny = loc->ny;
   // const size_t nf = loc->nf;  // Unused variable commented out May 24, 2011
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
      return 0;
   }
}

/**
 * Returns the send data offset for the given neighbor
 *  - send from interior
 */
size_t Communicator::sendOffset(int n, const PVLayerLoc * loc)
{
   const size_t nx = loc->nx;
   const size_t ny = loc->ny;
   // const size_t nf = loc->nf;  // Unused variable commented out May 24, 2011
   const size_t nxBorder = loc->nb;
   const size_t nyBorder = loc->nb;

   const size_t sx = strideXExtended(loc);
   const size_t sy = strideYExtended(loc);

   bool has_north_nbr = hasNorthernNeighbor(commRow(), commColumn());
   bool has_west_nbr = hasWesternNeighbor(commRow(), commColumn());
   bool has_east_nbr = hasEasternNeighbor(commRow(), commColumn());
   bool has_south_nbr = hasSouthernNeighbor(commRow(), commColumn());

   switch (n) {
   case LOCAL:
      return (sx*nxBorder                      + sy*nyBorder);
   case NORTHWEST:
      return (sx*has_west_nbr*nxBorder         + sy*has_north_nbr*nyBorder);
   case NORTH:
      return (sx*nxBorder                      + sy*nyBorder);
   case NORTHEAST:
      return (sx*(nx + !has_east_nbr*nxBorder) + sy*has_north_nbr*nyBorder);
   case WEST:
      return (sx*nxBorder                      + sy*nyBorder);
   case EAST:
      return (sx*nx                            + sy*nyBorder);
   case SOUTHWEST:
      return (sx*has_west_nbr*nxBorder         + sy*(ny + !has_south_nbr*nyBorder));
   case SOUTH:
      return (sx*nxBorder                      + sy*ny);
   case SOUTHEAST:
      return (sx*(nx + !has_east_nbr*nxBorder) + sy*(ny + !has_south_nbr*nyBorder));
   default:
      fprintf(stderr, "ERROR:sendOffset: bad neighbor index\n");
      return 0;
   }
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
   
   const int nxBorder = loc->nb;
   const int nyBorder = loc->nb;

   // TODO - is this numFeatures
   const int nf = loc->nf;

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
#endif // DEBUG_OUTPUT
      MPI_Irecv(recvBuf, 1, neighborDatatypes[n], neighbors[n], tags[n], icComm,
                &requests[nreq++]);
      MPI_Send( sendBuf, 1, neighborDatatypes[n], neighbors[n], tags[n], icComm);
   }

   // don't recv interior
   int count = numberOfNeighbors() - 1;
#ifdef DEBUG_OUTPUT
   fprintf(stderr, "[%2d]: waiting for data, count==%d\n", icRank, count); fflush(stdout);
#endif // DEBUG_OUTPUT
   MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);

#endif // PV_USE_MPI

   return 0;
}

} // end namespace PV
