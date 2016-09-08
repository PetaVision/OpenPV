/*
 * Communicator.cpp
 */

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>

#include "Communicator.hpp"
#include "utils/PVLog.hpp"
#include "utils/conversions.h"
#include "io/io.hpp"

namespace PV {

int Communicator::gcd ( int a, int b ){
   int c;
   while ( a != 0 ) {
      c = a; a = b%a;  b = c;
   }
   return b;
}

Communicator::Communicator(PV_Arguments * argumentList)
{
   float r;

   int totalSize;
   localIcComm = NULL;
   globalIcComm = NULL;
   MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);
   MPI_Comm_size(MPI_COMM_WORLD, &totalSize);

   numRows = argumentList->getNumRows();
   numCols = argumentList->getNumColumns();
   batchWidth = argumentList->getBatchWidth();

   bool rowsDefined = numRows!=0;
   bool colsDefined = numCols!=0;
   bool batchDefined = batchWidth!=0;

   bool inferingDim = !rowsDefined || !colsDefined || !batchDefined;

   if(!batchDefined){
      batchWidth = 1;
   }

   int procsLeft = totalSize/batchWidth;
   if( rowsDefined && !colsDefined ) {
      numCols = (int) ceil(procsLeft / numRows);
   }
   if( !rowsDefined && colsDefined ) {
      numRows = (int) ceil(procsLeft / numCols);
   }
   if( !rowsDefined  && !colsDefined ) {
      r = sqrtf(procsLeft);
      numRows = (int) r;
      if(numRows == 0){
         pvError() << "Not enough processes left, error\n";
      }
      numCols = (int) ceil(procsLeft / numRows);
   }

   int commSize = batchWidth * numRows * numCols;

   //For debugging
   if(globalRank == 0){
      pvInfo() << "Running with batchWidth=" << batchWidth << ", numRows=" << numRows << ", and numCols=" << numCols << "\n";
   }

   if(commSize > totalSize){
      pvError() << "Total number of specified processes (" << commSize << ") must be bigger than the number of processes launched (" << totalSize << ")\n";
   }

#ifdef PV_USE_MPI
   //Create a new split of useful mpi processes vs extras
   isExtra = globalRank >= commSize ? 1 : 0;
   MPI_Comm_split(MPI_COMM_WORLD, isExtra, globalRank % commSize, &globalIcComm);
   if(isExtra){
      pvWarn() << "Global process rank " << globalRank << " is extra, as only " << commSize << " mpiProcesses are required. Process exiting\n";
      return;
   }
   //Grab globalSize now that extra processes have been exited
   MPI_Comm_size(globalIcComm, &globalSize);


   //globalIcComm is now a communicator with only useful mpi processes
   
   //Calculate the batch idx from global rank 
   int batchColIdx = commBatch(globalRank);
   //Set local rank
   localRank = globalToLocalRank(globalRank, batchWidth, numRows, numCols);
   //Make new local communicator
   MPI_Comm_split(globalIcComm, batchColIdx, localRank, &localIcComm);
#else // PV_USE_MPI
   isExtra = 0;
   globalIcComm = MPI_COMM_WORLD;
   globalSize = 1;
   localRank = 0;
   localIcComm = MPI_COMM_WORLD;
#endif // PV_USE_MPI

//#ifdef DEBUG_OUTPUT
//      pvDebug().printf("[%2d]: Formed resized communicator, size==%d cols==%d rows==%d\n", icRank, icSize, numCols, numRows);
//#endif // DEBUG_OUTPUT

//Grab local rank and check for errors
   int tmpLocalRank;
   MPI_Comm_size(localIcComm, &localSize);
   MPI_Comm_rank(localIcComm, &tmpLocalRank);
   //This should be equiv
   assert(tmpLocalRank == localRank);

   commName[0] = '\0';
   if (globalSize > 1) {
      snprintf(commName, COMMNAME_MAXLENGTH, "[%2d]: ", globalRank);
   }

   if (globalSize > 0) {
      neighborInit();
   }

   MPI_Barrier(globalIcComm);
}

Communicator::~Communicator()
{
#ifdef PV_USE_MPI
   MPI_Barrier(globalIcComm);
   if(localIcComm){
      MPI_Comm_free(&localIcComm);
   }
   if(globalIcComm){
      MPI_Comm_free(&globalIcComm);
   }
#endif
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
   int tags[9] = {0, 33, 34, 35, 34, 34, 35, 34, 33};
   // NW and SE corners have tag 33; edges have tag 34; NE and SW corners have tag 35.
   // In the top row of processes in the hypercolumn, a process is both the
   // northeast and east neighbor of the process to its left.  If there is only one
   // row, a process is the northeast, east, and southeast neighbor of the process
   // to its left.  The numbering of tags ensures that the MPI_Send/MPI_Irecv calls
   // can be distinguished.

   for (int i = 0; i < NUM_NEIGHBORHOOD; i++) {
      int n = neighborIndex(localRank, i);
      neighbors[i] = localRank;   // default neighbor is self
      remoteNeighbors[i] = 0;
      if (n >= 0) {
         neighbors[i] = n;
         remoteNeighbors[num_neighbors++] = n;
#ifdef DEBUG_OUTPUT
         pvDebug().printf("[%2d]: neighborInit: remote[%d] of %d is %d, i=%d, neighbor=%d\n",
                localRank, num_neighbors - 1, this->numNeighbors, n, i, neighbors[i]);
#endif // DEBUG_OUTPUT
      } else {
#ifdef DEBUG_OUTPUT
         pvDebug().printf("[%2d]: neighborInit: i=%d, neighbor=%d\n", localRank, i, neighbors[i]);
#endif // DEBUG_OUTPUT
      }
      this->tags[i] = tags[i];
   }
   assert(this->numNeighbors == num_neighbors);

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
 * Returns the batch column id for the given communication id
 */
int Communicator::commBatch(int commId)
{
   return batchFromRank(commId, batchWidth, numRows, numCols);
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
   int nbrIdx = neighborIndex(localRank, neighbor);
   return nbrIdx >= 0;
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
int Communicator::neighborIndex(int commId, int direction)
{
   int row = commRow(commId);
   int column = commColumn(commId);
   switch (direction) {
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
      pvErrorNoExit().printf("neighborIndex %d: bad index\n", direction);
      return -1;
   }
}

/*
 * In a send/receive exchange, when rank A makes an MPI send to its neighbor in direction x,
 * that neighbor must make a complementary MPI receive call.  To get the tags correct,
 * the receiver needs to know the direction that the sender was using in determining which
 * process to send to.
 *
 * Thus, if every process does an MPI send in each direction, to the process of rank neighborIndex(icRank,direction) with tag[direction],
 * every process must also do an MPI receive in each direction, to the process of rank neighborIndex(icRank,direction) with tag[reverseDirection(icRank,direction)].
 */
int Communicator::reverseDirection(int commId, int direction) {
   int neighbor = neighborIndex(commId, direction);
   if (neighbor == commId) {
      return -1;
   }
   int revdir = 9-direction; // Correct unless at an edge of the MPI quilt
   int col = commColumn(commId);
   int row = commRow(commId);
   switch(direction) {
   case LOCAL:
      assert(0); // Should have neighbor==commId, so should have already returned
      break;
   case NORTHWEST : /* northwest */
      assert(revdir==SOUTHEAST);
      if (row==0) {
         assert(col>0);
         revdir=NORTHEAST;
      }
      if (col==0) {
         assert(row>0);
         revdir=SOUTHWEST;
      }
      break;
   case NORTH     : /* north */
      assert(commRow(commId)>0); // If row==0, there is no north neighbor so should have already returned.
      break;
   case NORTHEAST : /* northeast */
      assert(revdir==SOUTHWEST);
      if (row==0) {
         assert(col<numCols-1);
         revdir=NORTHWEST;
      }
      if (col==numCols-1) {
         assert(row>0);
         revdir=SOUTHEAST;
      }
      break;
   case WEST      : /* west */
      assert(commColumn(commId)>0);
      break;
   case EAST      : /* east */
      assert(commColumn(commId)<numCols-1);
      break;
   case SOUTHWEST : /* southwest */
      assert(revdir==NORTHEAST);
      if (row==numRows-1) {
         assert(col>0);
         revdir=SOUTHEAST;
      }
      if (col==0) {
         assert(row<numRows-1);
         revdir=NORTHWEST;
      }
      break;
   case SOUTH     : /* south */
      assert(commRow(commId)<numRows-1);
      break;
   case SOUTHEAST : /* southeast */
      assert(revdir==NORTHWEST);
      if (row==numRows-1) {
         assert(col<numCols-1);
         revdir=SOUTHWEST;
      }
      if (col==numCols-1) {
         assert(row<numRows-1);
         revdir=NORTHEAST;
      }
      break;
   default:
      pvErrorNoExit().printf("neighborIndex %d: bad index\n", direction);
      revdir = -1;
      break;
   }
   return revdir;
}


/**
 * Returns the recv data offset for the given neighbor
 *  - recv into borders
 */
size_t Communicator::recvOffset(int n, const PVLayerLoc * loc)
{
   //This check should make sure n is a local rank
   const int nx = loc->nx;
   const int ny = loc->ny;
   const int leftBorder = loc->halo.lt;
   const int topBorder = loc->halo.dn;

   const int sx = strideXExtended(loc);
   const int sy = strideYExtended(loc);

   switch (n) {
   case LOCAL:
      return (sx*leftBorder         + sy * topBorder);
   case NORTHWEST:
      return ((size_t) 0                            );
   case NORTH:
      return (sx*leftBorder                         );
   case NORTHEAST:
      return (sx*leftBorder + sx*nx                 );
   case WEST:
      return (                        sy * topBorder);
   case EAST:
      return (sx*leftBorder + sx*nx + sy * topBorder);
   case SOUTHWEST:
      return (                      + sy * (topBorder + ny));
   case SOUTH:
      return (sx*leftBorder         + sy * (topBorder + ny));
   case SOUTHEAST:
      return (sx*leftBorder + sx*nx + sy * (topBorder + ny));
   default:
      pvErrorNoExit().printf("recvOffset: bad neighbor index %d\n", n);
      return (size_t) 0;
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
   const size_t leftBorder = loc->halo.lt;
   const size_t topBorder = loc->halo.up;

   const size_t sx = strideXExtended(loc);
   const size_t sy = strideYExtended(loc);

   bool has_north_nbr = hasNorthernNeighbor(commRow(), commColumn());
   bool has_west_nbr = hasWesternNeighbor(commRow(), commColumn());
   bool has_east_nbr = hasEasternNeighbor(commRow(), commColumn());
   bool has_south_nbr = hasSouthernNeighbor(commRow(), commColumn());

   switch (n) {
   case LOCAL:
      return (sx*leftBorder                      + sy*topBorder);
   case NORTHWEST:
      return (sx*has_west_nbr*leftBorder         + sy*has_north_nbr*topBorder);
   case NORTH:
      return (sx*leftBorder                      + sy*topBorder);
   case NORTHEAST:
      return (sx*(nx + !has_east_nbr*leftBorder) + sy*has_north_nbr*topBorder);
   case WEST:
      return (sx*leftBorder                      + sy*topBorder);
   case EAST:
      return (sx*nx                            + sy*topBorder);
   case SOUTHWEST:
      return (sx*has_west_nbr*leftBorder         + sy*(ny + !has_south_nbr*topBorder));
   case SOUTH:
      return (sx*leftBorder                      + sy*ny);
   case SOUTHEAST:
      return (sx*(nx + !has_east_nbr*leftBorder) + sy*(ny + !has_south_nbr*topBorder));
   default:
      pvErrorNoExit().printf("sendOffset: bad neighbor index %d\n", n);
      return 0;
   }
}

/**
 * Create a set of data types for inter-neighbor communication
 *   - caller should delete the MPI_Datatype array by calling Communicator::freeDatatypes
 */
MPI_Datatype * Communicator::newDatatypes(const PVLayerLoc * loc)
{
#ifdef PV_USE_MPI
   int count, blocklength, stride;

   MPI_Datatype * comms = new MPI_Datatype [NUM_NEIGHBORHOOD];
   
   const int leftBorder = loc->halo.lt;
   const int rightBorder = loc->halo.rt;
   const int bottomBorder = loc->halo.dn;
   const int topBorder = loc->halo.up;

   const int nf = loc->nf;

   count       = loc->ny;
   blocklength = nf*loc->nx;
   stride      = nf*(loc->nx + leftBorder + rightBorder);

   /* local interior */
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[LOCAL]);
   MPI_Type_commit(&comms[LOCAL]);

   count = topBorder;

   /* northwest */
   blocklength = nf*leftBorder;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[NORTHWEST]);
   MPI_Type_commit(&comms[NORTHWEST]);

   /* north */
   blocklength = nf*loc->nx;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[NORTH]);
   MPI_Type_commit(&comms[NORTH]);

   /* northeast */
   blocklength = nf*rightBorder;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[NORTHEAST]);
   MPI_Type_commit(&comms[NORTHEAST]);

   count       = loc->ny;

   /* west */
   blocklength = nf*leftBorder;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[WEST]);
   MPI_Type_commit(&comms[WEST]);

   /* east */
   blocklength = nf*rightBorder;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[EAST]);
   MPI_Type_commit(&comms[EAST]);

   count = bottomBorder;

   /* southwest */
   blocklength = nf*leftBorder;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[SOUTHWEST]);
   MPI_Type_commit(&comms[SOUTHWEST]);

   /* south */
   blocklength = nf*loc->nx;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[SOUTH]);
   MPI_Type_commit(&comms[SOUTH]);

   /* southeast */
   blocklength = nf*rightBorder;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &comms[SOUTHEAST]);
   MPI_Type_commit(&comms[SOUTHEAST]);

   return comms;
#else // PV_USE_MPI
   return NULL;
#endif // PV_USE_MPI
}

/* Frees an MPI_Datatype array previously created with Communicator::newDatatypes */
int Communicator::freeDatatypes(MPI_Datatype * mpi_datatypes) {
#ifdef PV_USE_MPI
   if(mpi_datatypes) {
      for ( int n=0; n<NUM_NEIGHBORHOOD; n++ ) {
         MPI_Type_free(&mpi_datatypes[n]);
      }
      delete[] mpi_datatypes;
   }
#endif // PV_USE_MPI
   return PV_SUCCESS;
}

/**
 * Exchange data with neighbors
 *   - the data regions to be sent are described by the datatypes
 *   - do irecv first so there is a location for send data to be received
 */
int Communicator::exchange(pvdata_t * data,
                           const MPI_Datatype neighborDatatypes [],
                           const PVLayerLoc * loc, std::vector<MPI_Request> & req)
{
#ifdef PV_USE_MPI
   PVHalo const * halo = &loc->halo;
   if (halo->lt==0 && halo->rt==0 && halo->dn==0 && halo->up==0) { return PV_SUCCESS; }

   req.clear();
   // don't send interior
   for (int n = 1; n < NUM_NEIGHBORHOOD; n++) {
      if (neighbors[n] == localRank) continue;  // don't send interior/self
      pvdata_t * recvBuf = data + recvOffset(n, loc);
#ifdef DEBUG_OUTPUT
      pvInfo().printf("[%2d]: recv,send to %d, n=%d recvOffset==%ld sendOffset==%ld send[0]==%f\n", localRank, neighbors[n], n, recvOffset(n,loc), sendOffset(n,loc), sendBuf[0]);
      pvInfo().flush();
#endif // DEBUG_OUTPUT
      auto sz = req.size();
      req.resize(sz+1);
      MPI_Irecv(recvBuf, 1, neighborDatatypes[n], neighbors[n], getReverseTag(n), localIcComm,
                &(req.data())[sz]);
   }

   for (int n = 1; n < NUM_NEIGHBORHOOD; n++) {
      if (neighbors[n] == localRank) continue;  // don't send interior/self
      pvdata_t * sendBuf = data + sendOffset(n, loc);
#ifdef DEBUG_OUTPUT
      pvInfo().printf("[%2d]: recv,send to %d, n=%d recvOffset==%ld sendOffset==%ld send[0]==%f\n", localRank, neighbors[n], n, recvOffset(n,loc), sendOffset(n,loc), sendBuf[0]);
      pvInfo().flush();
#endif // DEBUG_OUTPUT
      auto sz = req.size();
      req.resize(sz+1);
      MPI_Isend( sendBuf, 1, neighborDatatypes[n], neighbors[n], getTag(n), localIcComm,
                 &(req.data())[sz]);
   }

   // don't recv interior
#ifdef DEBUG_OUTPUT
   pvInfo().printf("[%2d]: waiting for data, count==%d\n", localRank, nreq);
   pvInfo().flush();
#endif // DEBUG_OUTPUT

#endif // PV_USE_MPI

   return PV_SUCCESS;
}

int Communicator::wait(std::vector<MPI_Request> & req) {
   int status = MPI_Waitall(req.size(), req.data(), MPI_STATUSES_IGNORE);
   req.clear();
   return status;
}

} // end namespace PV
