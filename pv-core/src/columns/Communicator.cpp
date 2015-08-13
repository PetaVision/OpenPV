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

int Communicator::gcd ( int a, int b ){
   int c;
   while ( a != 0 ) {
      c = a; a = b%a;  b = c;
   }
   return b;
}

Communicator::Communicator(int argc, char** argv, int nbatch)
{
   float r;

#ifdef PV_USE_MPI
   MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);
   MPI_Comm_size(MPI_COMM_WORLD, &globalSize);
#else // PV_USE_MPI
   globalRank = 0;
   globalSize = 1;
#endif // PV_USE_MPI
   //commInit(argc, argv);

   // sprintf(commName, "[%2d]: ", icRank); // icRank not initialized yet; commName not used until later.

   bool rowsDefined = pv_getopt_int(argc,  argv, "-rows", &numRows, NULL)==0;
   bool colsDefined = pv_getopt_int(argc, argv, "-columns", &numCols, NULL)==0;
   bool batchDefined = pv_getopt_int(argc, argv, "-batchwidth", &batchWidth, NULL)==0;

   bool inferingDim = !rowsDefined || !colsDefined || !batchDefined;

   if(!batchDefined){
      //Case where both rows and cols are defined, we can find out what the batch width is
      if(rowsDefined && colsDefined){
         batchWidth = globalSize/(numRows * numCols);
      }
      else if(rowsDefined && !colsDefined){
         batchWidth = gcd(globalSize/numRows, nbatch);
      }
      else if(!rowsDefined && colsDefined){
         batchWidth = gcd(globalSize/numCols, nbatch);
      }
      else{
         //Find gcd between np and nbatch, and set that as the batchWidth
         batchWidth = gcd(globalSize, nbatch);
      }
   }
   if(batchWidth > nbatch){
      std::cout << "Error: batchWidth of " << batchWidth << " must be bigger than nbatch of " << nbatch << "\n";
      exit(-1);
   }
   if(nbatch % batchWidth != 0){
      std::cout << "Error: batchWidth of " << batchWidth << " must be a multiple of nbatch of " << nbatch << "\n";
      exit(-1);
   }

   int procsLeft = globalSize/batchWidth;
   if( rowsDefined && !colsDefined ) {
      numCols = (int) procsLeft / numRows;
   }
   if( !rowsDefined && colsDefined ) {
      numRows = (int) procsLeft / numCols;
   }
   if( !rowsDefined  && !colsDefined ) {
      r = sqrtf(procsLeft);
      numRows = (int) r;
      numCols = (int) procsLeft / numRows;
   }

   int commSize = batchWidth * numRows * numCols;

   //For debugging
   if(globalRank == 0){
      std::cout << "Running with batchWidth=" << batchWidth << ", numRows=" << numRows << ", and numCols=" << numCols << "\n";
   }

#ifdef PV_USE_MPI
   int exclsize = globalSize - commSize;

   if (exclsize != 0) {
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
      MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);
      if (globalRank==0) {
         if (exclsize < 0) {
            fprintf(stderr, "Error: %d batchwidth, %d rows, and %d columns specified but only %d processes are available.\n", batchWidth, numRows, numCols, globalSize);
         }
         else {
            assert(exclsize > 0);
            if (!inferingDim) {
               fprintf(stderr, "Error: %d batchwidth, %d rows, and %d columns specified but %d processes available.  Excess processes not yet supported.  Exiting.\n", batchWidth, numRows, numCols, globalSize);
            }
            else {
               fprintf(stderr, "Error: trying %d batchwidth, %d rows, and %d columns but this does not correspond to the %d processes specified.\n", batchWidth, numRows, numCols, globalSize);
               fprintf(stderr, "You can use the \"-batchwidth\", \"-rows\", and \"-columns\" options to specify the arrangement of processes.\n");
            }
         }
      }
      exit(EXIT_FAILURE);
   }
   MPI_Comm_dup(MPI_COMM_WORLD, &globalIcComm);
   //Calculate the batch idx from global rank 
   int batchColIdx = commBatch(globalRank);
   //Set local rank
   localRank = globalToLocalRank(globalRank, batchWidth, numRows, numCols);
   //Make new local communicator
   MPI_Comm_split(globalIcComm, batchColIdx, localRank, &localIcComm);
#endif // PV_USE_MPI

//#ifdef DEBUG_OUTPUT
//      fprintf(stderr, "[%2d]: Formed resized communicator, size==%d cols==%d rows==%d\n", icRank, icSize, numCols, numRows);
//#endif // DEBUG_OUTPUT

//   // some ranks are excluded if they don't fit in the processor quilt
//   if (worldRank < commSize) {

//Grab local rank and check for errors
   int tmpLocalRank;
#ifdef PV_USE_MPI
   MPI_Comm_size(localIcComm, &localSize);
   MPI_Comm_rank(localIcComm, &tmpLocalRank);
#else // PV_USE_MPI
   localSize = 1;
   tmpLocalRank = 0;
#endif // PV_USE_MPI
   //This should be equiv
   assert(tmpLocalRank == localRank);


//   }
//   else {
//      icSize = 0;
//      icRank = -worldRank;
//   }

   commName[0] = '\0';
   if (globalSize > 1) {
      snprintf(commName, COMMNAME_MAXLENGTH, "[%2d]: ", globalRank);
   }

   if (globalSize > 0) {
      neighborInit();
   }

#ifdef PV_USE_MPI
   MPI_Barrier(MPI_COMM_WORLD);
#endif

   // install timers
   this->exchange_timer = new Timer("Communicator", " comm", "exchng ");
}

Communicator::~Communicator()
{
#ifdef PV_USE_MPI
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Comm_free(&localIcComm);
   MPI_Comm_free(&globalIcComm);
#endif
   //commFinalize(); // calls MPI_Finalize

   // delete timers
   //
   if (globalCommRank() == 0) {
      exchange_timer->fprint_time(stdout);
      fflush(stdout);
   }
   delete exchange_timer; exchange_timer = NULL;
}

//int Communicator::commInit(int* argc, char*** argv)
//{
//#ifdef PV_USE_MPI
//   // If MPI wasn't initialized, initialize it.
//   // Remember if it was initialized on entry; the destructor will only finalize if the constructor init'ed.
//   // This way, you can do several simulations sequentially by initializing MPI before creating
//   // the first HyPerCol; after running the first simulation the MPI environment will still exist and you
//   // can run the second simulation, etc.
//   MPI_Initialized(&mpi_initialized_on_entry);
//   if( !mpi_initialized_on_entry ) {
//      assert((*argv)[*argc]==NULL); // Open MPI 1.7 assumes this.
//      MPI_Init(argc, argv);
//   }
//   MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
//   MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
//#else // PV_USE_MPI
//   worldRank = 0;
//   worldSize = 1;
//#endif // PV_USE_MPI
//
//#ifdef DEBUG_OUTPUT
//   fprintf(stderr, "[%2d]: Communicator::commInit: world_size==%d\n", worldRank, worldSize);
//#endif // DEBUG_OUTPUT
//
//   return 0;
//}

//int Communicator::commFinalize()
//{
//#ifdef PV_USE_MPI
//   if( !mpi_initialized_on_entry ) MPI_Finalize();
//#endif
//   return 0;
//}

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
         fprintf(stderr, "[%2d]: neighborInit: remote[%d] of %d is %d, i=%d, neighbor=%d\n",
                localRank, num_neighbors - 1, this->numNeighbors, n, i, neighbors[i]);
#endif // DEBUG_OUTPUT
      } else {
         borders[num_borders++] = -n;
#ifdef DEBUG_OUTPUT
         fprintf(stderr, "[%2d]: neighborInit: i=%d, neighbor=%d\n", localRank, i, neighbors[i]);
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
      fprintf(stderr, "ERROR:neighborIndex: bad index\n");
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
      fprintf(stderr, "ERROR:neighborIndex: bad index\n");
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
      fprintf(stderr, "ERROR:recvOffset: bad neighbor index\n");
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
      fprintf(stderr, "ERROR:sendOffset: bad neighbor index\n");
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
                           const PVLayerLoc * loc)
{
#ifdef PV_USE_MPI
   PVHalo const * halo = &loc->halo;
   if (halo->lt==0 && halo->rt==0 && halo->dn==0 && halo->up==0) { return PV_SUCCESS; }
   exchange_timer->start();

   // don't send interior
   int nreq = 0;
   for (int n = 1; n < NUM_NEIGHBORHOOD; n++) {
      if (neighbors[n] == localRank) continue;  // don't send interior/self
      pvdata_t * recvBuf = data + recvOffset(n, loc);
      pvdata_t * sendBuf = data + sendOffset(n, loc);
#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: recv,send to %d, n=%d recvOffset==%ld sendOffset==%ld send[0]==%f\n", localRank, neighbors[n], n, recvOffset(n,loc), sendOffset(n,loc), sendBuf[0]); fflush(stdout);
#endif // DEBUG_OUTPUT
      MPI_Irecv(recvBuf, 1, neighborDatatypes[n], neighbors[n], getReverseTag(n), localIcComm,
                &requests[nreq++]);
      MPI_Send( sendBuf, 1, neighborDatatypes[n], neighbors[n], getTag(n), localIcComm);
   }

   // don't recv interior
   int count = numberOfNeighbors() - 1;
#ifdef DEBUG_OUTPUT
   fprintf(stderr, "[%2d]: waiting for data, count==%d\n", localRank, count); fflush(stdout);
#endif // DEBUG_OUTPUT
   MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);

   exchange_timer->stop();
#endif // PV_USE_MPI

   return PV_SUCCESS;
}

} // end namespace PV
