/*
 * Communicator.cpp
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "Communicator.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include "utils/conversions.hpp"

namespace PV {

int Communicator::gcd(int a, int b) const {
   int c;
   while (a != 0) {
      c = a;
      a = b % a;
      b = c;
   }
   return b;
}

Communicator::Communicator(Arguments *argumentList) {
   int totalSize;
   MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);
   MPI_Comm_size(MPI_COMM_WORLD, &totalSize);

   numRows    = argumentList->getIntegerArgument("NumRows");
   numCols    = argumentList->getIntegerArgument("NumColumns");
   batchWidth = argumentList->getIntegerArgument("BatchWidth");

   bool rowsDefined  = numRows != 0;
   bool colsDefined  = numCols != 0;
   bool batchDefined = batchWidth != 0;

   if (!batchDefined) {
      batchWidth = 1;
   }

   int procsLeft = totalSize / batchWidth;
   if (rowsDefined && !colsDefined) {
      numCols = (int)ceil(procsLeft / numRows);
   }
   if (!rowsDefined && colsDefined) {
      numRows = (int)ceil(procsLeft / numCols);
   }
   if (!rowsDefined && !colsDefined) {
      double r = std::sqrt(procsLeft);
      numRows  = (int)r;
      if (numRows == 0) {
         Fatal() << "Not enough processes left\n";
      }
      numCols = (int)ceil(procsLeft / numRows);
   }

   int commSize = batchWidth * numRows * numCols;

   // For debugging
   if (globalRank == 0) {
      InfoLog() << "Running with batchWidth=" << batchWidth << ", numRows=" << numRows
                << ", and numCols=" << numCols << "\n";
   }

   if (commSize > totalSize) {
      Fatal() << "Number of required processes (NumRows * NumColumns * BatchWidth = " << commSize
              << ") should be the same as, and cannot be larger than, the number of processes "
                 "launched ("
              << totalSize << ")\n";
   }

   globalMPIBlock =
         new MPIBlock(MPI_COMM_WORLD, numRows, numCols, batchWidth, numRows, numCols, batchWidth);
   isExtra = (globalRank >= commSize);
   if (isExtra) {
      WarnLog() << "Global process rank " << globalRank << " is extra, as only " << commSize
                << " mpiProcesses are required. Process exiting\n";
      return;
   }
   // globalMPIBlock's communicator now has only useful mpi processes

   // If RequireReturn was set, wait until global root process gets keyboard input.
   bool requireReturn = argumentList->getBooleanArgument("RequireReturn");
   if (requireReturn) {
      fflush(stdout);
      MPI_Barrier(globalCommunicator());
      if (globalRank == 0) {
         std::printf("Hit enter to begin! ");
         fflush(stdout);
         int charhit = -1;
         while (charhit != '\n') {
            charhit = std::getc(stdin);
         }
      }
      MPI_Barrier(globalCommunicator());
   }

   // Grab globalSize now that extra processes have been exited
   MPI_Comm_size(globalCommunicator(), &globalSize);

   // Make new local communicator
   localMPIBlock =
         new MPIBlock{globalCommunicator(), numRows, numCols, batchWidth, numRows, numCols, 1};
   // Set local rank
   localRank = localMPIBlock->getRank();
   // Make new batch communicator
   batchMPIBlock =
         new MPIBlock{globalCommunicator(), numRows, numCols, batchWidth, 1, 1, batchWidth};

   //#ifdef DEBUG_OUTPUT
   //      DebugLog().printf("[%2d]: Formed resized communicator, size==%d
   //      cols==%d rows==%d\n",
   //      icRank, icSize, numCols, numRows);
   //#endif // DEBUG_OUTPUT

   // Grab local rank and check for errors
   int tmpLocalRank;
   MPI_Comm_size(communicator(), &localSize);
   MPI_Comm_rank(communicator(), &tmpLocalRank);
   // This should be equiv
   pvAssert(tmpLocalRank == localRank);

   if (globalSize > 0) {
      neighborInit();
   }
   MPI_Barrier(globalCommunicator());
}

Communicator::~Communicator() {
#ifdef PV_USE_MPI
   MPI_Barrier(globalCommunicator());
#endif
   delete localMPIBlock;
   delete batchMPIBlock;
   delete globalMPIBlock;
}

/**
 * Initialize the communication neighborhood
 */
int Communicator::neighborInit() {
   int num_neighbors = 0;

   // initialize neighbor and border lists
   // (local borders and remote neighbors form the complete neighborhood)

   this->numNeighbors = numberOfNeighbors();
   int tags[9]        = {0, 1, 2, 3, 2, 2, 3, 2, 1};
   // NW and SE corners have tag 1; edges have tag 2; NE and SW corners have
   // tag 3.
   // In the top row of processes in the hypercolumn, a process is both the
   // northeast and east neighbor of the process to its left.  If there is only
   // one row, a process is the northeast, east, and southeast neighbor of the
   // process to its left.  The numbering of tags ensures that the
   // MPI_Send/MPI_Irecv pairs can be distinguished.

   for (int i = 0; i < NUM_NEIGHBORHOOD; i++) {
      int n        = neighborIndex(localRank, i);
      neighbors[i] = localRank; // default neighbor is self
      if (n >= 0) {
         neighbors[i] = n;
         num_neighbors++;
#ifdef DEBUG_OUTPUT
         DebugLog().printf(
               "[%2d]: neighborInit: remote[%d] of %d is %d, i=%d, neighbor=%d\n",
               localRank,
               num_neighbors - 1,
               this->numNeighbors,
               n,
               i,
               neighbors[i]);
#endif // DEBUG_OUTPUT
      }
      else {
#ifdef DEBUG_OUTPUT
         DebugLog().printf("[%2d]: neighborInit: i=%d, neighbor=%d\n", localRank, i, neighbors[i]);
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
int Communicator::commRow(int commId) const { return rowFromRank(commId, numRows, numCols); }

/**
 * Returns the communication column id for the given communication id
 */
int Communicator::commColumn(int commId) const { return columnFromRank(commId, numRows, numCols); }

/**
 * Returns the batch column id for the given communication id
 */
int Communicator::commBatch(int commId) const {
   return batchFromRank(commId, batchWidth, numRows, numCols);
}

/**
 * Returns the communication id for a given row and column
 */
int Communicator::commIdFromRowColumn(int commRow, int commColumn) const {
   return rankFromRowAndColumn(commRow, commColumn, numRows, numCols);
}

/**
 * Returns true if the given neighbor is present
 * (false otherwise)
 */
bool Communicator::hasNeighbor(int neighbor) const {
   int nbrIdx = neighborIndex(localRank, neighbor);
   return nbrIdx >= 0;
}

/**
 * Returns true if the given commId has a northwestern neighbor
 * (false otherwise)
 */
bool Communicator::hasNorthwesternNeighbor(int row, int column) const {
   return (hasNorthernNeighbor(row, column) || hasWesternNeighbor(row, column));
}

/**
 * Returns true if the given commId has a northern neighbor
 * (false otherwise)
 */
bool Communicator::hasNorthernNeighbor(int row, int column) const { return row > 0; }

/**
 * Returns true if the given commId has a northeastern neighbor
 * (false otherwise)
 */
bool Communicator::hasNortheasternNeighbor(int row, int column) const {
   return (hasNorthernNeighbor(row, column) || hasEasternNeighbor(row, column));
}

/**
 * Returns true if the given commId has a western neighbor
 * (false otherwise)
 */
bool Communicator::hasWesternNeighbor(int row, int column) const { return column > 0; }

/**
 * Returns true if the given commId has an eastern neighbor
 * (false otherwise)
 */
bool Communicator::hasEasternNeighbor(int row, int column) const {
   return column < numCommColumns() - 1;
}

/**
 * Returns true if the given commId has a southwestern neighbor
 * (false otherwise)
 */
bool Communicator::hasSouthwesternNeighbor(int row, int column) const {
   return (hasSouthernNeighbor(row, column) || hasWesternNeighbor(row, column));
}

/**
 * Returns true if the given commId has a southern neighbor
 * (false otherwise)
 */
bool Communicator::hasSouthernNeighbor(int row, int column) const {
   return row < numCommRows() - 1;
}

/**
 * Returns true if the given commId has a southeastern neighbor
 * (false otherwise)
 */
bool Communicator::hasSoutheasternNeighbor(int row, int column) const {
   return (hasSouthernNeighbor(row, column) || hasEasternNeighbor(row, column));
}

/**
 * Returns the number in communication neighborhood (local included)
 */
int Communicator::numberOfNeighbors() {
   int n = 1 + hasNorthwesternNeighbor(commRow(), commColumn())
           + hasNorthernNeighbor(commRow(), commColumn())
           + hasNortheasternNeighbor(commRow(), commColumn())
           + hasWesternNeighbor(commRow(), commColumn())
           + hasEasternNeighbor(commRow(), commColumn())
           + hasSouthwesternNeighbor(commRow(), commColumn())
           + hasSouthernNeighbor(commRow(), commColumn())
           + hasSoutheasternNeighbor(commRow(), commColumn());
   return n;
}

/**
 * Returns the communication id of the northwestern HyperColumn
 */
int Communicator::northwest(int commRow, int commColumn) const {
   int nbr_id = -NORTHWEST;
   if (hasNorthwesternNeighbor(commRow, commColumn)) {
      int nbr_row    = commRow - (commRow > 0);
      int nbr_column = commColumn - (commColumn > 0);
      nbr_id         = commIdFromRowColumn(nbr_row, nbr_column);
   }
   return nbr_id;
}

/**
 * Returns the communication id of the northern HyperColumn
 */
int Communicator::north(int commRow, int commColumn) const {
   int nbr_id = -NORTH;
   if (hasNorthernNeighbor(commRow, commColumn)) {
      nbr_id = commIdFromRowColumn(commRow - 1, commColumn);
   }
   return nbr_id;
}

/**
 * Returns the communication id of the northeastern HyperColumn
 */
int Communicator::northeast(int commRow, int commColumn) const {
   int nbr_id = -NORTHEAST;
   if (hasNortheasternNeighbor(commRow, commColumn)) {
      int nbr_row    = commRow - (commRow > 0);
      int nbr_column = commColumn + (commColumn < numCommColumns() - 1);
      nbr_id         = commIdFromRowColumn(nbr_row, nbr_column);
   }
   return nbr_id;
}

/**
 * Returns the communication id of the western HyperColumn
 */
int Communicator::west(int commRow, int commColumn) const {
   int nbr_id = -WEST;
   if (hasWesternNeighbor(commRow, commColumn)) {
      nbr_id = commIdFromRowColumn(commRow, commColumn - 1);
   }
   return nbr_id;
}

/**
 * Returns the communication id of the eastern HyperColumn
 */
int Communicator::east(int commRow, int commColumn) const {
   int nbr_id = -EAST;
   if (hasEasternNeighbor(commRow, commColumn)) {
      nbr_id = commIdFromRowColumn(commRow, commColumn + 1);
   }
   return nbr_id;
}

/**
 * Returns the communication id of the southwestern HyperColumn
 */
int Communicator::southwest(int commRow, int commColumn) const {
   int nbr_id = -SOUTHWEST;
   if (hasSouthwesternNeighbor(commRow, commColumn)) {
      int nbr_row    = commRow + (commRow < numCommRows() - 1);
      int nbr_column = commColumn - (commColumn > 0);
      nbr_id         = commIdFromRowColumn(nbr_row, nbr_column);
   }
   return nbr_id;
}

/**
 * Returns the communication id of the southern HyperColumn
 */
int Communicator::south(int commRow, int commColumn) const {
   int nbr_id = -SOUTH;
   if (hasSouthernNeighbor(commRow, commColumn)) {
      nbr_id = commIdFromRowColumn(commRow + 1, commColumn);
   }
   return nbr_id;
}

/**
 * Returns the communication id of the southeastern HyperColumn
 */
int Communicator::southeast(int commRow, int commColumn) const {
   int nbr_id = -SOUTHEAST;
   if (hasSoutheasternNeighbor(commRow, commColumn)) {
      int nbr_row    = commRow + (commRow < numCommRows() - 1);
      int nbr_column = commColumn + (commColumn < numCommColumns() - 1);
      nbr_id         = commIdFromRowColumn(nbr_row, nbr_column);
   }
   return nbr_id;
}

/**
 * Returns the intercolumn rank of the neighbor in the given direction
 * If there is no neighbor, returns a negative value
 */
int Communicator::neighborIndex(int commId, int direction) const {
   int row    = commRow(commId);
   int column = commColumn(commId);
   switch (direction) {
      case LOCAL: /* local */ return commId;
      case NORTHWEST: /* northwest */ return northwest(row, column);
      case NORTH: /* north */ return north(row, column);
      case NORTHEAST: /* northeast */ return northeast(row, column);
      case WEST: /* west */ return west(row, column);
      case EAST: /* east */ return east(row, column);
      case SOUTHWEST: /* southwest */ return southwest(row, column);
      case SOUTH: /* south */ return south(row, column);
      case SOUTHEAST: /* southeast */ return southeast(row, column);
      default: ErrorLog().printf("neighborIndex %d: bad index\n", direction); return -1;
   }
}

/*
 * In a send/receive exchange, when rank A makes an MPI send to its neighbor in
 * direction x,
 * that neighbor must make a complementary MPI receive call.  To get the tags
 * correct,
 * the receiver needs to know the direction that the sender was using in
 * determining which
 * process to send to.
 *
 * Thus, if every process does an MPI send in each direction, to the process of
 * rank
 * neighborIndex(icRank,direction) with tag[direction],
 * every process must also do an MPI receive in each direction, to the process
 * of rank
 * neighborIndex(icRank,direction) with tag[reverseDirection(icRank,direction)].
 */
int Communicator::reverseDirection(int commId, int direction) const {
   int neighbor = neighborIndex(commId, direction);
   if (neighbor == commId) {
      return -1;
   }
   int revdir = 9 - direction; // Correct unless at an edge of the MPI quilt
   int col    = commColumn(commId);
   int row    = commRow(commId);
   switch (direction) {
      case LOCAL:
         assert(0); // Should have neighbor==commId, so should have already returned
         break;
      case NORTHWEST: /* northwest */
         assert(revdir == SOUTHEAST);
         if (row == 0) {
            assert(col > 0);
            revdir = NORTHEAST;
         }
         if (col == 0) {
            assert(row > 0);
            revdir = SOUTHWEST;
         }
         break;
      case NORTH: /* north */
         assert(commRow(commId) > 0); // If row==0, there is no north neighbor so
         // should have already returned.
         break;
      case NORTHEAST: /* northeast */
         assert(revdir == SOUTHWEST);
         if (row == 0) {
            assert(col < numCols - 1);
            revdir = NORTHWEST;
         }
         if (col == numCols - 1) {
            assert(row > 0);
            revdir = SOUTHEAST;
         }
         break;
      case WEST: /* west */ assert(commColumn(commId) > 0); break;
      case EAST: /* east */ assert(commColumn(commId) < numCols - 1); break;
      case SOUTHWEST: /* southwest */
         assert(revdir == NORTHEAST);
         if (row == numRows - 1) {
            assert(col > 0);
            revdir = SOUTHEAST;
         }
         if (col == 0) {
            assert(row < numRows - 1);
            revdir = NORTHWEST;
         }
         break;
      case SOUTH: /* south */ assert(commRow(commId) < numRows - 1); break;
      case SOUTHEAST: /* southeast */
         assert(revdir == NORTHWEST);
         if (row == numRows - 1) {
            assert(col < numCols - 1);
            revdir = SOUTHWEST;
         }
         if (col == numCols - 1) {
            assert(row < numRows - 1);
            revdir = NORTHEAST;
         }
         break;
      default:
         ErrorLog().printf("neighborIndex %d: bad index\n", direction);
         revdir = -1;
         break;
   }
   return revdir;
}

} // end namespace PV
