/*
 * BorderExchange.cpp
 */

#include "BorderExchange.hpp"
#include "include/pv_common.h"
#include "utils/PVAssert.hpp"
#include "utils/conversions.hpp"

namespace PV {

BorderExchange::BorderExchange(MPIBlock const &mpiBlock, PVLayerLoc const &loc) {
   mMPIBlock = &mpiBlock; // TODO: copy the block instead of storing a reference.
   mLayerLoc = loc;
   newDatatypes();
   initNeighbors();
}

BorderExchange::~BorderExchange() { freeDatatypes(); }

void BorderExchange::newDatatypes() {
#ifdef PV_USE_MPI
   int count, blocklength, stride;
   mDatatypes.resize(NUM_NEIGHBORHOOD);

   int const leftBorder   = mLayerLoc.halo.lt;
   int const rightBorder  = mLayerLoc.halo.rt;
   int const bottomBorder = mLayerLoc.halo.dn;
   int const topBorder    = mLayerLoc.halo.up;

   int const nf = mLayerLoc.nf;

   count       = mLayerLoc.ny;
   blocklength = nf * mLayerLoc.nx;
   stride      = nf * (mLayerLoc.nx + leftBorder + rightBorder);

   /* local interior */
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &mDatatypes[LOCAL]);
   MPI_Type_commit(&mDatatypes[LOCAL]);

   count = topBorder;

   /* northwest */
   blocklength = nf * leftBorder;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &mDatatypes[NORTHWEST]);
   MPI_Type_commit(&mDatatypes[NORTHWEST]);

   /* north */
   blocklength = nf * mLayerLoc.nx;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &mDatatypes[NORTH]);
   MPI_Type_commit(&mDatatypes[NORTH]);

   /* northeast */
   blocklength = nf * rightBorder;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &mDatatypes[NORTHEAST]);
   MPI_Type_commit(&mDatatypes[NORTHEAST]);

   count = mLayerLoc.ny;

   /* west */
   blocklength = nf * leftBorder;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &mDatatypes[WEST]);
   MPI_Type_commit(&mDatatypes[WEST]);

   /* east */
   blocklength = nf * rightBorder;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &mDatatypes[EAST]);
   MPI_Type_commit(&mDatatypes[EAST]);

   count = bottomBorder;

   /* southwest */
   blocklength = nf * leftBorder;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &mDatatypes[SOUTHWEST]);
   MPI_Type_commit(&mDatatypes[SOUTHWEST]);

   /* south */
   blocklength = nf * mLayerLoc.nx;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &mDatatypes[SOUTH]);
   MPI_Type_commit(&mDatatypes[SOUTH]);

   /* southeast */
   blocklength = nf * rightBorder;
   MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &mDatatypes[SOUTHEAST]);
   MPI_Type_commit(&mDatatypes[SOUTHEAST]);
#else // PV_USE_MPI
   mDatatypes.clear();
#endif // PV_USE_MPI
}

void BorderExchange::freeDatatypes() {
#ifdef PV_USE_MPI
   for (auto &d : mDatatypes) {
      MPI_Type_free(&d);
   }
   mDatatypes.clear();
#endif // PV_USE_MPI
}

void BorderExchange::initNeighbors() {
   neighbors.resize(NUM_NEIGHBORHOOD, -1);
   mNumNeighbors = 0U;

   // initialize neighbor and border lists
   // (local borders and remote neighbors form the complete neighborhood)

   for (int i = 0; i < NUM_NEIGHBORHOOD; i++) {
      int n = neighborIndex(mMPIBlock->getRank(), i);
      if (n >= 0) {
         neighbors[i] = n;
         mNumNeighbors++;
      }
      else {
         neighbors[i] = mMPIBlock->getRank();
      }
   }
}

void BorderExchange::exchange(float *data, std::vector<MPI_Request> &req) {
#ifdef PV_USE_MPI
   PVHalo const &halo = mLayerLoc.halo;
   if (halo.lt == 0 && halo.rt == 0 && halo.dn == 0 && halo.up == 0) {
      return;
   }

   req.clear();
   // Start at n=1 because n=0 is the interior
   for (int n = 1; n < NUM_NEIGHBORHOOD; n++) {
      if (neighbors[n] == mMPIBlock->getRank())
         continue; // don't send interior/self
      float *recvBuf = data + recvOffset(n);
      auto sz        = req.size();
      req.resize(sz + 1);
      int revDir = reverseDirection(mMPIBlock->getRank(), n);
      MPI_Irecv(
            recvBuf,
            1,
            mDatatypes[n],
            neighbors[n],
            exchangeCounter * 16 + mTags[revDir],
            mMPIBlock->getComm(),
            &(req.data())[sz]);
   }

   for (int n = 1; n < NUM_NEIGHBORHOOD; n++) {
      if (neighbors[n] == mMPIBlock->getRank())
         continue; // don't send interior/self
      float *sendBuf = data + sendOffset(n);
      auto sz        = req.size();
      req.resize(sz + 1);
      MPI_Isend(
            sendBuf,
            1,
            mDatatypes[n],
            neighbors[n],
            exchangeCounter * 16 + mTags[n],
            mMPIBlock->getComm(),
            &(req.data())[sz]);
   }

   exchangeCounter = (exchangeCounter == 2047) ? 1024 : exchangeCounter + 1;

// don't recv interior
#endif // PV_USE_MPI
}

int BorderExchange::wait(std::vector<MPI_Request> &req) {
   int status = MPI_Waitall(req.size(), req.data(), MPI_STATUSES_IGNORE);
   req.clear();
   return status;
}

/**
 * Returns the intercolumn rank of the neighbor in the given direction
 * If there is no neighbor, returns a negative value
 */
int BorderExchange::neighborIndex(int commId, int direction) {
   int numRows       = mMPIBlock->getNumRows();
   int numColumns    = mMPIBlock->getNumColumns();
   int rankRowColumn = commId % (numRows * numColumns);
   int row           = rowFromRank(rankRowColumn, numRows, numColumns);
   int column        = columnFromRank(rankRowColumn, numRows, numColumns);
   int neighborRank;
   switch (direction) {
      case LOCAL: return commId;
      case NORTHWEST: neighborRank = northwest(row, column, numRows, numColumns); break;
      case NORTH: neighborRank     = north(row, column, numRows, numColumns); break;
      case NORTHEAST: neighborRank = northeast(row, column, numRows, numColumns); break;
      case WEST: neighborRank      = west(row, column, numRows, numColumns); break;
      case EAST: neighborRank      = east(row, column, numRows, numColumns); break;
      case SOUTHWEST: neighborRank = southwest(row, column, numRows, numColumns); break;
      case SOUTH: neighborRank     = south(row, column, numRows, numColumns); break;
      case SOUTHEAST: neighborRank = southeast(row, column, numRows, numColumns); break;
      default:
         neighborRank = -1;
         pvAssert(0);
         break;
   }
   if (neighborRank >= 0) {
      int rankBatchStart = commId - rankRowColumn;
      neighborRank += rankBatchStart;
   }
   return neighborRank;
}

int BorderExchange::northwest(int row, int column, int numRows, int numColumns) {
   if (hasNorthwesternNeighbor(row, column, numRows, numColumns)) {
      int const neighborRow    = row - (row > 0);
      int const neighborColumn = column - (column > 0);
      return rankFromRowAndColumn(neighborRow, neighborColumn, numRows, numColumns);
   }
   else {
      return -1;
   }
}

int BorderExchange::north(int row, int column, int numRows, int numColumns) {
   if (hasNorthernNeighbor(row, column, numRows, numColumns)) {
      int const neighborRow    = row - 1;
      int const neighborColumn = column;
      return rankFromRowAndColumn(neighborRow, neighborColumn, numRows, numColumns);
   }
   else {
      return -1;
   }
}

int BorderExchange::northeast(int row, int column, int numRows, int numColumns) {
   if (hasNortheasternNeighbor(row, column, numRows, numColumns)) {
      int const neighborRow    = row - (row > 0);
      int const neighborColumn = column + (column < numColumns - 1);
      return rankFromRowAndColumn(neighborRow, neighborColumn, numRows, numColumns);
   }
   else {
      return -1;
   }
}

int BorderExchange::west(int row, int column, int numRows, int numColumns) {
   if (hasWesternNeighbor(row, column, numRows, numColumns)) {
      int const neighborRow    = row;
      int const neighborColumn = column - 1;
      return rankFromRowAndColumn(neighborRow, neighborColumn, numRows, numColumns);
   }
   else {
      return -1;
   }
}

int BorderExchange::east(int row, int column, int numRows, int numColumns) {
   if (hasEasternNeighbor(row, column, numRows, numColumns)) {
      int const neighborRow    = row;
      int const neighborColumn = column + 1;
      return rankFromRowAndColumn(neighborRow, neighborColumn, numRows, numColumns);
   }
   else {
      return -1;
   }
}

int BorderExchange::southwest(int row, int column, int numRows, int numColumns) {
   if (hasSouthwesternNeighbor(row, column, numRows, numColumns)) {
      int const neighborRow    = row + (row < numRows - 1);
      int const neighborColumn = column - (column > 0);
      return rankFromRowAndColumn(neighborRow, neighborColumn, numRows, numColumns);
   }
   else {
      return -1;
   }
}

int BorderExchange::south(int row, int column, int numRows, int numColumns) {
   if (hasSouthernNeighbor(row, column, numRows, numColumns)) {
      int const neighborRow    = row + 1;
      int const neighborColumn = column;
      return rankFromRowAndColumn(neighborRow, neighborColumn, numRows, numColumns);
   }
   else {
      return -1;
   }
}

int BorderExchange::southeast(int row, int column, int numRows, int numColumns) {
   if (hasSoutheasternNeighbor(row, column, numRows, numColumns)) {
      int const neighborRow    = row + (row < numRows - 1);
      int const neighborColumn = column + (column < numColumns - 1);
      return rankFromRowAndColumn(neighborRow, neighborColumn, numRows, numColumns);
   }
   else {
      return -1;
   }
}

bool BorderExchange::hasNorthwesternNeighbor(int row, int column, int numRows, int numColumns) {
   return (row > 0) || (column > 0);
}

bool BorderExchange::hasNorthernNeighbor(int row, int column, int numRows, int numColumns) {
   return row > 0;
}

bool BorderExchange::hasNortheasternNeighbor(int row, int column, int numRows, int numColumns) {
   return (row > 0) || (column < numColumns - 1);
}

bool BorderExchange::hasWesternNeighbor(int row, int column, int numRows, int numColumns) {
   return column > 0;
}

bool BorderExchange::hasEasternNeighbor(int row, int column, int numRows, int numColumns) {
   return column < numColumns - 1;
}

bool BorderExchange::hasSouthwesternNeighbor(int row, int column, int numRows, int numColumns) {
   return (row < numRows - 1) || (column > 0);
}

bool BorderExchange::hasSouthernNeighbor(int row, int column, int numRows, int numColumns) {
   return row < numRows - 1;
}

bool BorderExchange::hasSoutheasternNeighbor(int row, int column, int numRows, int numColumns) {
   return (row < numRows - 1) || (column < numColumns - 1);
}

int BorderExchange::reverseDirection(int commId, int direction) {
   int neighbor = neighborIndex(commId, direction);
   if (neighbor == commId) {
      return LOCAL;
   }
   int revdir        = 9 - direction; // Correct unless at an edge of the MPI quilt
   int numRows       = mMPIBlock->getNumRows();
   int numCols       = mMPIBlock->getNumColumns();
   int rankRowColumn = commId % (numRows * numCols);
   int row           = rowFromRank(rankRowColumn, numRows, numCols);
   int col           = columnFromRank(rankRowColumn, numRows, numCols);
   switch (direction) {
      case LOCAL:
         pvAssert(0); // Should have neighbor==commId, so should have already returned
         break;
      case NORTHWEST:
         pvAssert(revdir == SOUTHEAST);
         if (row == 0) {
            pvAssert(col > 0);
            revdir = NORTHEAST;
         }
         if (col == 0) {
            pvAssert(row > 0);
            revdir = SOUTHWEST;
         }
         break;
      case NORTH:
         pvAssert(row > 0); // If row==0, there is no north neighbor so
         // should have already returned.
         break;
      case NORTHEAST:
         pvAssert(revdir == SOUTHWEST);
         if (row == 0) {
            pvAssert(col < numCols - 1);
            revdir = NORTHWEST;
         }
         if (col == numCols - 1) {
            pvAssert(row > 0);
            revdir = SOUTHEAST;
         }
         break;
      case WEST: pvAssert(col > 0); break;
      case EAST: pvAssert(col < numCols - 1); break;
      case SOUTHWEST:
         pvAssert(revdir == NORTHEAST);
         if (row == numRows - 1) {
            pvAssert(col > 0);
            revdir = SOUTHEAST;
         }
         if (col == 0) {
            pvAssert(row < numRows - 1);
            revdir = NORTHWEST;
         }
         break;
      case SOUTH: pvAssert(row < numRows - 1); break;
      case SOUTHEAST:
         pvAssert(revdir == NORTHWEST);
         if (row == numRows - 1) {
            pvAssert(col < numCols - 1);
            revdir = SOUTHWEST;
         }
         if (col == numCols - 1) {
            pvAssert(row < numRows - 1);
            revdir = NORTHEAST;
         }
         break;
      default:
         pvAssert(0); // All possible directions handled in above cases.
         break;
   }
   return revdir;
}

std::size_t BorderExchange::recvOffset(int direction) {
   // This check should make sure n is a local rank
   const int nx         = mLayerLoc.nx;
   const int ny         = mLayerLoc.ny;
   const int leftBorder = mLayerLoc.halo.lt;
   const int topBorder  = mLayerLoc.halo.dn;

   const int sx = strideXExtended(&mLayerLoc);
   const int sy = strideYExtended(&mLayerLoc);

   int offset;

   switch (direction) {
      case LOCAL: offset     = sx * leftBorder + sy * topBorder; break;
      case NORTHWEST: offset = 0; break;
      case NORTH: offset     = sx * leftBorder; break;
      case NORTHEAST: offset = sx * leftBorder + sx * nx; break;
      case WEST: offset      = sy * topBorder; break;
      case EAST: offset      = sx * leftBorder + sx * nx + sy * topBorder; break;
      case SOUTHWEST: offset = sy * (topBorder + ny); break;
      case SOUTH: offset     = sx * leftBorder + sy * (topBorder + ny); break;
      case SOUTHEAST: offset = sx * leftBorder + sx * nx + sy * (topBorder + ny); break;
      default:
         offset = -1; // Suppresses g++ maybe-uninitialized warning
         pvAssert(0); /* All allowable directions handled in above cases */
         break;
   }
   return (std::size_t)offset;
}

/**
 * Returns the send data offset for the given neighbor
 *  - send from interior
 */
std::size_t BorderExchange::sendOffset(int direction) {
   const size_t nx         = mLayerLoc.nx;
   const size_t ny         = mLayerLoc.ny;
   const size_t leftBorder = mLayerLoc.halo.lt;
   const size_t topBorder  = mLayerLoc.halo.up;

   const size_t sx = strideXExtended(&mLayerLoc);
   const size_t sy = strideYExtended(&mLayerLoc);

   const int numRows = mMPIBlock->getNumRows();
   const int numCols = mMPIBlock->getNumColumns();
   const int row     = mMPIBlock->getRowIndex();
   const int col     = mMPIBlock->getColumnIndex();

   bool hasNorthNeighbor = row > 0;
   bool hasWestNeighbor  = col > 0;
   bool hasEastNeighbor  = col < numCols - 1;
   bool hasSouthNeighbor = row < numRows - 1;

   int offset;
   switch (direction) {
      case LOCAL: offset = sx * leftBorder + sy * topBorder; break;
      case NORTHWEST:
         offset = sx * hasWestNeighbor * leftBorder + sy * hasNorthNeighbor * topBorder;
         break;
      case NORTH: offset = sx * leftBorder + sy * topBorder; break;
      case NORTHEAST:
         offset = sx * (nx + !hasEastNeighbor * leftBorder) + sy * hasNorthNeighbor * topBorder;
         break;
      case WEST: offset = sx * leftBorder + sy * topBorder; break;
      case EAST: offset = sx * nx + sy * topBorder; break;
      case SOUTHWEST:
         offset = sx * hasWestNeighbor * leftBorder + sy * (ny + !hasSouthNeighbor * topBorder);
         break;
      case SOUTH: offset = sx * leftBorder + sy * ny; break;
      case SOUTHEAST:
         return sx * (nx + !hasEastNeighbor * leftBorder)
                + sy * (ny + !hasSouthNeighbor * topBorder);
         break;
      default:
         offset = -1; // Suppresses g++ maybe-uninitialized warning
         pvAssert(0); /* All allowable directions handled in above cases */
         break;
   }
   return (std::size_t)offset;
}

// NW and SE corners have tag 33; edges have tag 34; NE and SW have tag 35.
// In the top row of processes in the hypercolumn, a process is both the
// northeast and east neighbor of the process to its left.  If there is only
// one row, a process is the northeast, east, and southeast neighbor of the
// process to its left.  The numbering of tags ensures that the
// MPI_Send/MPI_Irecv calls between a pair of processes can be distinguished.
std::vector<int> const BorderExchange::mTags = {0, 33, 34, 35, 34, 34, 35, 34, 33};

int BorderExchange::exchangeCounter = 1024;

} // end namespace PV
