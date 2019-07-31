#include "MPIBlock.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include "utils/conversions.hpp"
#include <cmath>
#include <stdexcept>

namespace PV {

MPIBlock::MPIBlock(
      MPI_Comm comm,
      int globalNumRows,
      int globalNumColumns,
      int globalBatchDimension,
      int blockNumRows,
      int blockNumColumns,
      int blockBatchDimension) {

   mGlobalComm = comm;
   initGlobalDimensions(comm, globalNumRows, globalNumColumns, globalBatchDimension);
   initBlockDimensions(blockNumRows, blockNumColumns, blockBatchDimension);
   initBlockLocation(comm);
   createBlockComm(comm);
}

void MPIBlock::initGlobalDimensions(
      MPI_Comm comm,
      int globalNumRows,
      int globalNumColumns,
      int globalBatchDimension) {

   int globalRank, numProcsAvailable;
   MPI_Comm_rank(comm, &globalRank);
   MPI_Comm_size(comm, &numProcsAvailable);

   mGlobalNumRows        = globalNumRows;
   mGlobalNumColumns     = globalNumColumns;
   mGlobalBatchDimension = globalBatchDimension;

   bool numRowsDefined    = globalNumRows > 0;
   bool numColumnsDefined = globalNumColumns > 0;
   mGlobalBatchDimension  = globalBatchDimension > 0 ? globalBatchDimension : 1;

   int procsLeft = numProcsAvailable / mGlobalBatchDimension;
   if (numRowsDefined && numColumnsDefined) {
      mGlobalNumRows    = globalNumRows;
      mGlobalNumColumns = globalNumColumns;
   }
   if (numRowsDefined && !numColumnsDefined) {
      mGlobalNumRows    = globalNumRows;
      mGlobalNumColumns = (int)ceil(procsLeft / globalNumRows);
   }
   if (!numRowsDefined && numColumnsDefined) {
      mGlobalNumRows    = (int)ceil(procsLeft / globalNumColumns);
      mGlobalNumColumns = globalNumRows;
   }
   if (!numRowsDefined && !numColumnsDefined) {
      double r       = std::sqrt(procsLeft);
      mGlobalNumRows = (int)r;
      FatalIf(mGlobalNumRows == 0, "Not enough processes left\n");
      mGlobalNumColumns = (int)ceil(procsLeft / mGlobalNumRows);
   }

   int numProcsNeeded = mGlobalBatchDimension * mGlobalNumRows * mGlobalNumColumns;

   if (globalRank == 0) {
      FatalIf(
            numProcsNeeded > numProcsAvailable,
            "Number of processes required (%d) is larger than the "
            "number of processes available (%d)\n",
            numProcsNeeded,
            numProcsAvailable);
   }
}

void MPIBlock::initBlockDimensions(int blockNumRows, int blockNumColumns, int blockBatchDimension) {
   mNumRows        = blockNumRows > 0 ? blockNumRows : mGlobalNumRows;
   mNumColumns     = blockNumColumns > 0 ? blockNumColumns : mGlobalNumColumns;
   mBatchDimension = blockBatchDimension > 0 ? blockBatchDimension : mGlobalBatchDimension;
}

void MPIBlock::initBlockLocation(MPI_Comm comm) {
   MPI_Comm_rank(comm, &mGlobalRank);
   int const globalColumnIndex = mGlobalRank % mGlobalNumColumns;
   int const globalRowIndex    = (mGlobalRank / mGlobalNumColumns) % mGlobalNumRows;
   int const globalBatchIndex  = mGlobalRank / (mGlobalNumColumns * mGlobalNumRows);
   int checkRank               = globalBatchIndex * mGlobalNumRows + globalRowIndex;
   checkRank                   = checkRank * mGlobalNumColumns + globalColumnIndex;
   pvAssert(checkRank == mGlobalRank);

   mRowIndex = globalRowIndex % mNumRows;
   mStartRow = globalRowIndex - mRowIndex;

   mColumnIndex = globalColumnIndex % mNumColumns;
   mStartColumn = globalColumnIndex - mColumnIndex;

   mBatchIndex = globalBatchIndex % mBatchDimension;
   mStartBatch = globalBatchIndex - mBatchIndex;
}

void MPIBlock::createBlockComm(MPI_Comm comm) {
   int rowBlock            = mStartRow / mNumRows;
   int columnBlock         = mStartColumn / mNumColumns;
   int batchBlock          = mStartBatch / mBatchDimension;
   int cellsInGlobalRow    = calcNumCells(mNumRows, mGlobalNumRows);
   int cellsInGlobalColumn = calcNumCells(mNumColumns, mGlobalNumColumns);
   int cellIndex =
         rankFromRowAndColumn(rowBlock, columnBlock, cellsInGlobalRow, cellsInGlobalColumn);
   cellIndex += batchBlock * cellsInGlobalRow * cellsInGlobalColumn;

   int cellRank = rankFromRowAndColumn(mRowIndex, mColumnIndex, mNumRows, mNumColumns);
   cellRank += mBatchIndex * mNumRows * mNumColumns;
   int numProcsNeeded = mGlobalBatchDimension * mGlobalNumRows * mGlobalNumColumns;

   MPI_Comm_split(comm, cellIndex, cellRank, &mComm);
   MPI_Comm_rank(mComm, &mRank);

   char commName[256];
   sprintf(commName, "GlobalRank_%d_CellRank_%d_LocalRank_%d", mGlobalRank, cellRank, mRank);
   MPI_Comm_set_name(mComm, commName);

   if (mRank < numProcsNeeded && mRank != cellRank) {
      Fatal().printf("Global rank %d, cellRank %d, mRank %d\n", mGlobalRank, cellRank, mRank);
   }
}

int MPIBlock::calcNumCells(int cellSize, int overallSize) {
   int numCells = overallSize / cellSize; // integer division
   if (overallSize % cellSize != 0) {
      numCells++;
   }
   return numCells;
}

int MPIBlock::calcRankFromRowColBatch(
      int const rowIndex,
      int const columnIndex,
      int const batchIndex) const {
   if (rowIndex < 0 || rowIndex >= getNumRows() || columnIndex < 0 || columnIndex >= getNumColumns()
       || batchIndex >= getBatchDimension()) {
      throw std::invalid_argument("calcRankFromRowColBatch");
   }
   return columnIndex + getNumColumns() * (rowIndex + getNumRows() * batchIndex);
}

void MPIBlock::checkRankInBounds(int const rank) const {
   if (rank < 0 || rank >= getSize()) {
      throw std::invalid_argument("calcRowColBatchFromRank");
   }
}

void MPIBlock::calcRowColBatchFromRank(
      int const rank,
      int &rowIndex,
      int &columnIndex,
      int &batchIndex) const {
   checkRankInBounds(rank);
   columnIndex = calcColumnFromRankInternal(rank);
   rowIndex    = calcRowFromRankInternal(rank);
   batchIndex  = calcBatchIndexFromRankInternal(rank);
}

int MPIBlock::calcRowFromRank(int const rank) const {
   checkRankInBounds(rank);
   return calcRowFromRankInternal(rank);
}

int MPIBlock::calcColumnFromRank(int const rank) const {
   checkRankInBounds(rank);
   return calcColumnFromRankInternal(rank);
}

int MPIBlock::calcBatchIndexFromRank(int const rank) const {
   checkRankInBounds(rank);
   return calcBatchIndexFromRankInternal(rank);
}

int MPIBlock::calcRowFromRankInternal(int const rank) const {
   return (rank / getNumColumns()) % getNumRows(); // Integer division
}

int MPIBlock::calcColumnFromRankInternal(int const rank) const { return rank % getNumColumns(); }

int MPIBlock::calcBatchIndexFromRankInternal(int const rank) const {
   return rank / (getNumColumns() * getNumRows()); // Integer division
}

} // end namespace PV
