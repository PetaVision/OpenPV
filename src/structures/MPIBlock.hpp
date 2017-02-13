/*
 * MPIBlock.hpp
 *
 *  Created on: Dec 7, 2016
 *      Author: Pete Schultz
 */

#ifndef MPIBLOCK_HPP_
#define MPIBLOCK_HPP_

#include "arch/mpi/mpi.h"

namespace PV {

/**
 * A class to define a set of MPI processes made up of a block of
 * rows, columns, and batch elements in the MPI setup.
 * The buffer MPI utilities take an MPIBlock as an argument.
 * The Communicator class contains two MPIBlock data members, one for the
 * global communicator and one for the local communicator that slices along
 * the batch dimension.
 */
class MPIBlock {
  public:
   /**
    * The constructor for MPIBlock. The MPI_Comm argument specifies
    * the communicator that will be used to create the MPI communicators
    * for the subblocks. Typically, it is the global communicator after
    * any excess processes are removed. (We cannot use the Communicator
    * object because the Communicator uses MPIBlocks as data members.)
    *
    * globalNumRows, globalNumColumns, globalBatchDimension indicate
    * the dimensions of the MPI configuration covered by the global
    * MPI communicator.
    *
    * blockNumRows, blockNumColumns, blockBatchDimension indicate
    * the dimensions of the local block.
    *
    * The constructor saves the dimensions of the block; and computes
    * the global row, column, and batch index of the start of the block;
    * and the row, column, and batch index of the process relative to the
    * start of the block. These values all have get-methods.
    */
   MPIBlock(
         MPI_Comm comm,
         int globalNumRows,
         int globalNumColumns,
         int globalBatchDimension,
         int blockNumRows,
         int blockNumColumns,
         int blockBatchDimension);

   /**
    * Returns the rank of the process in the MPI block with the indicated
    * row, column, and batch index. If any of the arguments are out of bounds,
    * throws an invalid_argument exception.
    */
   int
   calcRankFromRowColBatch(int const rowIndex, int const columnIndex, int const batchIndex) const;

   /**
    * Calculates the row, column, and batch index of the process with the
    * given rank. Throws an invalid_argument exception if rank is out of
    * bounds.
    */
   void
   calcRowColBatchFromRank(int const rank, int &rowIndex, int &columnIndex, int &batchIndex) const;

   /**
    * Calculates the row index of the process with the given rank.
    * Throws an invalid_argument exception if rank is out of bounds.
    */
   int calcRowFromRank(int const rank) const;

   /**
    * Calculates the column index of the process with the given rank.
    * Throws an invalid_argument exception if rank is out of bounds.
    */
   int calcColumnFromRank(int const rank) const;

   /**
    * Calculates the batch index of the process with the given rank.
    * Throws an invalid_argument exception if rank is out of bounds.
    */
   int calcBatchIndexFromRank(int const rank) const;

   /**
    * Returns the MPI_Comm object for the block the running process belongs to.
    */
   MPI_Comm getComm() const { return mComm; }

   /**
    * Returns the global MPI_Comm object for the entire column.
    */
   MPI_Comm getGlobalComm() const { return mGlobalComm; }

   /**
    * Returns the rank within the MPI block of the running process.
    */
   int getRank() const { return mRank; }

   /**
    * Returns the rank within the global MPI block
    */
   int getGlobalRank() const { return mGlobalRank; }

   /**
    * Returns the global number of rows, across all blocks
    */
   int getGlobalNumRows() const { return mGlobalNumRows; }

   /**
    * Returns the global number of columns, across all blocks
    */
   int getGlobalNumColumns() const { return mGlobalNumColumns; }

   /**
    * Returns the global batch dimension, across all blocks
    */
   int getGlobalBatchDimension() const { return mGlobalBatchDimension; }

   /**
    * Returns the size of the MPI block in the row dimension.
    */
   int getNumRows() const { return mNumRows; }

   /**
    * Returns the size of the MPI block in the column dimension.
    */
   int getNumColumns() const { return mNumColumns; }

   /**
    * Returns the number of MPI process the block covers in the batch dimension.
    */
   int getBatchDimension() const { return mBatchDimension; }

   /**
    * Returns the number of processes in the block: that is, the product of
    * getNumRows(), getNumColumns(), and getBatchDimension().
    */
   int getSize() const { return mNumRows * mNumColumns * mBatchDimension; }

   /**
    * Returns the global row index of the start of the MPI block.
    */
   int getStartRow() const { return mStartRow; }

   /**
    * Returns the global column index of the start of the MPI block.
    */
   int getStartColumn() const { return mStartColumn; }

   /**
    * Returns the global MPI-batch index of the start of the MPI block.
    */
   int getStartBatch() const { return mStartBatch; }

   /**
    * Returns the row index within the MPI block of the running process.
    */
   int getRowIndex() const { return mRowIndex; }

   /**
    * Returns the column index within the MPI block of the running process.
    */
   int getColumnIndex() const { return mColumnIndex; }

   /**
    * Returns the MPI-batch index within the MPI block of the running process.
    */
   int getBatchIndex() const { return mBatchIndex; }

  private:
   /**
    * Used internally by the constructor to set GlobalNumRows,
    * GlobalNumColumns and GlobalBatchDimension data members. If the arguments
    * are positive, the data member is the specified value.
    * Nonpositive arguments are handled as follows:
    *    globalBatchDimension<=0: GlobalBatchDimension becomes 1.
    *    globalNumColumns<=0 but globalNumRows>0:
    *        GlobalNumColumns is the largest integer such that
    *        GlobalNumColumns * GlobalNumRows * GlobalBatchDimension is
    *        no larger than the MPI_Comm size.
    *    globalNumRows<=0 but globalNumColumns>0:
    *        GlobalNumRows is the largest integer such that
    *        GlobalNumColumns * GlobalNumRows * GlobalBatchDimension is
    *        no larger than the MPI_Comm size.
    *    Both globalNumRows<=0 and globalNumColumns<=0:
    *        GlobalNumRows is the largest integer not larger than
    *        sqrt(MPI_Comm size / GlobalBatchDimension),
    *        and GlobalNumRows determined as above.
    */
   void initGlobalDimensions(
         MPI_Comm comm,
         int const globalNumRows,
         int const globalNumColumns,
         int const globalBatchDimension);

   /**
    * Used internally by the constructor to set NumRows, NumColumns, and
    * BatchDimension. These are the dimensions of the MPIBlock, not the
    * dimensions of the global MPI configuration.
    * If the arguments are positive, the data member is the specified value.
    * If any argument is positive, the corresponding data member is
    * the same as the global data member in the corresponding dimension
    * (so that there is a single block in that dimension).
    * Must be called after the initGlobalDimensions method.
    */
   void initBlockDimensions(
         int const blockNumRows,
         int const blockNumColumns,
         int const globalBatchDimension);

   /**
    * Used internally by the constructor to set the StartRow, RowIndex,
    * StartColumn, ColumnIndex, StartBatch, and BatchIndex data members.
    * Start[dimension] is the index in that dimension in the global
    * MPI configuration of the start of the current process's block.
    * [dimension]Index is the local index within the current process's block.
    * Must be called after the initGlobalDimensions and initBlockDimensions
    * methods.
    */
   void initBlockLocation(MPI_Comm comm);

   /**
    * Used internally by the constructor to create the MPI communicator
    * for the various blocks defined by the data members.
    * Must be called after the initBlockLocation method.
    */
   void createBlockComm(MPI_Comm comm);

   /**
    * Used internally by the public calc-methods that take rank as an
    * argument. It tests whether rank is >=0 and <= getSize(), and
    * throws an invalid_argument exception if not.
    */
   void checkRankInBounds(int const rank) const;

   /**
    * Calculates the row index of the process with the given rank.
    * Does not do any bounds-checking. (The calcRowFromRank method
    * calls this internally.)
    */
   int calcRowFromRankInternal(int const rank) const;

   /**
    * Calculates the column index of the process with the given rank.
    * Does not do any bounds-checking. (The calcColumnFromRank method
    * calls this internally.)
    */
   int calcColumnFromRankInternal(int const rank) const;

   /**
    * Calculates the batch index of the process with the given rank.
    * Does not do any bounds-checking. (The calcBatchINdexFromRank method
    * calls this internally.)
    */
   int calcBatchIndexFromRankInternal(int const rank) const;

   /**
    * Used internally by createBlockComm. Returns the number of cells needed
    * in one dimension (which dimension it is doesn't affect the calculation)
    * based on the given cell size and overall dimension. If cellSize divides
    * overallSize evenly, the return value is the quotient. Otherwise, it
    * is one more than the integer quotient (e.g. if cellSize is 3 and overall
    * size is 16, the return value is 6, since 5 cells would only cover 15
    * processes).
    */
   static int calcNumCells(int cellSize, int overallSize);

   // Data members
  private:
   MPI_Comm mGlobalComm;
   MPI_Comm mComm;
   int mGlobalRank           = 0;
   int mRank                 = 0;
   int mGlobalNumRows        = 0;
   int mGlobalNumColumns     = 0;
   int mGlobalBatchDimension = 0;
   int mNumRows              = 0;
   int mNumColumns           = 0;
   int mBatchDimension       = 0;
   int mStartRow             = 0;
   int mStartColumn          = 0;
   int mStartBatch           = 0;
   int mRowIndex             = 0;
   int mColumnIndex          = 0;
   int mBatchIndex           = 0;

}; // end classe MPIBlock

} // end namespace PV

#endif // MPIBLOCK_HPP_
