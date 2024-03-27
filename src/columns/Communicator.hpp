/*
 * Communicator.hpp
 */

#ifndef COMMUNICATOR_HPP_
#define COMMUNICATOR_HPP_

#include "Arguments.hpp"
#include "include/pv_arch.h"
#include "include/pv_common.h"
#include "io/FileManager.hpp"
#include "structures/MPIBlock.hpp"
#include <cstdio>
#include <memory>
#include <vector>

#include "arch/mpi/mpi.h"

#define COMMNAME_MAXLENGTH 16

namespace PV {

class Communicator {
  public:
   Communicator(Arguments const *arguments);
   virtual ~Communicator();

   // Previous names of MPI getter functions now default to local ranks and sizes
   int commRank() const { return mLocalRank; }
   int globalCommRank() const { return mGlobalRank; }
   int commSize() const { return mLocalSize; }
   int globalCommSize() const { return mGlobalSize; }

   MPI_Comm communicator() const { return mLocalMPIBlock->getComm(); }
   MPI_Comm batchCommunicator() const { return mBatchMPIBlock->getComm(); }
   MPI_Comm globalCommunicator() const { return mGlobalMPIBlock->getComm(); }
   MPI_Comm ioCommunicator() const { return mIOMPIBlock->getComm(); }

   std::shared_ptr<MPIBlock const> getLocalMPIBlock() const { return mLocalMPIBlock; }
   std::shared_ptr<MPIBlock const> getBatchMPIBlock() const { return mBatchMPIBlock; }
   std::shared_ptr<MPIBlock const> getGlobalMPIBlock() const { return mGlobalMPIBlock; }
   std::shared_ptr<MPIBlock const> getIOMPIBlock() const { return mIOMPIBlock; }

   std::shared_ptr<FileManager const> getOutputFileManager() const { return mOutputFileManager; }
   std::shared_ptr<FileManager> getOutputFileManager() { return mOutputFileManager; }

   int numberOfNeighbors(); // includes interior (self) as a neighbor

   bool hasNeighbor(int neighborId) const;
   int neighborIndex(int commId, int index) const;
   int reverseDirection(int commId, int direction) const;

   int commRow() const { return commRow(mLocalRank); }
   int commColumn() const { return commColumn(mLocalRank); }
   int commBatch() const { return commBatch(mGlobalRank); }
   int numCommRows() const { return mNumRows; }
   int numCommColumns() const { return mNumCols; }
   int numCommBatches() const { return mBatchWidth; }

   int getTag(int neighbor) const { return mTags[neighbor]; }
   int getReverseTag(int neighbor) const { return mTags[reverseDirection(mLocalRank, neighbor)]; }

   bool isExtraProc() const { return mExtraFlag; }

   static const int LOCAL     = 0;
   static const int NORTHWEST = 1;
   static const int NORTH     = 2;
   static const int NORTHEAST = 3;
   static const int WEST      = 4;
   static const int EAST      = 5;
   static const int SOUTHWEST = 6;
   static const int SOUTH     = 7;
   static const int SOUTHEAST = 8;

  protected:
   int commRow(int commId) const;
   int commColumn(int commId) const;
   int commBatch(int commId) const;
   int commIdFromRowColumn(int commRow, int commColumn) const;

   /**
    * Sets number of rows, number of columns, and batch width in the communicator
    * Uses values of NumRows, NumColumns, and BatchWidth from the arguments, and if any of them are
    * nonpositive, fills in default values so that all of the Communicator's NumRows, NumCols,
    * and BatchWidth values are positive.
    * The default value of BatchWidth is one.
    * If the argument object's NumRows is nonpositive but NumColumns is positive, the Communicator's
    * NumRows is the number of processes divided by BatchWidth, divided by NumColumns.
    * If the argument's object NumColumns is nonpositive but NumRows is positive, the Communicator's
    * NumColumns is the number of processes divided by BatchWidth, divided by NumRows.
    * If the argument's NumRows and NumColumns are both nonpositive, NumRows is the square root
    * of the number of processes divided by the batch width, rounded down, and NumCols is then
    * computed from the batch width and the new NumRows as above. 
    * setDimensions does not
    */
   void setDimensions(Arguments const *arguments, int totalProcs);

   int mNumNeighbors;
   // # of remote neighbors plus local.  NOT the size of the neighbors array,
   // which uses negative values to mark directions where there is no remote neighbor.

   bool mExtraFlag; // Defines if the process is an extra process

   int neighbors[NUM_NEIGHBORHOOD]; // [0] is interior (local)
   int mTags[NUM_NEIGHBORHOOD]; // diagonal communication needs a different tag
   // from left/right or up/down communication.

  private:
   int gcd(int a, int b) const;

   int mLocalRank;
   int mLocalSize;
   int mGlobalRank;
   int mGlobalSize;
   int mNumRows;
   int mNumCols;
   int mBatchWidth;

   std::shared_ptr<MPIBlock> mLocalMPIBlock  = nullptr;
   std::shared_ptr<MPIBlock> mBatchMPIBlock  = nullptr;
   std::shared_ptr<MPIBlock> mGlobalMPIBlock = nullptr;
   std::shared_ptr<MPIBlock> mIOMPIBlock     = nullptr;
   std::shared_ptr<FileManager> mOutputFileManager = nullptr;

   int neighborInit();

   bool hasNorthwesternNeighbor(int commRow, int commColumn) const;
   bool hasNorthernNeighbor(int commRow, int commColumn) const;
   bool hasNortheasternNeighbor(int commRow, int commColumn) const;
   bool hasWesternNeighbor(int commRow, int commColumn) const;
   bool hasEasternNeighbor(int commRow, int commColumn) const;
   bool hasSouthwesternNeighbor(int commRow, int commColumn) const;
   bool hasSouthernNeighbor(int commRow, int commColumn) const;
   bool hasSoutheasternNeighbor(int commRow, int commColumn) const;

   int northwest(int commRow, int commColumn) const;
   int north(int commRow, int commColumn) const;
   int northeast(int commRow, int commColumn) const;
   int west(int commRow, int commColumn) const;
   int east(int commRow, int commColumn) const;
   int southwest(int commRow, int commColumn) const;
   int south(int commRow, int commColumn) const;
   int southeast(int commRow, int commColumn) const;
};

} // namespace PV

#endif /* COMMUNICATOR_HPP_ */
