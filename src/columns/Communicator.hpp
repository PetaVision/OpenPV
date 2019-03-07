/*
 * Communicator.hpp
 */

#ifndef COMMUNICATOR_HPP_
#define COMMUNICATOR_HPP_

#include "Arguments.hpp"
#include "include/pv_arch.h"
#include "include/pv_types.h"
#include "structures/MPIBlock.hpp"
#include <cstdio>
#include <vector>

#include "arch/mpi/mpi.h"

#define COMMNAME_MAXLENGTH 16

namespace PV {

class Communicator {
  public:
   Communicator(Arguments *argumentList);
   virtual ~Communicator();

   // Previous names of MPI getter functions now default to local ranks and sizes
   int commRank() const { return localRank; }
   int globalCommRank() const { return globalRank; }
   int commSize() const { return localSize; }
   int globalCommSize() const { return globalSize; }

   MPI_Comm communicator() const { return localMPIBlock->getComm(); }
   MPI_Comm batchCommunicator() const { return batchMPIBlock->getComm(); }
   MPI_Comm globalCommunicator() const { return globalMPIBlock->getComm(); }

   MPIBlock const *getLocalMPIBlock() const { return localMPIBlock; }
   MPIBlock const *getBatchMPIBlock() const { return batchMPIBlock; }
   MPIBlock const *getGlobalMPIBlock() const { return globalMPIBlock; }

   int numberOfNeighbors(); // includes interior (self) as a neighbor

   bool hasNeighbor(int neighborId) const;
   int neighborIndex(int commId, int index) const;
   int reverseDirection(int commId, int direction) const;

   int commRow() const { return commRow(localRank); }
   int commColumn() const { return commColumn(localRank); }
   int commBatch() const { return commBatch(globalRank); }
   int numCommRows() const { return numRows; }
   int numCommColumns() const { return numCols; }
   int numCommBatches() const { return batchWidth; }

   int getTag(int neighbor) const { return tags[neighbor]; }
   int getReverseTag(int neighbor) const { return tags[reverseDirection(localRank, neighbor)]; }

   bool isExtraProc() const { return isExtra; }

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

   int numNeighbors; // # of remote neighbors plus local.  NOT the size of the
   // neighbors array,
   // which uses negative values to mark directions where there is no remote
   // neighbor.

   bool isExtra; // Defines if the process is an extra process

   int neighbors[NUM_NEIGHBORHOOD]; // [0] is interior (local)
   int tags[NUM_NEIGHBORHOOD]; // diagonal communication needs a different tag
   int exchangeCounter = 1024;
   // from left/right or
   // up/down communication.

  private:
   int gcd(int a, int b) const;

   int localRank;
   int localSize;
   int globalRank;
   int globalSize;
   int batchRank;
   int numRows;
   int numCols;
   int batchWidth;

   MPIBlock *localMPIBlock  = nullptr;
   MPIBlock *batchMPIBlock  = nullptr;
   MPIBlock *globalMPIBlock = nullptr;

   // These methods are private for now, move to public as needed

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
