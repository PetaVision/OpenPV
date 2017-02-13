/*
 * BorderExchange.hpp
 */

#ifndef BORDEREXCHANGE_HPP_
#define BORDEREXCHANGE_HPP_

#include "arch/mpi/mpi.h"
#include "include/PVLayerLoc.h"
#include "structures/MPIBlock.hpp"
#include <vector>

namespace PV {

class BorderExchange {
  public:
   BorderExchange(MPIBlock const &mpiBlock, PVLayerLoc const &loc);
   ~BorderExchange();

   void exchange(float *data, std::vector<MPI_Request> &req);

   static int wait(std::vector<MPI_Request> &req);

   MPIBlock const *getMPIBlock() const { return mMPIBlock; }

   int getRank() const { return mMPIBlock->getRank(); }

   int getNumNeighbors() const { return mNumNeighbors; }

   static int northwest(int row, int column, int numRows, int numColumns);
   static int north(int row, int column, int numRows, int numColumns);
   static int northeast(int row, int column, int numRows, int numColumns);
   static int west(int row, int column, int numRows, int numColumns);
   static int east(int row, int column, int numRows, int numColumns);
   static int southwest(int row, int column, int numRows, int numColumns);
   static int south(int row, int column, int numRows, int numColumns);
   static int southeast(int row, int column, int numRows, int numColumns);

   static bool hasNorthwesternNeighbor(int row, int column, int numRows, int numColumns);
   static bool hasNorthernNeighbor(int row, int column, int numRows, int numColumns);
   static bool hasNortheasternNeighbor(int row, int column, int numRows, int numColumns);
   static bool hasWesternNeighbor(int row, int column, int numRows, int numColumns);
   static bool hasEasternNeighbor(int row, int column, int numRows, int numColumns);
   static bool hasSouthwesternNeighbor(int row, int column, int numRows, int numColumns);
   static bool hasSouthernNeighbor(int row, int column, int numRows, int numColumns);
   static bool hasSoutheasternNeighbor(int row, int column, int numRows, int numColumns);

  private:
   void newDatatypes();
   void freeDatatypes();

   void initNeighbors();

   /**
    * In a send/receive exchange, when rank A makes an MPI send to its neighbor
    * in direction x, that neighbor must make a complementary MPI receive call.
    * To get the tags correct, the receiver needs to know the direction that
    * the sender was using in determining which process to send to.
    *
    * Thus, if every process does an MPI send in each direction, to the process
    * of rank neighborIndex(icRank,direction) with mTags[direction], every
    * process must also do an MPI receive in each direction, to the process of
    * rank neighborIndex(icRank,direction) with
    * tag[reverseDirection(icRank,direction)].
    */
   int reverseDirection(int commId, int direction);

   /**
    * Returns the recv data offset for the given neighbor
    *  - recv into borders
    */
   std::size_t recvOffset(int direction);

   /**
    * returns the send data offset for the given neighbor
    *  - send from interior
    */
   std::size_t sendOffset(int direction);

   // Data members
  private:
   MPIBlock const *mMPIBlock = nullptr; // TODO: copy mpiBlock instead of storing a pointer.
   PVLayerLoc mLayerLoc;
   std::vector<MPI_Datatype> mDatatypes;
   std::vector<int> neighbors;
   unsigned int mNumNeighbors;

   /**
    * Returns the rank of the neighbor in the given direction
    * If there is no neighbor, returns a negative value
    */
   int neighborIndex(int commId, int direction);

   static int const LOCAL     = 0;
   static int const NORTHWEST = 1;
   static int const NORTH     = 2;
   static int const NORTHEAST = 3;
   static int const WEST      = 4;
   static int const EAST      = 5;
   static int const SOUTHWEST = 6;
   static int const SOUTH     = 7;
   static int const SOUTHEAST = 8;
   static std::vector<int> const mTags;

   static int exchangeCounter;

}; // end class BorderExchange

} // end namespace PV

#endif // BORDEREXCHANGE_HPP_
