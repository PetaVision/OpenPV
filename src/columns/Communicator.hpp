/*
 * Communicator.hpp
 */

#ifndef COMMUNICATOR_HPP_
#define COMMUNICATOR_HPP_

#include "../include/pv_arch.h"
#include "../include/pv_types.h"
#include <stdlib.h>

#ifdef PV_USE_MPI
#  include <mpi.h>
#else
#  include "../include/mpi_stubs.h"
#endif

// number in communicating neighborhood
#define NUM_NEIGHBORHOOD 9

// directional indices
#define LOCAL     0
#define NORTHWEST 1
#define NORTH     2
#define NORTHEAST 3
#define WEST      4
#define EAST      5
#define SOUTHWEST 6
#define SOUTH     7
#define SOUTHEAST 8

namespace PV {

class Communicator {
public:

   static size_t recvOffset(int n, const PVLayerLoc * loc);
   static size_t sendOffset(int n, const PVLayerLoc * loc);

   static MPI_Datatype * newDatatypes(const PVLayerLoc * loc);

   Communicator(int * argc, char *** argv);
   virtual ~Communicator();

   int commInit(int * argc, char *** argv);
   int commFinalize();

   char * name()                { return commName; }

   int commRank()               { return icRank; }
   int commSize()               { return icSize; }
   MPI_Comm communicator()      { return icComm; }

   int numberOfNeighbors(); // includes interior (self) as a neighbor
   int numberOfBorders()        {return numBorders;}

   bool hasNeighbor(int neighborId);

   int commRow()          {return commRow(icRank);}
   int commColumn()       {return commColumn(icRank);}
   int numCommRows()      {return numRows;}
   int numCommColumns()   {return numCols;}

   int exchange(pvdata_t * data,
                const MPI_Datatype neighborDatatypes [],
                const PVLayerLoc * loc);

protected:

   int commRow(int commId);
   int commColumn(int commId);

   int numNeighbors;  // # of remote neighbors plus local
   int numBorders;    // # of border regions (no communicating neighbor)

   //TODO - can this be cleaned up?
   int borders[NUM_NEIGHBORHOOD-1];
   int neighbors[NUM_NEIGHBORHOOD];        // [0] is interior (local)
   int remoteNeighbors[NUM_NEIGHBORHOOD];

private:

   int icRank;
   int icSize;
   int worldRank;
   int worldSize;
   int numRows;
   int numCols;

   char commName[16];

   MPI_Comm    icComm;
   MPI_Request requests[NUM_NEIGHBORHOOD-1];

   // These methods are private for now, move to public as needed

   int neighborInit();

   bool hasNorthwesternNeighbor(int commId);
   bool hasNorthernNeighbor(int commId);
   bool hasNortheasternNeighbor(int commId);
   bool hasWesternNeighbor(int commId);
   bool hasEasternNeighbor(int commId);
   bool hasSouthwesternNeighbor(int commId);
   bool hasSouthernNeighbor(int commId);
   bool hasSoutheasternNeighbor(int commId);

   int northwest(int commId);
   int north(int commId);
   int northeast(int commId);
   int west(int commId);
   int east(int commId);
   int southwest(int commId);
   int south(int commId);
   int southeast(int commId);

   int neighborIndex(int commId, int index);

};

} // namespace PV

#endif /* COMMUNICATOR_HPP_ */
