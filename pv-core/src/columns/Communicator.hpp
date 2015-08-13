/*
 * Communicator.hpp
 */

#ifndef COMMUNICATOR_HPP_
#define COMMUNICATOR_HPP_

#include <stdio.h>
#include "../include/pv_arch.h"
#include "../include/pv_types.h"
#include "../utils/Timer.hpp"

#ifdef PV_USE_MPI
#  include <mpi.h>
#else
#  include "../include/mpi_stubs.h"
#endif // PV_USE_MPI

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

#define COMMNAME_MAXLENGTH 16

namespace PV {

class Communicator {
public:

   size_t recvOffset(int n, const PVLayerLoc * loc);
   size_t sendOffset(int n, const PVLayerLoc * loc);

   static MPI_Datatype * newDatatypes(const PVLayerLoc * loc);
   static int freeDatatypes(MPI_Datatype * mpi_datatypes);

   Communicator(int argc, char ** argv, int nbatch);
   virtual ~Communicator();

   //int commInit(int * argc, char *** argv);
   //int commFinalize();

   char * name()                { return commName; }

   //Previous names of MPI getter functions now default to local ranks and sizes
   int commRank()                     { return localRank; }
   int globalCommRank()               { return globalRank; }
   int commSize()                     { return localSize; }
   int globalCommSize()               { return globalSize; }

   MPI_Comm communicator()       { return localIcComm; }
   MPI_Comm globalCommunicator()      { return globalIcComm; }

   int numberOfNeighbors(); // includes interior (self) as a neighbor
   int numberOfBorders()        {return numBorders;}

   bool hasNeighbor(int neighborId);
   int neighborIndex(int commId, int index);
   int reverseDirection(int commId, int direction);

   int commRow()          {return commRow(globalRank);}
   int commColumn()       {return commColumn(globalRank);}
   int commBatch()        {return commBatch(globalRank);}
   int numCommRows()      {return numRows;}
   int numCommColumns()   {return numCols;}
   int numCommBatches()   {return batchWidth;}

   int exchange(pvdata_t * data,
                const MPI_Datatype neighborDatatypes [],
                const PVLayerLoc * loc);

   int getTag(int neighbor) { return tags[neighbor]; }
   int getReverseTag(int neighbor) { return tags[reverseDirection(localRank, neighbor)]; }
   double fprintTime(FILE * fp) {return exchange_timer->fprint_time(fp);}

protected:

   int commRow(int commId);
   int commColumn(int commId);
   int commBatch(int commId);
   int commIdFromRowColumn(int commRow, int commColumn);

   int numNeighbors;  // # of remote neighbors plus local
   int numBorders;    // # of border regions (no communicating neighbor)

   //TODO - can this be cleaned up?
   int borders[NUM_NEIGHBORHOOD-1];
   int neighbors[NUM_NEIGHBORHOOD];        // [0] is interior (local)
   int remoteNeighbors[NUM_NEIGHBORHOOD];
   int tags[NUM_NEIGHBORHOOD];             // diagonal communication needs a different tag from left/right or up/down communication.

private:

   int gcd(int a, int b);

#ifdef PV_USE_MPI
   int mpi_initialized_on_entry;
#endif // PV_USE_MPI
   int localRank;
   int localSize;
   int globalRank;
   int globalSize;
   int batchRank;
   int numRows;
   int numCols;
   int batchWidth;

   char commName[COMMNAME_MAXLENGTH];

   MPI_Comm    localIcComm;
   MPI_Comm    globalIcComm;
   MPI_Request requests[NUM_NEIGHBORHOOD-1];

   Timer * exchange_timer;

   // These methods are private for now, move to public as needed

   int neighborInit();

   bool hasNorthwesternNeighbor(int commRow, int commColumn);
   bool hasNorthernNeighbor(int commRow, int commColumn);
   bool hasNortheasternNeighbor(int commRow, int commColumn);
   bool hasWesternNeighbor(int commRow, int commColumn);
   bool hasEasternNeighbor(int commRow, int commColumn);
   bool hasSouthwesternNeighbor(int commRow, int commColumn);
   bool hasSouthernNeighbor(int commRow, int commColumn);
   bool hasSoutheasternNeighbor(int commRow, int commColumn);

   int northwest(int commRow, int commColumn);
   int north(int commRow, int commColumn);
   int northeast(int commRow, int commColumn);
   int west(int commRow, int commColumn);
   int east(int commRow, int commColumn);
   int southwest(int commRow, int commColumn);
   int south(int commRow, int commColumn);
   int southeast(int commRow, int commColumn);

};

} // namespace PV

#endif /* COMMUNICATOR_HPP_ */
