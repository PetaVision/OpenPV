/*
 * Communicator.hpp
 */

#ifndef COMMUNICATOR_HPP_
#define COMMUNICATOR_HPP_

#include <cstdio>
#include <vector>
#include "PV_Arguments.hpp"
#include "../include/pv_arch.h"
#include "../include/pv_types.h"
#include "../utils/Timer.hpp"

#include "../arch/mpi/mpi.h"

#define COMMNAME_MAXLENGTH 16

namespace PV {

class Communicator {
public:

   size_t recvOffset(int n, const PVLayerLoc * loc);
   size_t sendOffset(int n, const PVLayerLoc * loc);

   static MPI_Datatype * newDatatypes(const PVLayerLoc * loc);
   static int freeDatatypes(MPI_Datatype * mpi_datatypes);

   Communicator(PV_Arguments * argumentList);
   virtual ~Communicator();


   char * name()                { return commName; }

   //Previous names of MPI getter functions now default to local ranks and sizes
   int commRank()                     { return localRank; }
   int globalCommRank()               { return globalRank; }
   int commSize()                     { return localSize; }
   int globalCommSize()               { return globalSize; }

   MPI_Comm communicator()       { return localIcComm; }
   MPI_Comm globalCommunicator()      { return globalIcComm; }

   int numberOfNeighbors(); // includes interior (self) as a neighbor

   bool hasNeighbor(int neighborId);
   int neighborIndex(int commId, int index);
   int reverseDirection(int commId, int direction);

   int commRow()          {return commRow(localRank);}
   int commColumn()       {return commColumn(localRank);}
   int commBatch()        {return commBatch(globalRank);}
   int numCommRows()      {return numRows;}
   int numCommColumns()   {return numCols;}
   int numCommBatches()   {return batchWidth;}

   int exchange(pvdata_t * data,
                const MPI_Datatype neighborDatatypes [],
                const PVLayerLoc * loc, std::vector<MPI_Request> & req);
   int wait(std::vector<MPI_Request> & req);

   int getTag(int neighbor) { return tags[neighbor]; }
   int getReverseTag(int neighbor) { return tags[reverseDirection(localRank, neighbor)]; }

   bool isExtraProc(){return isExtra;}

   static const int LOCAL      = 0;
   static const int NORTHWEST  = 1;
   static const int NORTH      = 2;
   static const int NORTHEAST  = 3;
   static const int WEST       = 4;
   static const int EAST       = 5;
   static const int SOUTHWEST  = 6;
   static const int SOUTH      = 7;
   static const int SOUTHEAST  = 8;



protected:

   int commRow(int commId);
   int commColumn(int commId);
   int commBatch(int commId);
   int commIdFromRowColumn(int commRow, int commColumn);

   int numNeighbors;  // # of remote neighbors plus local.  NOT the size of the neighbors array, which uses negative values to mark directions where there is no remote neighbor.

   int isExtra; //Defines if the process is an extra process

   int neighbors[NUM_NEIGHBORHOOD];        // [0] is interior (local)
   int remoteNeighbors[NUM_NEIGHBORHOOD];
   int tags[NUM_NEIGHBORHOOD];             // diagonal communication needs a different tag from left/right or up/down communication.

private:

   
   int gcd(int a, int b);

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
