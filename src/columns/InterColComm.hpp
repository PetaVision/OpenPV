/*
 * InterColComm.h
 *
 *  Created on: Aug 28, 2008
 *      Author: rasmussn
 */

#ifndef INTERCOLCOMM_H_
#define INTERCOLCOMM_H_

#include "DataStore.hpp"
#include "../include/pv_common.h"

#include <mpi.h>

// maximum number of messages (each layer publishes to all neighbors)
#define MAX_MESSAGES    MAX_NEIGHBORS
#define MAX_PUBLISHERS  MAX_LAYERS
#define MAX_SUBSCRIBERS MAX_LAYERS

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

#include "../layers/PVLayer.h"

namespace PV {

class HyPerCol;
class HyPerLayer;
class HyPerConn;

class Publisher {
public:
   Publisher(int pubId, int numType1, size_t size1, int numType2, size_t size2, int numLevels);
   virtual ~Publisher();

   int publish(HyPerLayer* pub,
               int neighbors[], int numNeighbors,
               int borders[], int numBorders, PVLayerCube* data);
   int subscribe(HyPerConn* conn);
   int deliver(HyPerCol* hc, int numNeighbors, int numBorders);

   static int borderStoreIndex(int i, int numNeighbors)  {return i+numNeighbors;}

   int increaseTimeLevel()  {return store->newLevelIndex();}

   DataStore* dataStore()   {return store;}

private:

   PVLayerCube* recvBuffer(int neighborId)
         {return (PVLayerCube*) store->buffer(neighborId);}
   PVLayerCube* recvBuffer(int neighborId, int delay)
         {return (PVLayerCube*) store->buffer(neighborId, delay);}

   int pubId;
   int numSubscribers;
   HyPerConn* connection[MAX_SUBSCRIBERS];
   MPI_Request request[MAX_MESSAGES];
   DataStore* store;
};

class InterColComm {
public:
   InterColComm(int* argc, char*** argv, HyPerCol* col);
   virtual ~InterColComm();

   int commInit(int* argc, char*** argv);
   int commFinalize();

   int addPublisher(HyPerLayer* pub, size_t size1, size_t size2, int numLevels);
   int publish(HyPerLayer* pub, PVLayerCube* cube);
   int subscribe(HyPerConn* conn);
   int deliver(int pubId);
   int increaseTimeLevel(int pubId)       {return publishers[pubId]->increaseTimeLevel();}

   int rank()
   {
      return commRank;
   }

   DataStore* publisherStore(int pubId)   {return publishers[pubId]->dataStore();}

   int numberOfNeighbors();
   int numberOfBorders()                  {return numBorders;}

   int commRow(int commId);
   int commColumn(int commId);
   int numCommRows()      {return numRows;}
   int numCommColumns()   {return numCols;}

private:

   HyPerCol* hc; // the column that "owns" this communicator
   int commRank;
   int commSize;
   int numRows;
   int numCols;
   int numHyPerCols;
   int numNeighbors;
   int numBorders;
   int numPublishers;

   int neighbors[MAX_NEIGHBORS + 1];
   int borders[MAX_NEIGHBORS];
   Publisher* publishers[MAX_PUBLISHERS];

   // These methods are private for now, move to public as needed

   int neighborInit();

   bool hasWesternNeighbor(int commId);
   bool hasEasternNeighbor(int commId);
   bool hasNorthernNeighbor(int commId);
   bool hasSouthernNeighbor(int commId);

   int numberNeighbors();

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

}

#endif /* INTERCOLCOMM_H_ */
