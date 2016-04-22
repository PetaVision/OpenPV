/*
 * InterColComm.h
 *
 *  Created on: Aug 28, 2008
 *      Author: rasmussn
 */

#ifndef INTERCOLCOMM_HPP_
#define INTERCOLCOMM_HPP_

#include "Communicator.hpp"
#include "DataStore.hpp"
#include "../include/pv_common.h"

// maximum number of messages (each layer publishes to all neighbors)
#define MAX_MESSAGES    MAX_NEIGHBORS
// #define MAX_PUBLISHERS  MAX_LAYERS
// #define MAX_SUBSCRIBERS MAX_LAYERS

namespace PV {

class HyPerCol;
class HyPerLayer;
class BaseConnection;

class Publisher {

public:
//#ifdef PV_USE_OPENCL
//   Publisher(int pubId, HyPerCol * hc, int numItems, PVLayerLoc loc, int numLevels, bool copydstoreflag);
//#else
   Publisher(int pubId, HyPerCol * hc, int numItems, PVLayerLoc loc, int numLevels, bool isSparse);
//#endif
   virtual ~Publisher();
   int readData(int delay);
   int publish(HyPerLayer * pub, int neighbors[], int numNeighbors,
               int borders[], int numBorders, PVLayerCube * data,
               int delay=0);
   int subscribe(BaseConnection * conn);
   int exchangeBorders(int neighbors[], int numNeighbors, const PVLayerLoc * loc, int delay=0);
   int wait();

   static int borderStoreIndex(int i, int numNeighbors)  {return i+numNeighbors;}

   int increaseTimeLevel()   {return store->newLevelIndex();}

   DataStore * dataStore()   {return store;}

   int updateAllActiveIndices();
   int updateActiveIndices();

private:

   pvdata_t * recvBuffer(int bufferId)
         {return (pvdata_t *) store->buffer(bufferId);}
   pvdata_t * recvBuffer(int bufferId, int delay)
         {return (pvdata_t *) store->buffer(bufferId, delay);}

   long * recvNumActiveBuffer(int bufferId){
      return store->numActiveBuffer(bufferId);
   }
   long * recvNumActiveBuffer(int bufferId, int delay){
      return store->numActiveBuffer(bufferId, delay);
   }

   unsigned int * recvActiveIndicesBuffer(int bufferId){
      return store->activeIndicesBuffer(bufferId);
   }
   unsigned int * recvActiveIndicesBuffer(int bufferId, int delay){
      return store->activeIndicesBuffer(bufferId, delay);
   }

   int calcAllActiveIndices();
   int calcActiveIndices();

   int pubId;
   int numSubscribers;
   int subscriberArraySize;
   BaseConnection ** connection;
   DataStore * store;

   PVLayerCube cube;

   Communicator * comm;

   int            numRequests;
   MPI_Request *  requests;
   MPI_Datatype * neighborDatatypes;
};

class InterColComm : public Communicator {

public:
   InterColComm(PV_Arguments * argumentList);
   virtual ~InterColComm();

   int addPublisher(HyPerLayer * pub);
   int clearPublishers();
   int publish(HyPerLayer * pub, PVLayerCube * cube);
   int subscribe(BaseConnection * conn);
   int exchangeBorders(int pubId, const PVLayerLoc * loc, int delay=0);
   int updateAllActiveIndices(int pubId);
   int updateActiveIndices(int pubId);
   int wait(int pubId);

   int increaseTimeLevel(int pubId)        {return publishers[pubId]->increaseTimeLevel();}

   DataStore * publisherStore(int pubId)   {return publishers[pubId]->dataStore();}

private:

   int numPublishers;
   int publisherArraySize;
   Publisher ** publishers;
   int resizePublishersArray(int newSize);
};

} // namespace PV

#endif /* INTERCOLCOMM_HPP_ */
