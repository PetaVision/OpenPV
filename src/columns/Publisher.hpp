/*
 * Publisher.hpp
 *
 *  Created on: Jul 19, 2016
 *      Author: pschultz
 */

#ifndef PUBLISHER_HPP_
#define PUBLISHER_HPP_

#include "include/pv_datatypes.h"
#include "include/pv_types.h"
#include "include/PVLayerLoc.h"
#include "arch/mpi/mpi.h"
#include "columns/Communicator.hpp"
#include "columns/DataStore.hpp"

namespace PV {

class HyPerLayer;
class BaseConnection;

class Publisher {

public:
   Publisher(Communicator * comm, int numItems, PVLayerLoc loc, int numLevels, bool isSparse);
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

   Communicator * mComm;

   int            numRequests;
   MPI_Request *  requests;
   MPI_Datatype * neighborDatatypes;
};

} /* namespace PV */

#endif /* PUBLISHER_HPP_ */
