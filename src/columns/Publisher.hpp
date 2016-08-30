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

class Publisher {

public:
   Publisher(Communicator * comm, PVLayerCube * cube, int numLevels, bool isSparse);
   virtual ~Publisher();
   int publish(double currentTime, double lastUpdateTime);
   int exchangeBorders(const PVLayerLoc * loc, int delay=0);
   int wait();

   void increaseTimeLevel()   {store->newLevelIndex();}

   DataStore * dataStore()   {return store;}

   int updateAllActiveIndices();
   int updateActiveIndices();

private:

   pvdata_t * recvBuffer(int bufferId)
         {return store->buffer(bufferId);}
   pvdata_t * recvBuffer(int bufferId, int delay)
         {return store->buffer(bufferId, delay);}

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

   DataStore * store;

   PVLayerCube * mLayerCube;

   Communicator * mComm;

   std::vector<MPI_Request>  requests;
   MPI_Datatype * neighborDatatypes;
};

} /* namespace PV */

#endif /* PUBLISHER_HPP_ */
