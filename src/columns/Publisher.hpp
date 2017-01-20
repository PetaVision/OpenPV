/*
 * Publisher.hpp
 *
 *  Created on: Jul 19, 2016
 *      Author: pschultz
 */

#ifndef PUBLISHER_HPP_
#define PUBLISHER_HPP_

#include "../arch/mpi/mpi.h"
#include "columns/Communicator.hpp"
#include "columns/DataStore.hpp"
#include "include/PVLayerLoc.h"
#include "include/pv_types.h"

namespace PV {

class Publisher {

  public:
   Publisher(Communicator *comm, PVLayerCube *cube, int numLevels, bool isSparse);
   virtual ~Publisher();

   /**
    * Copies the data from the cube to the top level of the data store, and exchanges
    * the border.
    */
   int publish(double lastUpdateTime);

   /**
    * Keeps the data store in sync if the time advances but the data doesn't change.
    * If the number of levels is greater than one, copy the previous level to the
    * current level; otherwise do nothing. Using this instead of publish() avoids
    * an unnecessary border exchange.
    */
   void copyForward(double lastUpdateTime);
   int exchangeBorders(const PVLayerLoc *loc, int delay = 0);
   int wait();

   void increaseTimeLevel() { store->newLevelIndex(); }

   DataStore *dataStore() { return store; }

   int updateAllActiveIndices();
   int updateActiveIndices(int delay=0);

  private:
   float *recvBuffer(int bufferId) { return store->buffer(bufferId); }
   float *recvBuffer(int bufferId, int delay) { return store->buffer(bufferId, delay); }

   long *recvNumActiveBuffer(int bufferId) { return store->numActiveBuffer(bufferId); }
   long *recvNumActiveBuffer(int bufferId, int delay) {
      return store->numActiveBuffer(bufferId, delay);
   }

   unsigned int *recvActiveIndicesBuffer(int bufferId) {
      return store->activeIndicesBuffer(bufferId);
   }
   unsigned int *recvActiveIndicesBuffer(int bufferId, int delay) {
      return store->activeIndicesBuffer(bufferId, delay);
   }

   int calcAllActiveIndices();
   int calcActiveIndices(int delay=0);

   DataStore *store;

   PVLayerCube *mLayerCube;

   Communicator *mComm;

   std::vector<MPI_Request> requests;
   MPI_Datatype *neighborDatatypes;
};

} /* namespace PV */

#endif /* PUBLISHER_HPP_ */
