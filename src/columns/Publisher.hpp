/*
 * Publisher.hpp
 *
 *  Created on: Jul 19, 2016
 *      Author: pschultz
 */

#ifndef PUBLISHER_HPP_
#define PUBLISHER_HPP_

#include "arch/mpi/mpi.h"
#include "checkpointing/Checkpointer.hpp"
#include "columns/DataStore.hpp"
#include "include/PVLayerLoc.h"
#include "include/pv_types.h"
#include "structures/MPIBlock.hpp"
#include "utils/BorderExchange.hpp"

namespace PV {

class Publisher {

  public:
   Publisher(MPIBlock const &mpiBlock, PVLayerCube *cube, int numLevels, bool isSparse);
   virtual ~Publisher();

   void
   checkpointDataStore(Checkpointer *checkpointer, char const *objectName, char const *bufferName);

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
   int isExchangeFinished(int delay = 0);

   /**
    * creates a PVLayerCube pointing to the data in the data store at the given delay.
    * This method blocks until any pending border exchange for that delay level are completed.
    */
   PVLayerCube createCube(int delay = 0);

   int wait(int delay = 0);

   void increaseTimeLevel();

   void updateAllActiveIndices();
   void updateActiveIndices(int delay = 0);

  private:
   float *recvBuffer(int bufferId) { return store->buffer(bufferId); }
   float *recvBuffer(int bufferId, int delay) { return store->buffer(bufferId, delay); }

   long *recvNumActiveBuffer(int bufferId) { return store->numActiveBuffer(bufferId); }
   long *recvNumActiveBuffer(int bufferId, int delay) {
      return store->numActiveBuffer(bufferId, delay);
   }

   SparseList<float>::Entry *recvActiveIndicesBuffer(int bufferId) {
      return store->activeIndicesBuffer(bufferId);
   }
   SparseList<float>::Entry *recvActiveIndicesBuffer(int bufferId, int delay) {
      return store->activeIndicesBuffer(bufferId, delay);
   }

   DataStore *store;

   PVLayerCube *mLayerCube;

   BorderExchange *mBorderExchanger = nullptr;

   RingBuffer<std::vector<MPI_Request>> *mpiRequestsBuffer = nullptr;
   // std::vector<MPI_Request> requests;
   MPI_Datatype *neighborDatatypes;
};

} /* namespace PV */

#endif /* PUBLISHER_HPP_ */
