/*
 * InterColComm.h
 *
 *  Created on: Aug 28, 2008
 *      Author: rasmussn
 */

#ifndef INTERCOLCOMM_HPP_
#define INTERCOLCOMM_HPP_

#include "Communicator.hpp"
#include "columns/DataStore.hpp"
#include "columns/Publisher.hpp"
#include "include/pv_common.h"

// maximum number of messages (each layer publishes to all neighbors)
#define MAX_MESSAGES    MAX_NEIGHBORS
// #define MAX_PUBLISHERS  MAX_LAYERS
// #define MAX_SUBSCRIBERS MAX_LAYERS

namespace PV {

class HyPerLayer;
class BaseConnection;

class InterColComm : public Communicator {

public:
   InterColComm(PV_Arguments * argumentList);
   virtual ~InterColComm();

   int addPublisher(HyPerLayer * pub);
   int clearPublishers();
   int publish(HyPerLayer * pub, PVLayerCube * cube);
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
