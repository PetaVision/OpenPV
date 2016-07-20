/*
 * InterColComm.cpp
 *
 *  Created on: Aug 28, 2008
 *      Author: rasmussn
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "InterColComm.hpp"
#include "layers/HyPerLayer.hpp"

namespace PV {

InterColComm::InterColComm(PV_Arguments * argumentList) : Communicator(argumentList)
{
   numPublishers = 0;
   publisherArraySize = INITIAL_PUBLISHER_ARRAY_SIZE;
   publishers = (Publisher **) malloc( publisherArraySize * sizeof(Publisher *) );
   for (int i = 0; i < publisherArraySize; i++) {
      publishers[i] = NULL;
   }
}

InterColComm::~InterColComm()
{
   clearPublishers();
   free(publishers); publishers = NULL;
}

int InterColComm::addPublisher(HyPerLayer* pub)
{
   int numItems = pub->getNumExtended();
   int numLevels = pub->getNumDelayLevels();
   int isSparse = pub->getSparseFlag();
   int pubId = pub->getLayerId();
   if( pubId >= publisherArraySize) {
      int status = resizePublishersArray(pubId+1);
      assert(status == EXIT_SUCCESS);
   }
   publishers[pubId] = new Publisher(pub->getParent()->icCommunicator(), numItems, pub->clayer->loc, numLevels, isSparse);
   numPublishers += 1;

   return pubId;
}

int InterColComm::clearPublishers() {
   for (int i=0; i<numPublishers; i++) {
      delete publishers[i]; publishers[i] = NULL;
   }
   numPublishers = 0;
   return PV_SUCCESS;
}

int InterColComm::resizePublishersArray(int newSize) {
   /* If newSize is greater than the existing size publisherArraySize,
    * create a new array of size newSize, and copy over the existing
    * publishers.  publisherArraySize is updated, to equal newSize.
    * If newSize <= publisherArraySize, do nothing
    * Returns PV_SUCCESS if resizing was successful
    * (or not needed; i.e. if newSize<=publisherArraySize)
    * Returns PV_FAILURE if unable to allocate a new array; in this
    * (unlikely) case, publishers and publisherArraySize are unchanged.
    */
   if( newSize > publisherArraySize ) {
      Publisher ** newPublishers = (Publisher **) malloc( newSize * sizeof(Publisher *) );
      if( newPublishers == NULL) return PV_FAILURE;
      for( int k=0; k< publisherArraySize; k++ ) {
         newPublishers[k] = publishers[k];
      }
      for( int k=publisherArraySize; k<newSize; k++) {
          newPublishers[k] = NULL;
      }
      free(publishers);
      publishers = newPublishers;
      publisherArraySize = newSize;
   }
   return PV_SUCCESS;
}

int InterColComm::subscribe(BaseConnection* conn)
{
   int pubId = conn->preSynapticLayer()->getLayerId();
   assert( pubId < publisherArraySize && pubId >= 0);
   return publishers[pubId]->subscribe(conn);
}

int InterColComm::publish(HyPerLayer* pub, PVLayerCube* cube)
{
   int pubId = pub->getLayerId();
   return publishers[pubId]->publish(pub, neighbors, numNeighbors, borders, numBorders, cube);
}

int InterColComm::exchangeBorders(int pubId, const PVLayerLoc * loc, int delay/*default=0*/) {
   int status = publishers[pubId]->exchangeBorders(neighbors, numNeighbors, loc, delay);
   return status;
}

int InterColComm::updateAllActiveIndices(int pubId){
   int status = publishers[pubId]->updateAllActiveIndices();
   return status;
}

int InterColComm::updateActiveIndices(int pubId){
   int status = publishers[pubId]->updateActiveIndices();
   return status;
}

/**
 * wait until all outstanding published messages have arrived
 */
int InterColComm::wait(int pubId)
{
   return publishers[pubId]->wait();
}

} // end namespace PV
