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
   publishers.reserve(INITIAL_PUBLISHER_ARRAY_SIZE);
}

InterColComm::~InterColComm()
{
   clearPublishers();
}

int InterColComm::addPublisher(HyPerLayer* pub)
{
   Communicator * icComm = pub->getParent()->icCommunicator();
   int numItems = pub->getNumExtended();
   int numLevels = pub->getNumDelayLevels();
   int isSparse = pub->getSparseFlag();
   int pubId = pub->getLayerId();
   publishers.emplace_back(new Publisher(icComm, numItems, pub->clayer->loc, numLevels, isSparse));

   return pubId;
}

int InterColComm::clearPublishers() {
   for (auto& p : publishers) { delete p; } publishers.clear();
   return PV_SUCCESS;
}

int InterColComm::publish(HyPerLayer* pub, PVLayerCube* cube)
{
   int pubId = pub->getLayerId();
   return publishers[pubId]->publish(pub->getParent()->simulationTime(), pub->getLastUpdateTime(), neighbors, numNeighbors, cube);
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
