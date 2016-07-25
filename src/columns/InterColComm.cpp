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
#include <algorithm>

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

void InterColComm::addPublisher(Publisher * pub)
{
   publishers.emplace_back(pub);
}

void InterColComm::removePublisher(Publisher * pub) {
   if (publishers.size()>0) {
   publishers.erase(std::remove(publishers.begin(), publishers.end(), pub));
   }
}

int InterColComm::clearPublishers() {
   publishers.clear();
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
