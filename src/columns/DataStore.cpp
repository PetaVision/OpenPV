/*
 * DataStore.cpp
 *
 *  Created on: Sep 10, 2008
 *      Author: rasmussn
 */

#include "DataStore.hpp"

#include <stdlib.h>

namespace PV
{

DataStore::DataStore(int numBuffers, size_t bufSize, int numLevels)
{
	this->curLevel = numLevels - 1;  // start at bottom, work up
	this->bufSize = bufSize;
	this->numLevels = numLevels;
	this->numBuffers = numBuffers;
	this->recvBuffers = (char*) calloc(numBuffers * numLevels * bufSize, sizeof(char));
}

DataStore::~DataStore()
{
	free(recvBuffers);
}

}
