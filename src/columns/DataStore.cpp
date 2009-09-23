/*
 * DataStore.cpp
 *
 *  Created on: Sep 10, 2008
 *      Author: rasmussn
 */

#include "DataStore.hpp"
#include <assert.h>

#include <stdlib.h>

namespace PV
{

/**
 * @numBuffers
 * @bufSize
 * @numLevels
 */
DataStore::DataStore(int numBuffers, size_t bufSize, int numLevels)
{
	this->curLevel = numLevels - 1;  // start at bottom, work up
	this->bufSize = bufSize;
	this->numLevels = numLevels;
	this->numBuffers = numBuffers;
	this->recvBuffers = (char*) calloc(numBuffers * numLevels * bufSize, sizeof(char));
	assert(this->recvBuffers != NULL);
}

DataStore::~DataStore()
{
	free(recvBuffers);
}

}
