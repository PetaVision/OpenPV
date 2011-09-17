/*
 * DataStore.cpp
 *
 *  Created on: Sep 10, 2008
 *      Author: Craig Rasmussen
 */

#include "DataStore.hpp"
#include "HyPerCol.hpp"
#include "../include/pv_common.h"

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
#ifdef PV_USE_OPENCL
   if (clRecvBuffers != NULL) delete clRecvBuffers;
#endif

   free(recvBuffers);
}

#ifdef PV_USE_OPENCL
int DataStore::initializeThreadBuffers(HyPerCol * hc)
{
   const size_t size = numBuffers * numLevels * bufSize * sizeof(char);
   clRecvBuffers = hc->getCLDevice()->createBuffer(CL_MEM_COPY_HOST_PTR, size, recvBuffers);
   return PV_SUCCESS;
}
#endif

}
