/*
 * DataStore.cpp
 *
 *  Created on: Sep 10, 2008
 *      Author: Craig Rasmussen
 */

#include "DataStore.hpp"
#include "include/pv_common.h"
#include "utils/PVLog.hpp"

#include <assert.h>
#include <stdlib.h>
#include <limits>

namespace PV
{

/**
 * @numBuffers
 * @bufSize
 * @numLevels
 */
DataStore::DataStore(int numBuffers, int numItems, size_t dataSize, int numLevels, bool isSparse_flag)
{
   assert(numLevels > 0 && numBuffers > 0);
   this->curLevel = numLevels - 1;  // start at bottom, work up
   this->numItems = numItems;
   this->dataSize = dataSize;
   this->bufSize = numItems * dataSize;
   this->numLevels = numLevels;
   this->numBuffers = numBuffers;
   this->recvBuffers = (char*) calloc(numBuffers * numLevels * numItems * dataSize, sizeof(char));
   if (this->recvBuffers==NULL) {
      pvError().printf("DataStore unable to allocate data buffer for %d items, %d buffers and %d levels: %s\n", numItems, numBuffers, numLevels, strerror(errno));
   }
   this->lastUpdateTimes = (double *) malloc(numBuffers * numLevels * sizeof(double));
   if (this->lastUpdateTimes==NULL) {
      pvError().printf("DataStore unable to allocate lastUpdateTimes buffer for %d buffers and %d levels: %s\n", numBuffers, numLevels, strerror(errno));
   }
   double infvalue = std::numeric_limits<double>::infinity();
   for (int lvl=0; lvl<numLevels*numBuffers; lvl++) {
      lastUpdateTimes[lvl] = -infvalue;
   }

   this->isSparse_flag = isSparse_flag;
   if(this->isSparse_flag){
      this->activeIndices = (unsigned int*) calloc(numBuffers * numLevels * numItems, sizeof(unsigned int));
      if (this->activeIndices==NULL) {
         pvError().printf("DataStore unable to allocate activeIndices buffer for %d items, %d buffers and %d levels: %s\n", numItems, numBuffers, numLevels, strerror(errno));
      }
      this->numActive = (long *) calloc(numBuffers * numLevels, sizeof(long));
      if (this->numActive==NULL) {
         pvError().printf("DataStore unable to allocate numActive buffer for %d buffers and %d levels: %s\n", numBuffers, numLevels, strerror(errno));
      }
   }
   else{
      this->activeIndices = NULL;
      this->numActive = NULL;
   }
}

DataStore::~DataStore()
{
   free(recvBuffers);
   free(activeIndices);
   free(numActive);
   free(lastUpdateTimes);
}

}
