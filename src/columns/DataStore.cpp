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
//#ifdef PV_USE_OPENCL
//DataStore::DataStore(HyPerCol * hc, int numBuffers, size_t bufSize, int numLevels, bool copydstoreflag)
//#else
DataStore::DataStore(HyPerCol * hc, int numBuffers, int numItems, size_t dataSize, int numLevels, bool isSparse_flag)
//#endif // PV_USE_OPENCL
{
   assert(numLevels > 0 && numBuffers > 0);
   this->curLevel = numLevels - 1;  // start at bottom, work up
   this->numItems = numItems;
   this->dataSize = dataSize;
   this->bufSize = numItems * dataSize;
   this->numLevels = numLevels;
   this->numBuffers = numBuffers;
   this->recvBuffers = (char*) calloc(numBuffers * numLevels * numItems * dataSize, sizeof(char));
   this->isSparse_flag = isSparse_flag;
   if(this->isSparse_flag){
      this->activeIndicies = (unsigned int*) calloc(numBuffers * numLevels * numItems, sizeof(unsigned int));
      this->numActive = (unsigned int*) calloc(numBuffers * numLevels, sizeof(unsigned int));
   }
   else{
      this->activeIndicies = NULL;
      this->numActive = NULL;
   }
   assert(this->recvBuffers != NULL);

//#ifdef PV_USE_OPENCL
//   if(copydstoreflag) initializeThreadBuffers(hc);
//   else clRecvBuffers=NULL;
//#endif // PV_USE_OPENCL
}

DataStore::~DataStore()
{
//#ifdef PV_USE_OPENCL
//   if (clRecvBuffers != NULL) delete clRecvBuffers;
//   clRecvBuffers=NULL;
//#endif // PV_USE_OPENCL

   free(recvBuffers);
}

//#ifdef PV_USE_OPENCL
//int DataStore::initializeThreadBuffers(HyPerCol * hc)
//{
//   const size_t size = numBuffers * numLevels * bufSize * sizeof(char);
//   clRecvBuffers = hc->getCLDevice()->createBuffer(CL_MEM_COPY_HOST_PTR, size, recvBuffers);
//   numWait=0;
//   return PV_SUCCESS;
//}
//int DataStore::copyBufferToDevice() {
//   numWait++;
//   return clRecvBuffers->copyToDevice(&evCopyDataStore);
//}
//int DataStore::waitForCopy() {
//   int status=PV_SUCCESS;
//   if(numWait>0) {
//      status |= clWaitForEvents(numWait, &evCopyDataStore);
//      clReleaseEvent(evCopyDataStore);
//      numWait=0;
//   }
//   return status;
//}
//#endif // PV_USE_OPENCL

}
