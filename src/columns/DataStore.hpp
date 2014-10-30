/*
 * DataStore.hpp
 *
 *  Created on: Sep 10, 2008
 *      Author: Craig Rasmussen
 */

#ifndef DATASTORE_HPP_
#define DATASTORE_HPP_

#include "../include/pv_arch.h"
#include <stdlib.h>

#ifdef PV_USE_OPENCL
#include "../arch/opencl/CLBuffer.hpp"
#include "../arch/opencl/CLDevice.hpp"
#endif

namespace PV
{
class HyPerCol;

class DataStore
{
public:
//#ifdef PV_USE_OPENCL
//   DataStore(HyPerCol * hc, int numBuffers, size_t size, int numLevels, bool copydstoreflag);
//#else
   DataStore(HyPerCol * hc, int numBuffers, int numItems, size_t dataSize, int numLevels, bool isSparse);
//#endif

   virtual ~DataStore();

   size_t size()         {return bufSize;}

   int numberOfLevels()  {return numLevels;}
   int numberOfBuffers() {return numBuffers;}
   int lastLevelIndex()
         {return ((numLevels + curLevel + 1) % numLevels);}
   int levelIndex(int level)
         {return ((level + curLevel) % numLevels);}
   int newLevelIndex()
         {return (curLevel = (numLevels + curLevel - 1) % numLevels);}
   void* buffer(int bufferId, int level)
         {return (recvBuffers + bufferId*numLevels*bufSize + levelIndex(level)*bufSize);}
   void* buffer(int bufferId)
         {return (recvBuffers + bufferId*numLevels*bufSize + curLevel*bufSize);}
   size_t bufferOffset(int bufferId, int level=0)
         {return (bufferId*numLevels*bufSize + levelIndex(level)*bufSize);}
   bool isSparse() {return isSparse_flag;}

   unsigned int* activeIndicesBuffer(int bufferId, int level)
         {return (activeIndices + bufferId*numLevels*numItems + levelIndex(level)*numItems);}

   unsigned int* activeIndicesBuffer(int bufferId){
      return (activeIndices + bufferId*numLevels*numItems + curLevel*numItems);
   }

   unsigned int* numActiveBuffer(int bufferId, int level){
      return (numActive + bufferId*numLevels + levelIndex(level));
   }

   unsigned int* numActiveBuffer(int bufferId){
      return (numActive + bufferId*numLevels + curLevel);
   }

   int getNumItems(){ return numItems;}

private:
   size_t dataSize;
   int    numItems;
   size_t bufSize;
   int    curLevel;
   int    numLevels;
   int    numBuffers;
   char*  recvBuffers;

   unsigned int*   activeIndices;
   unsigned int*   numActive;
   bool  isSparse_flag;

//#ifdef PV_USE_OPENCL
//   CLBuffer * clRecvBuffers;
//   cl_event   evCopyDataStore;
//   int numWait;
//public:
//   int initializeThreadBuffers(HyPerCol * hc);
//   int copyBufferToDevice();
//   int waitForCopy();
//   CLBuffer * getCLBuffer() {return clRecvBuffers;}
//#endif
};

} // NAMESPACE

#endif /* DATASTORE_HPP_ */
