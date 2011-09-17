/*
 * DataStore.h
 *
 *  Created on: Sep 10, 2008
 *      Author: Craig Rasmussen
 */

#ifndef DATASTORE_H_
#define DATASTORE_H_

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
   DataStore(int numBuffers, size_t size, int numLevels);
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

private:
   size_t bufSize;
   int    curLevel;
   int    numLevels;
   int    numBuffers;
   char*  recvBuffers;

#ifdef PV_USE_OPENCL
   CLBuffer * clRecvBuffers;
public:
   int initializeThreadBuffers(HyPerCol * hc);
   CLBuffer * getCLBuffer()  {return clRecvBuffers;}
#endif
};

} // NAMESPACE

#endif /* DATASTORE_H_ */
