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

namespace PV
{
class HyPerCol;

class DataStore
{
public:
   DataStore(HyPerCol * hc, int numBuffers, int numItems, size_t dataSize, int numLevels, bool isSparse);

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

   //Levels (delays) spins slower than bufferId (batches)

   void* buffer(int bufferId, int level)
         {return (recvBuffers + levelIndex(level)*numBuffers*bufSize + bufferId * bufSize);}
   void* buffer(int bufferId)
         {return (recvBuffers + curLevel*numBuffers*bufSize + bufferId * bufSize);}

   double getLastUpdateTime(int bufferId, int level) 
      { return lastUpdateTimes[levelIndex(level) * numBuffers + bufferId]; }
   double getLastUpdateTime(int bufferId) 
      { return lastUpdateTimes[levelIndex(0) * numBuffers + bufferId]; }

   void setLastUpdateTime(int bufferId, int level, double t) 
      { lastUpdateTimes[levelIndex(level)*numBuffers + bufferId] = t; }
   void setLastUpdateTime(int bufferId, double t) 
      { lastUpdateTimes[levelIndex(0)*numBuffers + bufferId] = t; }

   size_t bufferOffset(int bufferId, int level=0)
         {return (levelIndex(level)*numBuffers*bufSize + bufferId*bufSize);}
   bool isSparse() {return isSparse_flag;}

   unsigned int* activeIndicesBuffer(int bufferId, int level)
         {return (activeIndices + levelIndex(level)*numBuffers*numItems + bufferId*numItems);}

   unsigned int* activeIndicesBuffer(int bufferId){
      return (activeIndices + curLevel*numBuffers*numItems + bufferId*numItems);
   }

   long * numActiveBuffer(int bufferId, int level){
      //return (numActive + bufferId*numLevels + levelIndex(level));
      return (numActive + levelIndex(level)*numBuffers + bufferId);
   }

   long * numActiveBuffer(int bufferId){
      return (numActive + curLevel*numBuffers + bufferId);
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
   long *   numActive;
   bool  isSparse_flag;
   double * lastUpdateTimes; // A ring buffer for the getLastUpdateTime() function.
};

} // NAMESPACE

#endif /* DATASTORE_HPP_ */
