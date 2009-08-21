/*
 * DataStore.h
 *
 *  Created on: Sep 10, 2008
 *      Author: rasmussn
 */

#ifndef DATASTORE_H_
#define DATASTORE_H_

#include <stdlib.h>

namespace PV
{

class DataStore
{
public:
   DataStore(int numBuffers, size_t size, int numLevels);
   virtual ~DataStore();

   size_t size()         {return bufSize;}

   int numberOfLevels()  {return numLevels;}
   int numberOfBuffers() {return numBuffers;}
   int levelIndex(int level)
         {return ((level + curLevel) % numLevels);}
   int newLevelIndex()
         {return (curLevel = (numLevels + curLevel - 1) % numLevels);}
   void* buffer(int bufferId, int level)
         {return (recvBuffers + bufferId*numLevels*bufSize + levelIndex(level)*bufSize);}
   void* buffer(int bufferId)
         {return (recvBuffers + bufferId*numLevels*bufSize + curLevel*bufSize);}

private:
   size_t bufSize;
   int    curLevel;
   int    numLevels;
   int    numBuffers;
   char*  recvBuffers;
};

} // NAMESPACE

#endif /* DATASTORE_H_ */
