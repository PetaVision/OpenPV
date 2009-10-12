/*
 * AvConn.cpp
 *
 *  Created on: Oct 9, 2009
 *      Author: rasmussn
 */

#include "AvgConn.hpp"

namespace PV {

AvgConn::AvgConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
                 int channel, HyPerConn * delegate)
{
   HyPerConn::initialize(name, hc, pre, post, channel);
   initialize(delegate);
}

AvgConn::~AvgConn()
{
   free(avgActivity);
}

int AvgConn::initialize(HyPerConn * delegate)
{
   this->delegate = delegate;

   const int numItems = pre->clayer->activity->numItems;
   const int datasize = numItems * sizeof(pvdata_t);

   avgActivity = (PVLayerCube *) calloc(sizeof(PVLayerCube*) + datasize, sizeof(char));
   avgActivity->loc = pre->clayer->loc;
   avgActivity->numItems = numItems;
   avgActivity->size = datasize;
//   avgActivity->data = (pvdata_t *) ((char*) avgActivity + sizeof(PVLayerCube));

   pvcube_setAddr(avgActivity);

   return 0;
}

int AvgConn::deliver(Publisher * pub, PVLayerCube * cube, int neighbor)
{
   // need to update average values

   DataStore* store = pub->dataStore();

   const int numActive = pre->clayer->numExtended;
   const int numLevels = store->numberOfLevels();
   const int lastLevel = store->lastLevelIndex();

   pvdata_t * activity = pre->clayer->activity->data;
   pvdata_t * avg  = avgActivity->data;
   pvdata_t * last = (pvdata_t*) store->buffer(LOCAL, lastLevel);

   for (int k = 0; k < numActive; k++) {
      pvdata_t oldVal = last[k];
      pvdata_t newVal = activity[k];
      avg[k] += (newVal - oldVal) / numLevels;
   }

   if (delegate != NULL) {
      post->recvSynapticInput(delegate, avgActivity, neighbor);
   }

//   if (stdpFlag) {
//      updateWeights(cube, neighbor);
//   }

   return 0;
}

int AvgConn::write(const char * filename)
{
   return 0;
}

} // namespace PV
