/*
 * AvConn.cpp
 *
 *  Created on: Oct 9, 2009
 *      Author: rasmussn
 */

#include "AvgConn.hpp"

#include <assert.h>

namespace PV {

AvgConn::AvgConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
                 int channel, HyPerConn * delegate)
{
   this->delegate = delegate;
   HyPerConn::initialize(name, hc, pre, post, channel);
   initialize();
}

AvgConn::~AvgConn()
{
   free(avgActivity);
}

int AvgConn::initialize()
{
   const int numItems = pre->clayer->activity->numItems;
   const int datasize = numItems * sizeof(pvdata_t);

   avgActivity = (PVLayerCube *) calloc(sizeof(PVLayerCube*) + datasize, sizeof(char));
   avgActivity->loc = pre->clayer->loc;
   avgActivity->numItems = numItems;
   avgActivity->size = datasize;

   pvcube_setAddr(avgActivity);

   return 0;
}

PVPatch ** AvgConn::initializeWeights(PVPatch ** patches,
                                      int numPatches, const char * filename)
{
   if (delegate != this) {
      // using weights from delegate so can free weight memory
      if (patches != NULL) {
         for (int k = 0; k < numPatches; k++) {
            pvpatch_inplace_delete(patches[k]);
         }
         free(patches);
         patches = NULL;
      }
   }

   return NULL;
}

int AvgConn::deliver(Publisher * pub, PVLayerCube * cube, int neighbor)
{
   // update average values

   DataStore* store = pub->dataStore();

   const int numActive = pre->clayer->numExtended;
   const int numLevels = store->numberOfLevels();
   const int lastLevel = store->lastLevelIndex();

   pvdata_t * activity = pre->clayer->activity->data;
   pvdata_t * avg  = avgActivity->data;
   pvdata_t * last = (pvdata_t*) store->buffer(LOCAL, lastLevel);

   pvdata_t max = 0;
   for (int k = 0; k < numActive; k++) {
      pvdata_t oldVal = last[k];
      pvdata_t newVal = activity[k];
      avg[k] += (newVal - oldVal) / numLevels;
      if (max < avg[k]) max = avg[k];
   }

   if (max > 1) {
      pvdata_t scale = 1.0f/max;
      printf("AvgConn::deliver: rescaling: max==%f scale==%f\n", max, scale);
      for (int k = 0; k < numActive; k++) {
         avg[k] = scale * avg[k];
      }
   }

   post->recvSynapticInput(this, avgActivity, neighbor);

   return 0;
}

int AvgConn::createAxonalArbors()
{
   pvdata_t * phi_base = post->clayer->phi[channel];
   pvdata_t * del_phi_base = delegate->postSynapticLayer()->clayer->phi[channel];

   const int numAxons = numAxonalArborLists;

   for (int n = 0; n < numAxons; n++) {
      int numArbors = numWeightPatches(n);
      axonalArborList[n] = (PVAxonalArbor*) calloc(numArbors, sizeof(PVAxonalArbor));
      assert(axonalArborList[n] != NULL);
   }

   for (int n = 0; n < numAxons; n++) {
      int numArbors = numWeightPatches(n);
      PVPatch * dataPatches = (PVPatch *) calloc(numArbors, sizeof(PVPatch));
      assert(dataPatches != NULL);

      for (int kex = 0; kex < numArbors; kex++) {
         PVAxonalArbor * arbor = axonalArbor(kex, n);

         PVAxonalArbor * del_arbor = delegate->axonalArbor(kex, LOCAL);

         dataPatches[kex] = *del_arbor->data;
         arbor->data = &dataPatches[kex];

         // use same offsets as delegate
         size_t offset = del_arbor->data->data - del_phi_base;
         arbor->data->data = phi_base + offset;
         arbor->offset = del_arbor->offset;

         // use weights of delegate
         arbor->weights = del_arbor->weights;  // use weights of delegate

         // no STDP
         arbor->plasticIncr = NULL;

      } // loop over arbors (pre-synaptic neurons)
   } // loop over axons

   return 0;
}

int AvgConn::write(const char * filename)
{
   return 0;
}

} // namespace PV
