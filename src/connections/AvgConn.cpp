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
   this->delegate = (delegate == NULL) ? this : delegate;
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

   PVParams * params = parent->parameters();
   maxRate = params->value(name, "maxRate", 400);
   maxRate *= parent->getDeltaTime() / 1000;

   return 0;
}

PVPatch ** AvgConn::initializeWeights(PVPatch ** patches,
                                      int numPatches, const char * filename)
{
   // If I'm my own delegate, I need my own weights
   //
   if (delegate == this) {
      return HyPerConn::initializeWeights(patches, numPatches, filename);
   }

   // otherwise using weights from delegate so can free weight memory
   //
   if (patches != NULL) {
      for (int k = 0; k < numPatches; k++) {
         pvpatch_inplace_delete(patches[k]);
      }
      free(patches);
      patches = NULL;
   }

   return NULL;
}

/**
 * - PVLayerCube * cube has activity from pre-synaptic layer. This routine is delivering
 * the normalized average activity to the post-synaptic layer.
 * - numLevels is the length of the time window over which the average activity
 * of each neuron in the pre-synaptic layer is computed.
 * - numLevels is now set to be equal to MAX_F_DELAY in /src/include/pv_common.h
 */
int AvgConn::deliver(Publisher * pub, PVLayerCube * cube, int neighbor)
{
   // update average values

   DataStore* store = pub->dataStore();

   const int numActive = pre->clayer->numExtended;
   const int numLevels = store->numberOfLevels();
   const int lastLevel = store->lastLevelIndex();

   const float maxCount = maxRate * numLevels;

   pvdata_t * activity = pre->clayer->activity->data;
   pvdata_t * avg  = avgActivity->data;
   pvdata_t * last = (pvdata_t*) store->buffer(LOCAL, lastLevel);

   pvdata_t max = 0;
   for (int k = 0; k < numActive; k++) {
      pvdata_t oldVal = last[k];
      pvdata_t newVal = activity[k];
      avg[k] += newVal/maxCount - oldVal;
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
   // If I'm my own delegate, I need my own weights
   //
   if (delegate == this) {
      return HyPerConn::createAxonalArbors();
   }

   // otherwise just use weights from the delegate
   //
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

int AvgConn::write_patch_activity(FILE * fp, PVPatch * patch,
                                         const PVLayerLoc * loc, int kx0, int ky0, int kf0)
{
   int f, i, j;

   pvdata_t * avg  = avgActivity->data;

   const int nx = patch->nx;
   const int ny = patch->ny;
   const int nf = patch->nf;

   // these strides are from the layer, not the patch
   // NOTE: assumes nf from layer == nf from patch
   //
   const int sx = nf;
   const int sy = loc->nx * nf;

   assert(fp != NULL);

   const int k0 = kIndex(kx0, ky0, kf0, loc->nx, loc->ny, nf);

   fprintf(fp, "  ");

   // loop over patch indices (writing out layer indices and activity)
   //
   for (f = 0; f < nf; f++) {
      for (j = 0; j < ny; j++) {
         for (i = 0; i < nx; i++) {
            int kf = f;
            int kx = kx0 + i;
            int ky = ky0 + j;
            int k  = k0 + kf + i*sx + j*sy;
            fprintf(fp, "(%4d, (%4d,%4d,%4d) %f) ", k, kx, ky, kf, avg[k]);
         }
         fprintf(fp, "\n  ");
      }
      fprintf(fp, "\n");
   }

   return 0;
}



} // namespace PV
