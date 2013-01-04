/*
 * MaxPooling.cpp
 *
 *  Created on: Jan 3, 2013
 *      Author: gkenyon
 */

#include "MaxPooling.hpp"

namespace PV {

MaxPooling::MaxPooling()
{
   initialize_base();
}

MaxPooling::MaxPooling(const char * name, HyPerCol * hc, int numChannels)
{
   initialize_base();
   initialize(name, hc, numChannels);
}

MaxPooling::MaxPooling(const char * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc, 1);
}

MaxPooling::~MaxPooling()
{
}

int MaxPooling::initialize_base(){
   return PV_SUCCESS;
}

int MaxPooling::initialize(const char * name, HyPerCol * hc, int numChannels)
{
   return HyPerLayer::initialize(name, hc, numChannels);
}

int MaxPooling::recvSynapticInput(HyPerConn * conn, const PVLayerCube * activity,
      int arborID)
{
   recvsyn_timer->start();

   const int numExtended = activity->numItems;
   for (int kPre = 0; kPre < numExtended; kPre++) {
      float a = activity->data[kPre];
      if (a == 0.0f) continue;  // TODO - assume activity is sparse so make this common branch

      PVPatch * weights = conn->getWeights(kPre, arborID);
      int nk  = conn->fPatchSize() * weights->nx;
      int ny  = weights->ny;
      int sy  = conn->getPostNonextStrides()->sy;       // stride in layer
      int syw = conn->yPatchStride();                   // stride in patch
      pvdata_t * gSynPatchStart = conn->getGSynPatchStart(kPre, arborID);
      pvdata_t * data = conn->get_wData(arborID,kPre);
      for (int y = 0; y < ny; y++) {
         pvpatch_max_pooling(nk, gSynPatchStart + y*sy, a, data + y*syw);
//       if (err != 0) printf("  ERROR kPre = %d\n", kPre);
      }
   }

   recvsyn_timer->stop();

   return PV_SUCCESS;
}

} // namespace PV
