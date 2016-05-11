/*
 * MaxPooling.cpp
 *
 *  Created on: Jan 3, 2013
 *      Author: gkenyon
 */

#include "MaxPooling.hpp"
#include "../connections/HyPerConn.hpp"

namespace PV {

MaxPooling::MaxPooling()
{
   initialize_base();
}

MaxPooling::MaxPooling(const char * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

MaxPooling::~MaxPooling()
{
}

int MaxPooling::initialize_base(){
   numChannels = 1;
   return PV_SUCCESS;
}

int MaxPooling::initialize(const char * name, HyPerCol * hc)
{
   return HyPerLayer::initialize(name, hc);
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
      pvdata_t * gSynPatchHead = this->getChannel(conn->getChannel());
      pvdata_t * gSynPatchStart = gSynPatchHead + conn->getGSynPatchStart(kPre, arborID);
      pvwdata_t * data = conn->get_wData(arborID,kPre);
      for (int y = 0; y < ny; y++) {
         pvpatch_max_pooling(nk, gSynPatchStart + y*sy, a, data + y*syw, NULL);
//       if (err != 0) printf("  ERROR kPre = %d\n", kPre);
      }
   }

   recvsyn_timer->stop();

   return PV_SUCCESS;
}

} // namespace PV
