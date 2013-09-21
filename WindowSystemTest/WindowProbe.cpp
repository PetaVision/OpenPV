/*
 * WindowProbe.cpp
 * Author: slundquist
 */

#include "WindowProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <layers/HyPerLCALayer.hpp>
#include <assert.h>
#include <string.h>

namespace PV {
WindowProbe::WindowProbe(const char * filename, HyPerLayer * layer, const char * msg)
   : StatsProbe()
{
   initStatsProbe(filename, layer, BufActivity, msg);
}

WindowProbe::WindowProbe(HyPerLayer * layer, const char * msg)
   : StatsProbe()
{
   initStatsProbe(NULL, layer, BufActivity, msg);
}

int WindowProbe::outputState(double timed){
   //For now, the window information must come from HyPerLCALayer
   HyPerLCALayer * windowLayer = dynamic_cast<HyPerLCALayer *>(getTargetLayer());
   if (windowLayer ==NULL) {
      fprintf(stderr, "WindowProbe error: Target layer must be a HyPerLCALayer.\n");
      abort();
   }
   int status = StatsProbe::outputState(timed);
   int numRes = windowLayer->getNumNeurons();
   const pvdata_t * actLayer = windowLayer->getLayerData();
   const PVLayerLoc * loc = windowLayer->getLayerLoc();
   for (int kLocalRes = 0; kLocalRes < numRes; kLocalRes++){
      int kLocalExt = kIndexExtended(kLocalRes, loc->nx, loc->ny, loc->nf, loc->nb);
      int kxGlobalExt = kxPos(kLocalExt, loc->nx + 2*loc->nb, loc->ny + 2*loc->nb, loc->nf) + loc->kx0;
      int kyGlobalExt = kyPos(kLocalExt, loc->nx + 2*loc->nb, loc->ny + 2*loc->nb, loc->nf) + loc->ky0;
      //Get window from windowLayer
      int windowId = windowLayer->calcWindow(kxGlobalExt, kyGlobalExt);
      float diff = abs(actLayer[kLocalExt] - windowId);
      if(diff != 0 && timed != 0){
         std::cout << "Not Equal! layer: " << actLayer[kLocalExt] << " window: " << windowId << " diff: " << diff << "\n";
         std::cout << "x: " << kxGlobalExt << " y: " << kyGlobalExt << "\n";
      }
      //assert(actLayer[kLocalExt] == windowId);

   }

   return status;
}
}
