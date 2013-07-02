/*
 * ImageTestProbe.cpp
 * Author: slundquist
 */

#include "ImageTestProbe.hpp"
#include "../PetaVision/src/include/pv_arch.h"
#include "../PetaVision/src/layers/HyPerLayer.hpp"
#include "../PetaVision/src/layers/Image.hpp"
#include <assert.h>
#include <string.h>

namespace PV {
ImageTestProbe::ImageTestProbe(const char * filename, HyPerLayer * layer, const char * msg)
   : StatsProbe()
{
   initStatsProbe(filename, layer, BufActivity, msg);
}

ImageTestProbe::ImageTestProbe(HyPerLayer * layer, const char * msg)
   : StatsProbe()
{
   initStatsProbe(NULL, layer, BufActivity, msg);
}

int ImageTestProbe::outputState(double timed){
   int status = StatsProbe::outputState(timed);
   const pvdata_t * actLayer = getTargetLayer()->getLayerData();
   const PVLayerLoc * loc = getTargetLayer()->getLayerLoc(); 
   int nx = loc->nx;
   int ny = loc->ny;
   for (int i = 0; i < nx*ny; i++){
      assert(actLayer[i] == dynamic_cast<Image*>(getTargetLayer())->getFrameNumber());
   }
   return status;
}
}
