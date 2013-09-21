/*
 * MovieTestProbe.cpp
 * Author: slundquist
 */

#include "MovieTestProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <layers/Movie.hpp>
#include <assert.h>
#include <string.h>

namespace PV {
MovieTestProbe::MovieTestProbe(const char * filename, HyPerLayer * layer, const char * msg)
   : StatsProbe()
{
   initStatsProbe(filename, layer, BufActivity, msg);
}

MovieTestProbe::MovieTestProbe(HyPerLayer * layer, const char * msg)
   : StatsProbe()
{
   initStatsProbe(NULL, layer, BufActivity, msg);
}

int MovieTestProbe::outputState(double timed){
   int status = StatsProbe::outputState(timed);
   const pvdata_t * actLayer = getTargetLayer()->getLayerData();
   const PVLayerLoc * loc = getTargetLayer()->getLayerLoc(); 
   int nx = loc->nx;
   int ny = loc->ny;
   for (int i = 0; i < nx*ny; i++){
      assert(actLayer[i] == dynamic_cast<Movie*>(getTargetLayer())->getFrameNumber());
   }
   return status;
}
}
