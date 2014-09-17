/*
 * ImageTestProbe.cpp
 * Author: slundquist
 */

#include "ImageTestProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <layers/Image.hpp>
#include <assert.h>
#include <string.h>

namespace PV {
ImageTestProbe::ImageTestProbe(const char * probeName, HyPerCol * hc)
   : StatsProbe()
{
   initImageTestProbe(probeName, hc);
}

int ImageTestProbe::initImageTestProbe_base() { return PV_SUCCESS; }

int ImageTestProbe::initImageTestProbe(const char * probeName, HyPerCol * hc) {
   return initStatsProbe(probeName, hc);
}

void ImageTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   requireType(BufActivity);
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
