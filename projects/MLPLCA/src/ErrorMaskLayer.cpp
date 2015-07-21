/*
 * ErrorMaskLayer.cpp
 * Author: slundquist
 */

#include "ErrorMaskLayer.hpp"
#include <assert.h>
#include <iostream>
#include <fstream>
#include <string>

namespace PV {
ErrorMaskLayer::ErrorMaskLayer(const char * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

int ErrorMaskLayer::initialize_base()
{
   errThresh = .1;
   return PV_SUCCESS;
}

int ErrorMaskLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_ErrThresh(ioFlag);
   return status;
}

void ErrorMaskLayer::ioParam_ErrThresh(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "errThresh", &errThresh, errThresh);
}

int ErrorMaskLayer::updateState(double timef, double dt) {
   pvdata_t * GSynExt = getChannel(CHANNEL_EXC);
   pvdata_t * A = getCLayer()->activity->data;
   pvdata_t * V = getV();
   const PVLayerLoc * loc = getLayerLoc(); 
   for(int ni = 0; ni < getNumNeurons(); ni++){
      int next = kIndexExtended(ni, loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
      //std::cout << "ni: " << ni << "  error: " << GSynExt[ni] << "\n";
      //Only check for positive features
      if(GSynExt[ni] >= errThresh){
         A[next] = 1;
      }
      else{
         A[next] = 0;
      }
   }
   return PV_SUCCESS;
}
}
