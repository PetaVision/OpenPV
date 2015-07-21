/*
 * ConstGTLayer.cpp
 * Author: slundquist
 */

#include "ConstGTLayer.hpp"
#include <assert.h>
#include <iostream>
#include <fstream>
#include <string>

namespace PV {
ConstGTLayer::ConstGTLayer(const char * name, HyPerCol * hc)
{
   initialize(name, hc);
}

ConstGTLayer::~ConstGTLayer(){
}

int ConstGTLayer::initialize(const char * name, HyPerCol * hc) {
   int status = ANNLayer::initialize(name, hc);
   return status;
}

int ConstGTLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_GTVal(ioFlag);
   return status;
}

void ConstGTLayer::ioParam_GTVal(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "gtVal", &gtVal, gtVal);
}

int ConstGTLayer::updateState(double timef, double dt) {
   pvdata_t * A = getCLayer()->activity->data;
   const PVLayerLoc * loc = getLayerLoc(); 
   //Set binary vals for gtVal
   for(int ni = 0; ni < getNumNeurons(); ni++){
      int nExt = kIndexExtended(ni, loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
      int fi = featureIndex(nExt, loc->nx+loc->halo.rt+loc->halo.lt, loc->ny+loc->halo.up, loc->halo.dn);
      if(fi == gtVal){
         A[nExt] = 1;
      }
      else{
         A[nExt] = -1;
      }
   }
   return PV_SUCCESS;
}
}
