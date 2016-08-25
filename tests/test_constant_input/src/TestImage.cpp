/*
 * TestImage.cpp
 *
 *  Created on: Mar 19, 2010
 *      Author: Craig Rasmussen
 */

#include "TestImage.hpp"

namespace PV {

TestImage::TestImage() {
   initialize_base();
}

TestImage::TestImage(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

TestImage::~TestImage() {
}

int TestImage::initialize_base() {
   return PV_SUCCESS;
}

int TestImage::initialize(const char * name, HyPerCol * hc) {
   return HyPerLayer::initialize(name, hc);
}

int TestImage::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLayer::ioParamsFillGroup(ioFlag);
   ioParam_constantVal(ioFlag);
   return status;
}

void TestImage::ioParam_InitVType(enum ParamsIOFlag ioFlag) {
   parent->parameters()->handleUnnecessaryParameter(name, "InitVType");
   return;
}

void TestImage::ioParam_constantVal(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "constantVal", &val, (pvdata_t) 1);
}

int TestImage::allocateV() {
   pvErrorIf(!(getV()==NULL), "Test failed.\n");
   return PV_SUCCESS;
}

int TestImage::initializeActivity() {
   for (int k=0; k<getNumNeurons(); k++) {
      const PVLayerLoc * loc = getLayerLoc();
      int kExt = kIndexExtended(k,loc->nx,loc->ny,loc->nf,loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
      getActivity()[kExt] = val;
   }
   return PV_SUCCESS;
}

int TestImage::updateState(double timed, double dt) {
   return PV_SUCCESS;
}

} // namespace PV
