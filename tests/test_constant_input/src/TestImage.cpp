/*
 * TestImage.cpp
 *
 *  Created on: Mar 19, 2010
 *      Author: Craig Rasmussen
 */

#include "TestImage.hpp"

namespace PV {

TestImage::TestImage() { initialize_base(); }

TestImage::TestImage(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

TestImage::~TestImage() {}

int TestImage::initialize_base() { return PV_SUCCESS; }

int TestImage::initialize(const char *name, HyPerCol *hc) {
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
   parent->parameters()->ioParamValue(ioFlag, name, "constantVal", &val, (float)1);
}

void TestImage::allocateV() { FatalIf(!(getV() == nullptr), "Test failed.\n"); }

void TestImage::initializeActivity() {
   for (int k = 0; k < getNumNeurons(); k++) {
      const PVLayerLoc *loc = getLayerLoc();
      int kExt              = kIndexExtended(
            k, loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
      getActivity()[kExt] = val;
   }
}

Response::Status TestImage::updateState(double timed, double dt) { return Response::SUCCESS; }

} // namespace PV
