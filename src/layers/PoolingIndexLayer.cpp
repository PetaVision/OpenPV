/*
 * PoolingIndexLayer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#include "PoolingIndexLayer.hpp"
#include "../layers/updateStateFunctions.h"

namespace PV {

PoolingIndexLayer::PoolingIndexLayer() {
   initialize_base();
}

PoolingIndexLayer::PoolingIndexLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

PoolingIndexLayer::~PoolingIndexLayer() {}

int PoolingIndexLayer::initialize_base() {
   this->numChannels = 1;
   return PV_SUCCESS;
}

int PoolingIndexLayer::initialize(const char * name, HyPerCol * hc) {
   int status = HyPerLayer::initialize(name, hc);
   //This layer is storing it's buffers as ints. This is a check to make sure the sizes are the same
   assert(sizeof(int) == sizeof(pvdata_t));
   assert(status == PV_SUCCESS);
   return status;
}

int PoolingIndexLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLayer::ioParamsFillGroup(ioFlag);
   return status;
}

void PoolingIndexLayer::ioParam_dataType(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "dataType");
      dataType = PV_INT;
   }
}

//This function should never be called, since this layer should never be a post layer and only accessed from PoolingConn.
int PoolingIndexLayer::requireChannel(int channelNeeded, int * numChannelsResult) {
   std::cout << "Error, PoolingIndexLayer cannot be a post layer\n";
   exit(-1);
   return PV_SUCCESS;
}

//This is a function that's overwriting HyPerCol
int PoolingIndexLayer::resetGSynBuffers(double timef, double dt) {
   //Reset GSynBuffers does nothing, as the orig pooling connection deals with clearing this buffer
   return PV_SUCCESS;
}

BaseObject * createPoolingIndexLayer(char const * name, HyPerCol * hc) {
   return hc ? new PoolingIndexLayer(name, hc) : NULL;
}

}  // end namespace PV
