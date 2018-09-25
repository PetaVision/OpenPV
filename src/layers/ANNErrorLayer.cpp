/*
 * ANNErrorLayer.cpp
 *
 *  Created on: Jun 21, 2013
 *      Author: gkenyon
 */

#include "ANNErrorLayer.hpp"
#include "components/ErrScaleInternalStateBuffer.hpp"

namespace PV {

ANNErrorLayer::ANNErrorLayer() { initialize_base(); }

ANNErrorLayer::ANNErrorLayer(const char *name, HyPerCol *hc) {
   int status = initialize_base();
   if (status == PV_SUCCESS) {
      status = initialize(name, hc);
   }
   if (status != PV_SUCCESS) {
      Fatal().printf("Creating ANNErrorLayer \"%s\" failed.\n", name);
   }
}

ANNErrorLayer::~ANNErrorLayer() {}

int ANNErrorLayer::initialize_base() {
   errScale = 1;
   return PV_SUCCESS;
}

int ANNErrorLayer::initialize(const char *name, HyPerCol *hc) {
   int status = ANNLayer::initialize(name, hc);
   mLayerInput->requireChannel(CHANNEL_EXC);
   mLayerInput->requireChannel(CHANNEL_INH);
   return status;
}

InternalStateBuffer *ANNErrorLayer::createInternalState() {
   return new ErrScaleInternalStateBuffer(getName(), parent);
}

int ANNErrorLayer::setVertices() {
   pvAssert(!layerListsVerticesInParams());
   slopeNegInf = 1.0;
   slopePosInf = 1.0;
   if (VThresh > 0) {
      numVertices = 4;
      verticesV   = (float *)malloc((size_t)numVertices * sizeof(*verticesV));
      verticesA   = (float *)malloc((size_t)numVertices * sizeof(*verticesA));
      if (verticesV == NULL || verticesA == NULL) {
         Fatal().printf(
               "%s: unable to allocate memory for vertices: %s\n",
               getDescription_c(),
               strerror(errno));
      }
      verticesV[0] = -VThresh;
      verticesA[0] = -VThresh;
      verticesV[1] = -VThresh;
      verticesA[1] = 0.0;
      verticesV[2] = VThresh;
      verticesA[2] = 0.0;
      verticesV[3] = VThresh;
      verticesA[3] = VThresh;
   }
   else {
      // checkVertices will complain if VThresh is negative but not "negative infinity"
      numVertices = 1;
      verticesV   = (float *)malloc((size_t)numVertices * sizeof(*verticesV));
      verticesA   = (float *)malloc((size_t)numVertices * sizeof(*verticesA));
      if (verticesV == NULL || verticesA == NULL) {
         Fatal().printf(
               "%s: unable to allocate memory for vertices: %s\n",
               getDescription_c(),
               strerror(errno));
      }
      verticesV[0] = 0.0f;
      verticesA[0] = 0.0f;
   }
   return PV_SUCCESS;
}

int ANNErrorLayer::checkVertices() const {
   int status = PV_SUCCESS;
   if (VThresh < 0 && VThresh > -(float)0.999 * FLT_MAX) { // 0.999 is to allow for imprecision from
      // params files using 3.40282e+38 instead
      // of infinity
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: VThresh cannot be negative (value is %f).\n",
               getDescription_c(),
               (double)VThresh);
      }
      status = PV_FAILURE;
   }
   else {
      pvAssert(ANNLayer::checkVertices() == PV_SUCCESS);
   }
   return status;
}

} /* namespace PV */
