/*
 * ANNErrorLayer.cpp
 *
 *  Created on: Jun 21, 2013
 *      Author: gkenyon
 */

#include "ANNErrorLayer.hpp"

void ANNErrorLayer_update_state(
      const int nbatch,
      const int numNeurons,
      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,

      float *V,
      int numVertices,
      float *verticesV,
      float *verticesA,
      float *slopes,
      float *GSynHead,
      float *activity,
      const float errScale);

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
   return status;
}

int ANNErrorLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_errScale(ioFlag);
   return status;
}

void ANNErrorLayer::ioParam_errScale(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "errScale", &errScale, errScale, true /*warnIfAbsent*/);
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
      if (parent->columnId() == 0) {
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

Response::Status ANNErrorLayer::updateState(double time, double dt) {
   const PVLayerLoc *loc = getLayerLoc();
   float *A              = clayer->activity->data;
   float *V              = getV();
   int num_channels      = getNumChannels();
   float *gSynHead       = GSyn == NULL ? NULL : GSyn[0];
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;
   int num_neurons       = nx * ny * nf;
   int nbatch            = loc->nbatch;
   ANNErrorLayer_update_state(
         nbatch,
         num_neurons,
         nx,
         ny,
         nf,
         loc->halo.lt,
         loc->halo.rt,
         loc->halo.dn,
         loc->halo.up,
         V,
         numVertices,
         verticesV,
         verticesA,
         slopes,
         gSynHead,
         A,
         errScale);
   return Response::SUCCESS;
}

} /* namespace PV */

void ANNErrorLayer_update_state(
      const int nbatch,
      const int numNeurons,
      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,

      float *V,
      int numVertices,
      float *verticesV,
      float *verticesA,
      float *slopes,
      float *GSynHead,
      float *activity,
      const float errScale) {
   updateV_ANNErrorLayer(
         nbatch,
         numNeurons,
         V,
         GSynHead,
         activity,
         numVertices,
         verticesV,
         verticesA,
         slopes,
         nx,
         ny,
         nf,
         lt,
         rt,
         dn,
         up,
         errScale);
}
