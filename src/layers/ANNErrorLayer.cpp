/*
 * ANNErrorLayer.cpp
 *
 *  Created on: Jun 21, 2013
 *      Author: gkenyon
 */

#include "ANNErrorLayer.hpp"

#ifdef __cplusplus
extern "C" {
#endif

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

    float * V,
    int numVertices,
    float * verticesV,
    float * verticesA,
    float * slopes,
    float * GSynHead,
    float * activity,
    const float errScale);


#ifdef __cplusplus
}
#endif

namespace PV {

ANNErrorLayer::ANNErrorLayer()
{
   initialize_base();
}

ANNErrorLayer::ANNErrorLayer(const char * name, HyPerCol * hc)
{
   int status = initialize_base();
   if (status == PV_SUCCESS) { status = initialize(name, hc); }
   if (status != PV_SUCCESS) {
      fprintf(stderr, "Creating ANNErrorLayer \"%s\" failed.\n", name);
      exit(EXIT_FAILURE);
   }
}

ANNErrorLayer::~ANNErrorLayer()
{
}

int ANNErrorLayer::initialize_base()
{
   errScale = 1;
   return PV_SUCCESS;
}

int ANNErrorLayer::initialize(const char * name, HyPerCol * hc)
{
   int status = ANNLayer::initialize(name, hc);
   return status;
}

int ANNErrorLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_errScale(ioFlag);
   return status;
}

void ANNErrorLayer::ioParam_errScale(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "errScale", &errScale, errScale, true/*warnIfAbsent*/);
}

int ANNErrorLayer::setVertices() {
   slopeNegInf = 1.0;
   slopePosInf = 1.0;
   if (VThresh > 0) {
      numVertices = 4;
      verticesV = (pvpotentialdata_t *) malloc((size_t) numVertices * sizeof(*verticesV));
      verticesA = (pvadata_t *) malloc((size_t) numVertices * sizeof(*verticesA));
      if (verticesV==NULL || verticesA==NULL) {
         fprintf(stderr, "%s \"%s\": unable to allocate memory for vertices: %s\n",
               getKeyword(), name, strerror(errno));
         exit(EXIT_FAILURE);
      }
      verticesV[0] = -VThresh; verticesA[0] = -VThresh;
      verticesV[1] = -VThresh; verticesA[1] = 0.0;
      verticesV[2] = VThresh; verticesA[2] = 0.0;
      verticesV[3] = VThresh; verticesA[3] = VThresh;
   }
   else {
      // checkVertices will complain if VThresh is negative but not "negative infinity"
      numVertices = 1;
      verticesV = (pvpotentialdata_t *) malloc((size_t) numVertices * sizeof(*verticesV));
      verticesA = (pvadata_t *) malloc((size_t) numVertices * sizeof(*verticesA));
      if (verticesV==NULL || verticesA==NULL) {
         fprintf(stderr, "%s \"%s\": unable to allocate memory for vertices: %s\n",
               getKeyword(), name, strerror(errno));
         exit(EXIT_FAILURE);
      }
      verticesV[0] = 0.0f; verticesA[0] = 0.0f;
   }
   return PV_SUCCESS;
}

int ANNErrorLayer::checkVertices() {
   int status = PV_SUCCESS;
   if (VThresh < 0 && VThresh > -0.999*max_pvvdata_t) { // 0.999 is to allow for imprecision from params files using 3.40282e+38 instead of infinity
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: VThresh cannot be negative (value is %f).\n",
                  this->getKeyword(), this->getName(), VThresh);
      }
      status = PV_FAILURE;
   }
   else {
      assert(PtwiseLinearTransferLayer::checkVertices()==PV_SUCCESS);
   }
   return status;
}

int ANNErrorLayer::doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
      pvdata_t * V, int num_channels, pvdata_t * gSynHead)
{
//#ifdef PV_USE_OPENCL
//   if(gpuAccelerateFlag) {
//      updateStateOpenCL(time, dt);
//      //HyPerLayer::updateState(time, dt);
//   }
//   else {
//#endif
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   int nbatch = loc->nbatch;
   ANNErrorLayer_update_state(nbatch, num_neurons, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up,
         V, numVertices, verticesV, verticesA, slopes, gSynHead, A, errScale);
//#ifdef PV_USE_OPENCL
//   }
//#endif
   return PV_SUCCESS;
}

BaseObject * createANNErrorLayer(char const * name, HyPerCol * hc) {
   return hc ? new ANNErrorLayer(name, hc) : NULL;
}

} /* namespace PV */


#ifdef __cplusplus
extern "C" {
#endif

#ifndef PV_USE_OPENCL
#  include "../kernels/ANNErrorLayer_update_state.cl"
#else
#  undef PV_USE_OPENCL
#  include "../kernels/ANNErrorLayer_update_state.cl"
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif
