/*
 * DisparityLCALayer.cpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#include "DisparityLCALayer.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void HyPerLCALayer_update_state(
    const int numNeurons,
    const int nx,
    const int ny,
    const int nf,
    const int lt,
    const int rt,
    const int dn,
    const int up,
    const int numChannels,

    float * V,
    const float Vth,
    const float AMax,
    const float AMin,
    const float AShift,
    const float VWidth,
    const bool selfInteract,
    const float dt,
    float * GSynHead,
    float * activity);

#ifdef __cplusplus
}
#endif

namespace PV {

DisparityLCALayer::DisparityLCALayer()
{
   initialize_base();
}

DisparityLCALayer::DisparityLCALayer(const char * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

DisparityLCALayer::~DisparityLCALayer()
{
}

int DisparityLCALayer::initialize_base()
{
   disparityLayer = NULL;
   return PV_SUCCESS;
}

int DisparityLCALayer::communicateInitInfo() {
   int status = HyPerLCALayer::communicateInitInfo();
   HyPerLayer* h_layer = parent->getLayerFromName(disparityLayerName);
   if (h_layer==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: disparityLayerName \"%s\" is not a layer in the HyPerCol.\n",
                 parent->parameters()->groupKeywordFromName(name), name, disparityLayerName);
      }
   }
   disparityLayer = dynamic_cast<Movie *>(h_layer);
   if (disparityLayer==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: disparityLayerName \"%s\" is not a DisparityMovieLayer.\n",
                 parent->parameters()->groupKeywordFromName(name), name, disparityLayerName);
      }
   }
   return PV_SUCCESS;
}



int DisparityLCALayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLCALayer::ioParamsFillGroup(ioFlag);
   ioParam_disparityLayerName(ioFlag);
   return status;
}

void DisparityLCALayer::ioParam_disparityLayerName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "disparityLayerName", &disparityLayerName);
}


//Layer does exactly the same as HyPerLCA, but reads from disparityLayer which neuron should be active
int DisparityLCALayer::doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
      pvdata_t * V, int num_channels, pvdata_t * gSynHead)
{
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   int nxGlobal = loc->nxGlobal;
   int nyGlobal = loc->nyGlobal;
   //Only allow one mpi process for easy coding
   assert(nx == nxGlobal && ny == nyGlobal);

   //Initialize with the neuron we're looking at on, with everything else off
   std::string filename = std::string(disparityLayer->getFilename());
   //Parse filename to grab layer name and neuron index
   size_t und_pos = filename.find_last_of("_");
   size_t ext_pos = filename.find_last_of(".");
   int neuronIdx = atoi(filename.substr(und_pos+1, ext_pos-und_pos-1).c_str());

   size_t name_pos = filename.find_last_of("/");
   size_t len = strlen(name);
   std::string layerName = filename.substr(name_pos+1, len);

   //Calculate target index
   int target_idx = kIndex((nx/2)-1, (ny/2)-1, neuronIdx, nx, ny, nf);

   //Only update when the probe updates
   if (triggerLayer != NULL && triggerLayer->needUpdate(time, parent->getDeltaTime())){
      if(strcmp(name, layerName.c_str()) == 0){
         for (int i = 0; i<num_neurons; i++){
            if(i == target_idx){
               V[i]=1.0;
            }
            else{
               V[i]=0.0;
            }
         }
      }
      else{
         for (int i = 0; i<num_neurons; i++){
            V[i]=0.0;
         }
      }
   }

   HyPerLCALayer_update_state(num_neurons, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up, numChannels,
         V, VThresh, AMax, AMin, AShift, VWidth, 
         selfInteract, dt/timeConstantTau, gSynHead, A);

   //update_timer->stop();
   return PV_SUCCESS;
}

} /* namespace PV */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PV_USE_OPENCL
#  include <kernels/HyPerLCALayer_update_state.cl>
#else
#  undef PV_USE_OPENCL
#  include <kernels/HyPerLCALayer_update_state.cl>
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif



