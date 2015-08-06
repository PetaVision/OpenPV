/*
 * DisparityLCALayer.cpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#include "DisparityLCALayer.hpp"
#  include <kernels/HyPerLCALayer_update_state.cl>


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
         fprintf(stderr, "%s \"%s\" error: disparityLayerName \"%s\" is not a MovieLayer.\n",
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
   int kx0 = loc->kx0;
   int ky0 = loc->ky0;
   int nxGlobal = loc->nxGlobal;
   int nyGlobal = loc->nyGlobal;
   int num_local_neurons = nx*ny*nf;

   //Initialize with the neuron we're looking at on, with everything else off
   std::string filename = std::string(disparityLayer->getFilename(0));
   //Parse filename to grab layer name and neuron index
   size_t und_pos = filename.find_last_of("_");
   size_t ext_pos = filename.find_last_of(".");
   int neuronIdx = atoi(filename.substr(und_pos+1, ext_pos-und_pos-1).c_str());

   size_t name_pos = filename.find_last_of("/");
   size_t len = strlen(name);
   std::string layerName = filename.substr(name_pos+1, len);

   //Calculate target index
   int target_global_x = (nxGlobal/2)-1;
   int target_global_y = (nyGlobal/2)-1;
   int target_f = neuronIdx;

   //Only update when the probe updates
   if (triggerLayer != NULL && triggerLayer->needUpdate(time, parent->getDeltaTime())){
      if(strcmp(name, layerName.c_str()) == 0){
         for (int yi = 0; yi < ny ; yi++){
            for(int xi = 0; xi < nx; xi++){
               for(int fi = 0; fi < nf; fi++){
                  int target_local = kIndex(xi, yi, fi, nx, ny, nf);
                  if((yi+ky0) == target_global_y &&
                     (xi+kx0) == target_global_x &&
                     (fi)     == target_f)
                  {
                     V[target_local]=1.0;
                  }
                  else{
                     V[target_local]=0.0;
                  }
               }
            }
         }
      }
      else{
         for (int i = 0; i<num_local_neurons; i++){
            V[i]=0.0;
         }
      }
   }

   HyPerLCALayer_update_state(num_local_neurons, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up, numChannels,
         V, VThresh, AMax, AMin, AShift, VWidth, 
         selfInteract, dt/timeConstantTau, gSynHead, A);

   //update_timer->stop();
   return PV_SUCCESS;
}

} /* namespace PV */

