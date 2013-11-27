/*
 * HyPerLCALayer.cpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#include "HyPerLCALayer.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void HyPerLCALayer_update_state(
    const int numNeurons,
    const int nx,
    const int ny,
    const int nf,
    const int nb,
    const int numChannels,

    float * V,
    const float Vth,
    const float VMax,
    const float VMin,
    const float VShift,
    const float VWidth,
    const float tau_max,
    const float tau_min,
    const float slope_error_std,
    float * dt_tau,
    float * GSynHead,
    float * activity,
    double * error_mean,
    double * error_std);

#ifdef __cplusplus
}
#endif

namespace PV {

HyPerLCALayer::HyPerLCALayer()
{
   initialize_base();
}

HyPerLCALayer::HyPerLCALayer(const char * name, HyPerCol * hc, int num_channels)
{
   initialize_base();
   assert(num_channels <= 2);
   initialize(name, hc, num_channels);
}

HyPerLCALayer::HyPerLCALayer(const char * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc, 1);
}

HyPerLCALayer::~HyPerLCALayer()
{
}

int HyPerLCALayer::initialize_base()
{
   tauMax = 1.0;
   tauMin = tauMax;
   errorStd = 1.0;
   slopeErrorStd = 1.0;
   //Locality in KernelConn
   numWindowX = 1;
   numWindowY = 1;
   windowSymX = false;
   windowSymY = false;
   return PV_SUCCESS;
}

bool HyPerLCALayer::inWindowExt(int windowId, int neuronIdxExt){
   const PVLayerLoc * loc = this->getLayerLoc();
   int globalExtX = kxPos(neuronIdxExt, loc->nx + 2*loc->nb, loc->ny + 2*loc->nb, loc->nf) + loc->kx0;
   int globalExtY = kyPos(neuronIdxExt, loc->nx + 2*loc->nb, loc->ny + 2*loc->nb, loc->nf) + loc->ky0;
   int outWindow = calcWindow(globalExtX, globalExtY);
   //std::cout << globalExtX << "/" << loc->nxGlobal + 2*loc->nb << " " << globalExtY << "/" << loc->nyGlobal + 2*loc->nb << " " << outWindow << "," << windowId << "\n";
   return (outWindow == windowId);
}

bool HyPerLCALayer::inWindowRes(int windowId, int neuronIdxRes){
   const PVLayerLoc * loc = this->getLayerLoc();
   int globalExtX = kxPos(neuronIdxRes, loc->nx, loc->ny, loc->nf) + loc->kx0;
   int globalExtY = kyPos(neuronIdxRes, loc->nx, loc->ny, loc->nf) + loc->ky0;
   int outWindow = calcWindow(globalExtX, globalExtY);
   return (outWindow == windowId);
}

int HyPerLCALayer::calcWindow(int globalExtX, int globalExtY){
   const PVLayerLoc * loc = this->getLayerLoc();
   //Calculate x and y with symmetry on
   if(windowSymX && globalExtX >= floor((loc->nxGlobal + 2*loc->nb)/2)){
      globalExtX = loc->nxGlobal+2*loc->nb - globalExtX - 1;
   }
   if(windowSymY && globalExtY >= floor((loc->nyGlobal + 2*loc->nb)/2)){
      globalExtY = loc->nyGlobal+2*loc->nb - globalExtY - 1;
   }
   //Calculate the window x and y
   int windowX = floor(((float)globalExtX/(loc->nxGlobal+2*loc->nb)) * numWindowX); 
   int windowY = floor(((float)globalExtY/(loc->nyGlobal+2*loc->nb)) * numWindowY);
   //Change x and y into index
   int windowIdx = (windowY * numWindowY) + windowX;
   assert(windowIdx < getNumWindows());
   assert(windowIdx >= 0);
   return windowIdx;
}

int HyPerLCALayer::getNumWindows(){
   int windowsX, windowsY;
   if(windowSymX){
      windowsX = ceil((float)numWindowX / 2);
   }
   else{
      windowsX = numWindowX;
   }
   if(windowSymY){
      windowsY = ceil((float)numWindowY / 2);
   }
   else{
      windowsY = numWindowY;
   }
   return windowsX * windowsY;
}

int HyPerLCALayer::initialize(const char * name, HyPerCol * hc, int num_channels)
{
   ANNLayer::initialize(name, hc, num_channels);
   PVParams * params = parent->parameters();
   tauMax = params->value(name, "timeConstantTau", tauMax, true);
   tauMin = params->value(name, "timeConstantTauMinimum", tauMax, false); // default to no adaptation
   numWindowX = params->value(name, "numWindowX", numWindowX);
   numWindowY = params->value(name, "numWindowY", numWindowX);
   if(numWindowX != 1){
      windowSymX = (bool)params->value(name, "windowSymX", windowSymX);
   }
   if(numWindowY != 1){
      windowSymY = (bool)params->value(name, "windowSymY", windowSymY);
   }
   int windowXNeeded;

   if ((tauMax - tauMin) > 1.0){
	   slopeErrorStd = params->value(name, "slopeErrorStd", slopeErrorStd, true);
   }
   return PV_SUCCESS;
}

int HyPerLCALayer::doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
      pvdata_t * V, int num_channels, pvdata_t * gSynHead, bool spiking,
      unsigned int * active_indices, unsigned int * num_active)
{
   update_timer->start();
#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag) {
      updateStateOpenCL(time, dt);
      //HyPerLayer::updateState(time, dt);
   }
   else {
#endif
      int nx = loc->nx;
      int ny = loc->ny;
      int nf = loc->nf;
      int num_neurons = nx*ny*nf;
      dtTau = dt;
      double error_mean = 0;
    	  HyPerLCALayer_update_state(num_neurons, nx, ny, nf, loc->nb, numChannels,
    			  V, VThresh, VMax, VMin, VShift, VWidth, tauMax, tauMin, slopeErrorStd,
    			  &dtTau, gSynHead, A, &error_mean, &errorStd);
      if (this->writeSparseActivity){
         updateActiveIndices();  // added by GTK to allow for sparse output, can this be made an inline function???
      }
#ifdef PV_USE_OPENCL
   }
#endif

   update_timer->stop();
   return PV_SUCCESS;
}

} /* namespace PV */


#ifdef __cplusplus
extern "C" {
#endif

#ifndef PV_USE_OPENCL
#  include "../kernels/HyPerLCALayer_update_state.cl"
#else
#  undef PV_USE_OPENCL
#  include "../kernels/HyPerLCALayer_update_state.cl"
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif


