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

    float * V,
    const float Vth,
    const float VMax,
    const float VMin,
    const float VShift,
    const float tau_max,
    const float tau_min,
    const float slope_error_std,
    float * dt_tau,
    float * GSynHead,
    float * activity,
    double * error_mean,
    double * error_std);

void HyPerLCALayer2_update_state(
    const int numNeurons,
    const int nx,
    const int ny,
    const int nf,
    const int nb,

    float * V,
    const float Vth,
    const float VMax,
    const float VMin,
    const float VShift,
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
   return PV_SUCCESS;
}

int HyPerLCALayer::initialize(const char * name, HyPerCol * hc, int num_channels)
{
   ANNLayer::initialize(name, hc, num_channels);
   PVParams * params = parent->parameters();
   tauMax = params->value(name, "timeConstantTau", tauMax, true);
   tauMin = params->value(name, "timeConstantTauMinimum", tauMax, false); // default to no adaptation
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
      if(num_channels == 1){
    	  HyPerLCALayer_update_state(num_neurons, nx, ny, nf, loc->nb, V, VThresh,
    			  VMax, VMin, VShift, tauMax, tauMin, slopeErrorStd, &dtTau, gSynHead, A, &error_mean, &errorStd);
      }
      else if(num_channels == 2){
    	  HyPerLCALayer2_update_state(num_neurons, nx, ny, nf, loc->nb, V, VThresh,
    			  VMax, VMin, VShift, tauMax, tauMin, slopeErrorStd, &dtTau, gSynHead, A, &error_mean, &errorStd);
      }
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
#  include "../kernels/HyPerLCALayer2_update_state.cl"
#else
#  undef PV_USE_OPENCL
#  include "../kernels/HyPerLCALayer_update_state.cl"
#  include "../kernels/HyPerLCALayer2_update_state.cl"
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif


