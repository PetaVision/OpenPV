/*
 * ANNLabelLayer.cpp
 *
 *  Created on: Jul. 23, 2013
 *      Author: xinhuazhang
 */

#include "ANNLabelLayer.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void ANNLabelLayer_update_state(
    const int numNeurons,
    const int nx,
    const int ny,
    const int nf,
    const int lt,
    const int rt,
    const int dn,
    const int up,

    float * V,
    const float Vth,
    const float AMax,
    const float AMin,
    const float AShift,
    float * GSynHead,
    float * activity);

#ifdef __cplusplus
}
#endif


namespace PV{

    ANNLabelLayer::ANNLabelLayer()
    {
        initialize_base();
    }
    
    ANNLabelLayer::ANNLabelLayer(const char * name, HyPerCol * hc)
    {
        initialize_base();
        assert(numChannels == 2);
        initialize(name, hc);
    }

    ANNLabelLayer::~ANNLabelLayer()
    {
    }

    int ANNLabelLayer::initialize_base()
    {
        return PV_SUCCESS;
    }

    int ANNLabelLayer::initialize(const char * name, HyPerCol * hc)
    {
        ANNLayer::initialize(name, hc);
        return PV_SUCCESS;
    }

    int ANNLabelLayer::doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead)
    {
        //update_timer->start();
//#ifdef PV_USE_OPENCL
//        if(gpuAccelerateFlag) {
//            updateStateOpenCL(time, dt);
//            //HyPerLayer::updateState(time, dt);
//        }
//        else {
//#endif
            int nx = loc->nx;
            int ny = loc->ny;
            int nf = loc->nf;
            int num_neurons = nx*ny*nf;
            ANNLabelLayer_update_state(num_neurons, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up, V, VThresh,
                                       AMax, AMin, AShift, gSynHead, A);
//#ifdef PV_USE_OPENCL
//        }
//#endif

        //update_timer->stop();
        return PV_SUCCESS;
    }

}

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PV_USE_OPENCL
#  include "../kernels/ANNLabelLayer_update_state.cl"
#else
#  undef PV_USE_OPENCL
#  include "../kernels/ANNLabelLayer_update_state.cl"
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif
