/*
 * SigmoidLayer.hpp
 * can be used to implement Sigmoid junctions
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#ifndef SIGMOIDLAYER_HPP_
#define SIGMOIDLAYER_HPP_

#include "HyPerLayer.hpp"
#include "LIF.hpp"

#include "../kernels/LIF_params.h"

namespace PV {

// CloneLayer can be used to implement Sigmoid junctions between spiking neurons
class SigmoidLayer: public HyPerLayer {
public:
   SigmoidLayer(const char * name, HyPerCol * hc, LIF * clone);
   virtual ~SigmoidLayer();
   virtual int updateState(float timef, float dt);
   // virtual int updateV();
   // virtual int setActivity();
   // virtual int resetGSynBuffers();
   LIF * sourceLayer;
   virtual int setActivity();
protected:
   SigmoidLayer();
   int initialize(const char * name, HyPerCol * hc, LIF * clone);
   /* static */ int updateState(float timef, float dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead, float Vth, float V0, float sigmoid_alpha, bool sigmoid_flag, bool inverse_flag, unsigned int * active_indices, unsigned int * num_active);
private:
   int initialize_base();
   float V0;
   float Vth;
   bool  InverseFlag;
   bool  SigmoidFlag;
   float SigmoidAlpha;
};

}

#endif /* CLONELAYER_HPP_ */
