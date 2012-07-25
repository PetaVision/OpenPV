/*
 * BIDSCloneLayer.hpp
 * can be used to map BIDSLayers to larger dimensions
 *
 *  Created on: Jul 24, 2012
 *      Author: bnowers
 */

#ifndef BIDSCLONELAYER_HPP_
#define BIDSCLONELAYER_HPP_

#include "HyPerLayer.hpp"
#include "LIF.hpp"
#include "BIDSLayer.hpp"

#include "../kernels/LIF_params.h"

namespace PV {

// CloneLayer can be used to implement Sigmoid junctions between spiking neurons
class BIDSCloneLayer: public HyPerLayer {
public:
   BIDSCloneLayer(const char * name, HyPerCol * hc, LIF * clone);
   virtual ~BIDSCloneLayer();
   virtual int updateState(float timef, float dt);
   // virtual int updateV();
   // virtual int setActivity();
   // virtual int resetGSynBuffers();
   LIF * sourceLayer;
   virtual int setActivity();
   int outputState(float timef, bool last);
   int mapCoords();
protected:
   BIDSCloneLayer();
   int initialize(const char * name, HyPerCol * hc, LIF * clone);
   int initialize_base();
   float V0;
   float Vth;
   bool  InverseFlag;
   bool  SigmoidFlag;
   float SigmoidAlpha;
   int numNodes;
   unsigned int *sourceLayerA;
   BIDSCoords * coords;
   unsigned int *sourceLayerNumIndices;

private:

};

}

#endif /* BIDSCLONELAYER_HPP_ */
