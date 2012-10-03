/*
 * LCALIFLayer.hpp
 *
 *  Created on: Oct 3, 2012
 *      Author: slundquist
 */

#ifndef LCALIFLAYER_HPP_
#define LCALIFLAYER_HPP_

//#include "HyPerLayer.hpp"
#include "LIF.hpp"

namespace PV {
class LCALIFLayer : public PV::LIF {
public:
   LCALIFLayer(const char* name, HyPerCol * hc); // The constructor called by other methods
   virtual ~LCALIFLayer();
   int updateState(float timef, float dt);
   int updateState(float timef, float dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead, bool spiking, unsigned int * active_indices, unsigned int * num_active);
   int findFlag(int numMatrixCol, int numMatrixRow);

   inline float getTargetRate() {return targetRate;};
   float getTrace();
protected:
   int allocateBuffers();
   pvdata_t * integratedSpikeCount;      // plasticity decrement variable for postsynaptic layer
   float tau_LCA;
   float tau_thr;
   float targetRate;
   LCALIFLayer();
   int initialize(const char * name, HyPerCol * hc, int num_channels, const char * kernel_name);
   int initialize_base();
  // other methods and member variables
private:
  // other methods and member variables
};
}




#endif /* LCALIFLAYER_HPP_ */
