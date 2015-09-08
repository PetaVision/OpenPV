/*
 * BIDSLayer.hpp
 *
 *  Created on: Jun 26, 2012
 *      Author: Bren Nowers
 *
 */

#ifndef INITBIDSLAYER_HPP_
#define INITBIDSLAYER_HPP_

#include <layers/HyPerLayer.hpp>
#include <layers/LIF.hpp>

namespace PV {
class BIDSLayer : public PV::LIF {
public:
   BIDSLayer(const char* name, HyPerCol * hc); // The constructor called by other methods
   int updateState(double timef, double dt);
//   int updateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead, bool spiking, unsigned int * active_indices, unsigned int * num_active);
   int findFlag(int numMatrixCol, int numMatrixRow);
protected:
  BIDSLayer();
  int initialize(const char * name, HyPerCol * hc, const char * kernel_name);
  // other methods and member variables
private:
  // other methods and member variables
};
}

#endif /* INITBIDSLAYER_HPP_ */
