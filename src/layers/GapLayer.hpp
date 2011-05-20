/*
 * CloneLayer.hpp
 * can be used to implement gap junctions
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#ifndef GAPLAYER_HPP_
#define GAPLAYER_HPP_

#include "HyPerLayer.hpp"
#include "LIF.hpp"

namespace PV {

// CloneLayer can be used to implement gap junctions between spiking neurons
class GapLayer: public HyPerLayer {
public:
   GapLayer(const char * name, HyPerCol * hc, LIF * clone);
   virtual ~GapLayer();
   int initialize(LIF * clone);
   virtual int updateV();
   virtual int setActivity();
   LIF * sourceLayer;
};

}

#endif /* CLONELAYER_HPP_ */
