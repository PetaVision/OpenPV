/*
 * CloneLayer.hpp
 * can be used to implement gap junctions
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#ifndef CLONELAYER_HPP_
#define CLONELAYER_HPP_

#include "HyPerLayer.hpp"

namespace PV {

// CloneLayer can be used to implement gap junctions between spiking neurons
class CloneLayer: public HyPerLayer {
public:
   CloneLayer(const char * name, HyPerCol * hc, HyPerLayer * clone);
   virtual ~CloneLayer();
   int initialize(HyPerLayer * clone);
   virtual int updateV();
   virtual int setActivity();
   HyPerLayer * sourceLayer;
};

}

#endif /* CLONELAYER_HPP_ */
