/*
 * GapLayer.hpp
 * can be used to implement gap junctions
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#ifndef GAPLAYER_HPP_
#define GAPLAYER_HPP_

#include "HyPerLayer.hpp"
#include "LIFGap.hpp"

namespace PV {

// CloneLayer can be used to implement gap junctions between spiking neurons
class GapLayer: public HyPerLayer {
public:
   GapLayer(const char * name, HyPerCol * hc, LIFGap * clone);
   virtual ~GapLayer();
   int initialize(LIFGap * clone);
   virtual int updateV();
   virtual int setActivity();
   LIFGap * sourceLayer;
};

}

#endif /* GAPLAYER_HPP_ */
