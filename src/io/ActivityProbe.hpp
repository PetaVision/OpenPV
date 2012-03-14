/*
 * ActivityProbe.hpp
 *
 *  Created on: Oct 20, 2009
 *      Author: travel
 */

#ifndef ACTIVITYPROBE_HPP_
#define ACTIVITYPROBE_HPP_

#include "LayerProbe.hpp"
#include "../columns/HyPerCol.hpp"

namespace PV {

class ActivityProbe: public PV::LayerProbe {
public:
   ActivityProbe(const char * filename, HyPerLayer * layer);
   virtual ~ActivityProbe();

   virtual int outputState(float time);

private:
   HyPerCol * parent;
   FILE * outfp;
   long outFrame;
   pvdata_t * outBuf;
};

}

#endif /* ACTIVITYPROBE_HPP_ */
