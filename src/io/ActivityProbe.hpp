/*
 * ActivityProbe.hpp
 *
 *  Created on: Oct 20, 2009
 *      Author: travel
 */

#ifndef ACTIVITYPROBE_HPP_
#define ACTIVITYPROBE_HPP_

#include "PVLayerProbe.hpp"
#include "../columns/HyPerCol.hpp"

namespace PV {

class ActivityProbe: public PV::PVLayerProbe {
public:
   ActivityProbe(const char * filename, HyPerCol * hc, const LayerLoc * loc, int f);
   virtual ~ActivityProbe();

   virtual int outputState(float time, PVLayer * l);

private:
   HyPerCol * parent;
   FILE * outfp;
   long outFrame;
   pvdata_t * outBuf;
};

}

#endif /* ACTIVITYPROBE_HPP_ */
