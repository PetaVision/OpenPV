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

   virtual int outputState(double time);

protected:
   ActivityProbe();
   int initActivityProbe(const char * filename, HyPerLayer * layer);
   virtual int initOutputStream(const char * filename, HyPerLayer * layer);

private:
   int initActivityProbe_base();

private:
   long outFrame;
   pvdata_t * outBuf;
};

}

#endif /* ACTIVITYPROBE_HPP_ */
