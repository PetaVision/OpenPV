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
   ActivityProbe(const char * probeName, HyPerCol * hc);
   virtual ~ActivityProbe();

   virtual int outputState(double time);

protected:
   ActivityProbe();
   int initActivityProbe(const char * probeName, HyPerCol * hc);
   virtual int initOutputStream(const char * filename);

private:
   int initActivityProbe_base();

private:
   long outFrame;
   pvdata_t * outBuf;
};

}

#endif /* ACTIVITYPROBE_HPP_ */
