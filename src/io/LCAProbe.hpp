/*
 * LCAProbe.hpp
 *
 *  Created on: Oct 2, 2012
 *      Author: pschultz
 */

#ifndef LCAPROBE_HPP_
#define LCAPROBE_HPP_

#include "PointProbe.hpp"
#include "../layers/LCALayer.hpp"

namespace PV {

class LCAProbe: public PV::PointProbe {
public:
   LCAProbe(const char * filename, HyPerLayer * layer, int xLoc, int yLoc, int fLoc, const char * msg);
   LCAProbe(HyPerLayer * layer, int xLoc, int yLoc, int fLoc, const char * msg);
   virtual ~LCAProbe();

protected:
   LCAProbe();
   int initLCAProbe(const char * filename, HyPerLayer * layer, int xLoc, int yLoc, int fLoc, const char * msg);
   virtual int writeState(double timed, HyPerLayer * l, int k, int kex);

private:
   int initLCAProbe_base();
};

} /* namespace PV */
#endif /* LCAPROBE_HPP_ */
