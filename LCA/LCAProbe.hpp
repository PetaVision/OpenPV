/*
 * LCAProbe.hpp
 *
 *  Created on: Oct 2, 2012
 *      Author: pschultz
 */

#ifndef LCAPROBE_HPP_
#define LCAPROBE_HPP_

#include <src/io/PointProbe.hpp>
#include "LCALayer.hpp"

namespace PV {

class LCAProbe: public PV::PointProbe {
public:
   LCAProbe(const char * filename, HyPerLayer * layer, int xLoc, int yLoc, int fLoc, const char * msg);
   LCAProbe(HyPerLayer * layer, int xLoc, int yLoc, int fLoc, const char * msg);
   virtual ~LCAProbe();

protected:
   int initLCAProbe(const char * filename, HyPerLayer * layer, int xLoc, int yLoc, int fLoc, const char * msg);
   virtual int writeState(double timed, HyPerLayer * l, int k, int kex);

private:

};

} /* namespace PV */
#endif /* LCAPROBE_HPP_ */
