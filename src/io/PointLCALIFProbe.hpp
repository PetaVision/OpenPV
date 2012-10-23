/*
 * LCALIFProbe.hpp
 *
 *  Created on: Oct 4, 2012
 *      Author: pschultz
 */

#ifndef POINTLCALIFPROBE_HPP_
#define POINTLCALIFPROBE_HPP_

#include "PointLIFProbe.hpp"

namespace PV {

class PointLCALIFProbe: public PV::PointLIFProbe {
public:
   PointLCALIFProbe(const char * filename, HyPerLayer * layer, int xLoc, int yLoc, int fLoc, const char * msg);
   virtual ~PointLCALIFProbe();
   virtual int writeState(double timed, HyPerLayer * l, int k, int kex);
};

} /* namespace PV */
#endif /* POINTLCALIFPROBE_HPP_ */
