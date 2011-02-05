/*
 * PointLIFProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: rasmussn
 */

#ifndef POINTLIFPROBE_HPP_
#define POINTLIFPROBE_HPP_

#include "PointProbe.hpp"

namespace PV {

class PointLIFProbe: public PointProbe {
public:
   PointLIFProbe(const char * filename, int xLoc, int yLoc, int fLoc, const char * msg);
   PointLIFProbe(int xLoc, int yLoc, int fLoc, const char * msg);

   virtual int outputState(float time, HyPerLayer * l);
};

}

#endif /* POINTLIFPROBE_HPP_ */
