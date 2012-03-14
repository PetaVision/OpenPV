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
   PointLIFProbe(const char * filename, HyPerLayer * layer, int xLoc, int yLoc, int fLoc, const char * msg);
   PointLIFProbe(const char * filename, HyPerLayer * layer, int xLoc, int yLoc, int fLoc, float writeStep, const char * msg);

   PointLIFProbe(HyPerLayer * layer, int xLoc, int yLoc, int fLoc, const char * msg);
   PointLIFProbe(HyPerLayer * layer, int xLoc, int yLoc, int fLoc, float writeStep, const char * msg);

   virtual int writeState(float time, HyPerLayer * l, int k, int kex);

protected:
   float writeTime;             // time of next output
   float writeStep;             // output time interval

};

}

#endif /* POINTLIFPROBE_HPP_ */
