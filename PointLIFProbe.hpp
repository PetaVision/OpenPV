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

   virtual int writeState(double timed, HyPerLayer * l, int k, int kex);

protected:
   PointLIFProbe();
   int initPointLIFProbe(const char * filename, HyPerLayer * layer, int xLoc, int yLoc, int fLoc, float writeStep, const char * msg);
   int initPointLIFProbe(const char * filename, HyPerLayer * layer, int xLoc, int yLoc, int fLoc, const char * msg);

private:
   int initPointLIFProbe_base();

protected:
   double writeTime;             // time of next output
   double writeStep;             // output time interval

};

}

#endif /* POINTLIFPROBE_HPP_ */
