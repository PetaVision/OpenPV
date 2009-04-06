/*
 * PointProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: rasmussn
 */

#ifndef POINTPROBE_HPP_
#define POINTPROBE_HPP_

#include "PVLayerProbe.hpp"

namespace PV {

class PointProbe: public PV::PVLayerProbe {
public:
   PointProbe(const char * filename, int xLoc, int yLoc, int fLoc, const char * msg);
   PointProbe(int xLoc, int yLoc, int fLoc, const char * msg);
   virtual ~PointProbe();

   virtual int outputState(float time, PVLayer * l);

protected:
   int xLoc;
   int yLoc;
   int fLoc;
   char * msg;
};

}

#endif /* POINTPROBE_HPP_ */
