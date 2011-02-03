/*
 * PointLIFProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: rasmussn
 */

#ifndef POINTPROBE_HPP_
#define POINTPROBE_HPP_

#include "LayerProbe.hpp"

namespace PV {

class PointLIFProbe: public PV::LayerProbe {
public:
   PointLIFProbe(const char * filename, int xLoc, int yLoc, int fLoc, const char * msg);
   PointLIFProbe(int xLoc, int yLoc, int fLoc, const char * msg);
   virtual ~PointLIFProbe();

   virtual int outputState(float time, HyPerLayer * l);

   void setSparseOutput(bool flag) {sparseOutput = flag;}

protected:
   int xLoc;
   int yLoc;
   int fLoc;
   char * msg;

   bool sparseOutput;
};

}

#endif /* POINTPROBE_HPP_ */
