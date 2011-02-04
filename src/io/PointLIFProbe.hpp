/*
 * PointLIFProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: rasmussn
 */

#ifndef POINTLIFPROBE_HPP_
#define POINTLIFPROBE_HPP_

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
