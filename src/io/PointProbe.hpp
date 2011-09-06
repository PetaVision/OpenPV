/*
 * PointProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: rasmussn
 */

#ifndef POINTPROBE_HPP_
#define POINTPROBE_HPP_

#include "LayerProbe.hpp"

namespace PV {

class PointProbe: public PV::LayerProbe {
public:
   PointProbe(const char * filename, HyPerCol * hc, int xLoc, int yLoc, int fLoc, const char * msg);
   PointProbe(int xLoc, int yLoc, int fLoc, const char * msg);
   virtual ~PointProbe();

   virtual int outputState(float time, HyPerLayer * l);

   void setSparseOutput(bool flag) {sparseOutput = flag;}

protected:
   int xLoc;
   int yLoc;
   int fLoc;
   char * msg;

   bool sparseOutput;

   virtual int writeState(float time, HyPerLayer * l, int k, int kex);
};

}

#endif /* POINTPROBE_HPP_ */
