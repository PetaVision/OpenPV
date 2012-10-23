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
   PointProbe(const char * filename, HyPerLayer * layer, int xLoc, int yLoc, int fLoc, const char * msg);
   PointProbe(HyPerLayer * layer, int xLoc, int yLoc, int fLoc, const char * msg);
   virtual ~PointProbe();

   virtual int outputState(double timef);

   // void setSparseOutput(bool flag) {sparseOutput = flag;}

protected:
   int xLoc;
   int yLoc;
   int fLoc;
   char * msg;

   // bool sparseOutput;

   int initPointProbe(const char * filename, HyPerLayer * layer, int xLoc, int yLoc, int fLoc, const char * msg);
   virtual int initFilePointer(const char * filename, HyPerLayer * layer);
   virtual int writeState(double timef, HyPerLayer * l, int k, int kex);

private:
   int initMessage(const char * msg);
};

}

#endif /* POINTPROBE_HPP_ */
