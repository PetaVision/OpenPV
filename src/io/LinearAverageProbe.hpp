/*
 * LinearAverageProbe.hpp
 *
 *  Created on: Apr 22, 2009
 *      Author: rasmussn
 */

#ifndef LINEARAVERAGEPROBE_HPP_
#define LINEARAVERAGEPROBE_HPP_

#include "LinearActivityProbe.hpp"

namespace PV {

class LinearAverageProbe: public PV::LinearActivityProbe {
public:
   LinearAverageProbe(HyPerLayer * layer, PVDimType dim, int f, const char * gifFile);
   LinearAverageProbe(const char * filename, HyPerLayer * layer, PVDimType dim, int f, const char * gifFile);
   virtual ~LinearAverageProbe();

   virtual int outputState(double timef);

protected:
   int initLinearAverageProbe(const char * filename, HyPerLayer * layer, PVDimType dim, int f, const char * gifFile);

   const char * gifFile;
   FILE * fpGif;
   int * locs;
};

}

#endif /* LINEARAVERAGEPROBE_HPP_ */
