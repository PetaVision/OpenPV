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
   LinearAverageProbe();
   int initLinearAverageProbe(const char * filename, HyPerLayer * layer, PVDimType dim, int f, const char * gifFile);

private:
   int initLinearAverageProbe_base();

protected:
   const char * gifFilename;
   PV_Stream * gifFileStream;
};

}

#endif /* LINEARAVERAGEPROBE_HPP_ */
