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
   LinearAverageProbe(const char * probeName, HyPerCol * hc);
   virtual ~LinearAverageProbe();

   virtual int outputState(double timef);

protected:
   LinearAverageProbe();
   int initLinearAverageProbe(const char * probeName, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_gifFile(enum ParamsIOFlag ioFlag);

private:
   int initLinearAverageProbe_base();

protected:
   char * gifFilename;
   PV_Stream * gifFileStream;
};

}

#endif /* LINEARAVERAGEPROBE_HPP_ */
