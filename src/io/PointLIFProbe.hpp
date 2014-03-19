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
   PointLIFProbe(const char * probeName, HyPerCol * hc);

   virtual int writeState(double timed, HyPerLayer * l, int k, int kex);

protected:
   PointLIFProbe();
   int initPointLIFProbe(const char * probeName, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_writeStep(enum ParamsIOFlag ioFlag);

private:
   int initPointLIFProbe_base();

protected:
   double writeTime;             // time of next output
   double writeStep;             // output time interval

};

}

#endif /* POINTLIFPROBE_HPP_ */
