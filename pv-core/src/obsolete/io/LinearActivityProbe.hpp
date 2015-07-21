/*
 * ProbeActivityLinear.hpp
 *
 *  Created on: Mar 7, 2009
 *      Author: rasmussn
 */

#ifndef LINEARACTIVITYPROBE_HPP_
#define LINEARACTIVITYPROBE_HPP_

#include "LayerProbe.hpp"
#include "../columns/HyPerCol.hpp"

namespace PV {

typedef enum {
   DimX,
   DimY
} PVDimType;

class LinearActivityProbe: public PV::LayerProbe {
public:
   LinearActivityProbe(const char * filename, HyPerCol * hc);
   ~LinearActivityProbe();

   virtual int outputState(double timef);

protected:
   LinearActivityProbe();
   int initLinearActivityProbe(const char * filename, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_dim(enum ParamsIOFlag ioFlag);
   virtual void ioParam_linePos(enum ParamsIOFlag ioFlag);
   virtual void ioParam_f(enum ParamsIOFlag ioFlag);

private:
   int initLinearActivityProbe_base();

protected:
   HyPerCol * hc;
   char * dimString;
   PVDimType dim;
   int linePos;
   int f;
};

} // namespace PV

#endif /* LINEARACTIVITYPROBE_HPP_ */
