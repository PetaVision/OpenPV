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
   LinearActivityProbe(HyPerCol * hc, PVDimType dim, int linePos, int f);
   LinearActivityProbe(const char * filename, HyPerCol * hc, PVDimType dim, int linePos, int f);

   virtual int outputState(float time, HyPerLayer * l);

protected:
   HyPerCol * hc;
   PVDimType dim;
   int linePos;
   int f;
};

} // namespace PV

#endif /* LINEARACTIVITYPROBE_HPP_ */
