/*
 * LayerFunction.hpp
 *
 *  Created on: Nov 26, 2010
 *      Author: pschultz
 */

#ifndef LAYERFUNCTION_HPP_
#define LAYERFUNCTION_HPP_

#include <string.h>
#include "../include/pv_types.h"
#include "../layers/HyPerLayer.hpp"

namespace PV {

class LayerFunction {
public:
   LayerFunction(const char * name);
   virtual ~LayerFunction();
   virtual pvdata_t evaluate(float time, HyPerLayer * l, int batchIdx);
   virtual pvdata_t evaluateLocal(float time, HyPerLayer * l, int batchIdx) {return 0;}
#ifdef PV_USE_MPI
   virtual pvdata_t functionReduce(pvdata_t localValue, HyPerLayer * l);
#endif // PV_USE_MPI

   char * getName() {return name;}
   void setName(const char * name);

protected:
   char * name;
};

}  // end namespace PV

#endif /* LAYERFUNCTION_HPP_ */
