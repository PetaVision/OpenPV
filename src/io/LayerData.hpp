/*
 * LayerData.hpp
 *
 *  Created on: Jan 31, 2011
 *      Author: manghel
 */

#ifndef LAYERDATA_HPP_
#define LAYERDATA_HPP_

#include "LayerProbe.hpp"
#include "../columns/HyPerCol.hpp"
#include "../include/pv_types.h"
#include "../io/fileio.hpp"
#include <assert.h>

namespace PV {

class LayerData: public LayerProbe {
public:
   LayerData(const char * filename, HyPerCol * hc, HyPerLayer * l, pvdata_t * data, bool append);

   virtual int outputState(float time, HyPerLayer * l) = 0;

protected:
   HyPerCol * parent;
   const pvdata_t * data;
   bool  append;

};

}

#endif /* LAYERDATA_HPP_ */
