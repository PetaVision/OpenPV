/*
 * LayerData.hpp
 *
 *  Created on: Jan 31, 2011
 *      Author: manghel
 */

#ifndef LAYERDATA_HPP_
#define LAYERDATA_HPP_

#include "LayerProbe.hpp"
#include "../include/pv_types.h"




enum DataType{
   TYPE_VTH = 0,
   TYPE_WMAX = 1,    // extended
   TYPE_R = 2,       // restricted
   TYPE_VTHREST = 3, // restricted
};

namespace PV {

class HyPerCol;

class LayerData: public LayerProbe {
public:
   LayerData(DataType data_type, bool append);

   virtual int outputState(float time, HyPerLayer * l) ;

protected:
   DataType data_type;
   bool  append;
   bool  extended;
   bool  write_header;
   bool  open_file;
};

}

#endif /* LAYERDATA_HPP_ */
