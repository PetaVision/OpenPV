/*
 * LayerDataInterface.hpp
 *
 *  Created on: Jan 16, 2010
 *      Author: rasmussn
 */

#ifndef LAYERDATAINTERFACE_HPP_
#define LAYERDATAINTERFACE_HPP_

#include "PVLayer.h"

namespace PV {

/*
 * Interface providing access to a layer's data
 */

class LayerDataInterface {
public:
   LayerDataInterface();
   virtual ~LayerDataInterface();

   virtual const PVLayerLoc * getLayerLoc() = 0;
   virtual const pvdata_t * getLayerData() = 0;
   virtual bool isExtended() = 0;
   virtual int copyToInteriorBuffer(unsigned char * buf) = 0;
};

} // namespace PV

#endif /* LAYERDATAINTERFACE_HPP_ */
