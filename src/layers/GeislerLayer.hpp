/*
 * GeislerLayer.hpp
 *
 *  Created on: Apr 21, 2010
 *      Author: gkenyon
 */

#ifndef GEISLERLAYER_HPP_
#define GEISLERLAYER_HPP_

#include "ANNLayer.hpp"

namespace PV {

class GeislerLayer: public PV::ANNLayer {
public:
   GeislerLayer(const char* name, HyPerCol * hc);
   GeislerLayer(const char* name, HyPerCol * hc, PVLayerType type);
   virtual int updateState(float time, float dt);

private:
};

}

#endif /* GEISLERLAYER_HPP_ */
