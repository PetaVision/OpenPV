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

class ODDLayer: public PV::ANNLayer {
public:
   ODDLayer(const char* name, HyPerCol * hc);
   ODDLayer(const char* name, HyPerCol * hc, PVLayerType type);
   virtual ~ODDLayer();
   virtual int updateState(float time, float dt);

protected:
   ODDLayer();
   int initialize(const char * name, HyPerCol * hc);

private:
   int initialize_base();
};

}

#endif /* GEISLERLAYER_HPP_ */
