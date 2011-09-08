/*
 * CliqueLayer.h
 *
 *  Created on: Sep 3, 2011
 *      Author: gkenyon
 */

#ifndef CLIQUELAYER_H_
#define CLIQUELAYER_H_

#include "../columns/HyPerCol.hpp"
#include "../connections/HyPerConn.hpp"
#include "ANNLayer.hpp"


namespace PV {

class CliqueLayer: public PV::ANNLayer {
public:
   CliqueLayer(const char* name, HyPerCol * hc);
   CliqueLayer(const char* name, HyPerCol * hc, PVLayerType type);
   virtual int recvSynapticInput(HyPerConn * conn, PVLayerCube * cube, int neighbor);
   virtual int updateState(float time, float dt);
   virtual int updateActiveIndices();
};

} /* namespace PV */

#endif /* CLIQUELAYER_H_ */
