/*
 * SigmaPiLayer.h
 *
 *  Created on: Sep 3, 2011
 *      Author: gkenyon
 */

#ifndef SIGMAPILAYER_H_
#define SIGMAPILAYER_H_

#include "ANNLayer.hpp"

namespace PV {

class CliqueLayer: public PV::ANNLayer {
public:
   CliqueLayer();
   virtual ~CliqueLayer();
   CliqueLayer(const char* name, HyPerCol * hc);
   CliqueLayer(const char* name, HyPerCol * hc, PVLayerType type);
   virtual int recvSynapticInput(HyPerConn * conn, PVLayerCube * cube, int neighbor);
   virtual int updateState(float time, float dt);
};

} /* namespace PV */
#endif /* SIGMAPILAYER_H_ */
