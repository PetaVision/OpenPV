/*
 * HMaxSimple.h
 *
 *  Created on: Nov 20, 2008
 *      Author: bjt
 */

#ifndef HMAXSIMPLE_H_
#define HMAXSIMPLE_H_

#include "HyPerLayer.hpp"

namespace PV {

class HMaxSimple: public PV::HyPerLayer {
public:
   HMaxSimple(const char * name, HyPerCol * hc);

   virtual int recvSynapticInput(HyPerConn* conn, PVLayerCube* activity, int neighbor);
   virtual int updateState(float time, float dt);

   virtual int initFinish(int colId, int colRow, int colCol);

   virtual int outputState(float time);
};

}

#endif /* SIMPLEPOGGIO_H_ */
