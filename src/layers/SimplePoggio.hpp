/*
 * SimplePoggio.h
 *
 *  Created on: Nov 20, 2008
 *      Author: bjt
 */

#ifndef SIMPLEPOGGIO_H_
#define SIMPLEPOGGIO_H_

#include "HyPerLayer.hpp"

namespace PV {

class SimplePoggio: public PV::HyPerLayer {
public:
   SimplePoggio(const char * name, HyPerCol * hc);

   virtual int recvSynapticInput(HyPerConn* conn, PVLayerCube* activity, int neighbor);
   virtual int updateState(float time, float dt);

   virtual int initFinish(int colId, int colRow, int colCol);

   virtual int setParams(int numParams, float* params);

   virtual int outputState(float time);
};

}

#endif /* SIMPLEPOGGIO_H_ */
