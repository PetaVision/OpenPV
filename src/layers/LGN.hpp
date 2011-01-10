/*
 * LGN.h
 *
 *  Created on: Jul 30, 2008
 *      Author: rasmussn
 */

#ifndef LGN_HPP_
#define LGN_HPP_

#include "HyPerLayer.hpp"

namespace PV {

class LGN: public PV::HyPerLayer {
public:
   LGN(const char * name, HyPerCol * hc);

   virtual int recvSynapticInput(HyPerConn * conn, PVLayerCube * activity, int neighbor);
   virtual int updateState(float time, float dt);

   virtual int initFinish(int colId, int colRow, int colCol);

   virtual int outputState(float time);
};

}

#endif /* LGN_HPP_ */
