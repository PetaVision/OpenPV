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

   virtual ~HMaxSimple();

   virtual int recvSynapticInput(HyPerConn* conn, const PVLayerCube* activity, int neighbor);
   virtual int updateState(double time, double dt);

   virtual int initFinish(int colId, int colRow, int colCol);

   virtual int outputState(double time, bool last=false);

protected:
   HMaxSimple();
   int initialize(const char * name, HyPerCol * hc);

private:
   int initialize_base();

};

}

#endif /* SIMPLEPOGGIO_H_ */
