/*
 * Simple.hpp
 *
 *  Created on: Oct 10, 2009
 *      Author: travel
 */

#ifndef SIMPLE_HPP_
#define SIMPLE_HPP_

#include "HyPerLayer.hpp"

namespace PV {

class Simple: public PV::HyPerLayer {

public:
   Simple(const char* name, HyPerCol * hc);

   virtual int recvSynapticInput(HyPerConn* conn, PVLayerCube* activity, int neighbor);
   virtual int reconstruct(HyPerConn* conn, PVLayerCube* activity);
   virtual int updateState(float time, float dt);

protected:
#ifdef PV_USE_OPENCL
   virtual int initializeThreadData();
   virtual int initializeThreadKernels();
#endif

};

}

#endif /* SIMPLE_HPP_ */
