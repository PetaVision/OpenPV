/*
 * TestLayer.hpp
 *
 *  Created on: Oct 24, 2008
 *      Author: rasmussn
 */

#ifndef TESTLAYER_HPP_
#define TESTLAYER_HPP_

#include "../src/layers/HyPerLayer.hpp"

namespace PV {

class TestLayer: public PV::HyPerLayer {
public:
   TestLayer(const char* name, HyPerCol * hc);
   virtual ~TestLayer();

   virtual int recvSynapticInput(HyPerConn * conn, PVLayerCube* cube, int neighbor);
   virtual int updateState(double time, double dt);

   virtual int initFinish();

   virtual int setParams(int numParams, float* params);

   virtual int outputState(double time);

};

}

#endif /* TESTLAYER_HPP_ */
