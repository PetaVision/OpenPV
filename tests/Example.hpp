/*
 * Example.h
 *
 *  Created on: Oct 19, 2008
 *      Author: rasmussn
 */

#ifndef EXAMPLE_HPP_
#define EXAMPLE_HPP_

#include "../src/layers/HyPerLayer.hpp"

namespace PV
{

class Example: public PV::HyPerLayer
{
public:
   Example(const char* name, HyPerCol * hc);

#ifdef PV_USE_OPENCL
   virtual int initializeThreadBuffers();
   virtual int initializeThreadKernels();
#endif

   virtual int recvSynapticInput(HyPerConn* conn, const PVLayerCube* activity, int neighbor);
   virtual int updateState(float time, float dt);

   virtual int initFinish(int colId, int colRow, int colCol);

   virtual int outputState(float timef, bool last=false);
};

}

#endif /* EXAMPLE_HPP_ */
