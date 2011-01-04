/*
 * Example.h
 *
 *  Created on: Oct 19, 2008
 *      Author: rasmussn
 */

#ifndef EXAMPLE_HPP_
#define EXAMPLE_HPP_

#include "HyPerLayer.hpp"

namespace PV
{

class Example: public PV::HyPerLayer
{
public:
   Example(const char* name, HyPerCol * hc);

   virtual int recvSynapticInput(HyPerConn* conn, PVLayerCube* activity, int neighbor);
   virtual int updateState(float time, float dt);

   virtual int initFinish(int colId, int colRow, int colCol);

   virtual int outputState(float time);
};

}

#endif /* EXAMPLE_HPP_ */
