/*
 * TestLayer.cpp *
 *  Created on: Oct 24, 2008
 *      Author: rasmussn
 */

#include "TestLayer.hpp"

namespace PV {

TestLayer::TestLayer(const char* name, HyPerCol * hc) : HyPerLayer(name, hc)
{
}

TestLayer::~TestLayer()
{
}

int TestLayer::recvSynapticInput(HyPerConn * conn, PVLayerCube * cube, int neighbor)
{
   HyPerLayer * src = conn->preSynapticLayer();
#ifdef DEBUG_OUTPUT
   fprintf(stderr, "[%d]: TestLayer::recvSynapticInput: to layer %d from layer %d, cube=%p cube->data=%p\n",
           clayer->columnId, clayer->layerId, src->clayer->layerId, cube, cube->data);
   fprintf(stderr, "[%d]:      (nx,ny)=(%f,%f) (kx0,ky0)=(%f,%f) numItems=%d\n",
           clayer->columnId, cube->loc.nx, cube->loc.ny, cube->loc.kx0, cube->loc.ky0,
           cube->numItems);
   if (cube->numItems < 10) return -1;
   for (int i = 0; i < 5; i++) {
      fprintf(stderr, "[%d]:      %f\n", clayer->columnId, cube->data[i]);
   }
#endif

   float* V = clayer->V;
   for (int i = 0; i < cube->numItems; i++) {
      V[i] = 
      fprintf(stderr, "[%d]:      %f\n", clayer->columnId, cube->data[i]);
   }

   return 0;
}

int TestLayer::updateState(float time)
{
#ifdef DEBUG_OUTPUT
   fprintf(stderr, "[%d]: TestLayer::updateState:\n", clayer->columnId);
#endif
   return 0;
}

int TestLayer::initFinish()
{
#ifdef DEBUG_OUTPUT
   fprintf(stderr, "[%d]: TestLayer::initFinish:\n", clayer->columnId);
#endif
   HyPerLayer::initFinish();
   return 0;
}

int TestLayer::setParams(int numParams, float* params)
{
#ifdef DEBUG_OUTPUT
   fprintf(stderr, "[%d]: TestLayer::setParams: numParams=%d\n",
           clayer->columnId, numParams);
#endif
   return 0;
}

int TestLayer::outputState(float time)
{
#ifdef DEBUG_OUTPUT
   fprintf(stderr, "[%d]: TestLayer::outputState: time=%f\n", clayer->columnId, time);
#endif
   return 0;
}


}
