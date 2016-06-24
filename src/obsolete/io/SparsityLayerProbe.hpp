/*
 * SparsityLayerProbe.h
 *
 *  Created on: Apr 2, 2014
 *      Author: slundquist
 */

#ifndef SPARSITYLAYERPROBE_HPP_
#define SPARSITYLAYERPROBE_HPP_

#include "LayerProbe.hpp"
#include "../layers/ANNLayer.hpp"

namespace PV{

class SparsityLayerProbe: public PV::LayerProbe{
// Methods
public:
   SparsityLayerProbe(const char * probeName, HyPerCol * hc);
   ~SparsityLayerProbe();
   virtual int communicateInitInfo();
   virtual int outputState(double timef);
   float getSparsity();
   double getUpdateTime();
   float getInitSparsityVal(){return initSparsityVal;}
   double getDeltaUpdateTime(){return deltaUpdateTime;}
   double getWindowSize(){return windowSize;}
protected:
   SparsityLayerProbe();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_windowSize(enum ParamsIOFlag ioFlag);
   virtual void ioParam_calcNNZ(enum ParamsIOFlag ioFlag);
   virtual void ioParam_initSparsityVal(enum ParamsIOFlag ioFlag);

private:
   void updateBufIndex();
   float* sparsityVals;
   double* timeVals;
   int initSparsityLayerProbe_base();
   int bufIndex;
   int bufSize;
   bool calcNNZ;
   double windowSize;
   double deltaUpdateTime;
   float initSparsityVal;
   ANNLayer * ANNTargetLayer;
};

}

#endif /* LAYERPROBE_HPP_ */
