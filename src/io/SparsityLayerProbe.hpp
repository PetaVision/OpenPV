/*
 * SparsityLayerProbe.h
 *
 *  Created on: Apr 2, 2014
 *      Author: slundquist
 */

#ifndef SPARSITYLAYERPROBE_HPP_
#define SPARSITYLAYERPROBE_HPP_

#include "LayerProbe.hpp"

namespace PV{

class SparsityLayerProbe: public PV::LayerProbe{
// Methods
public:
   SparsityLayerProbe(const char * probeName, HyPerCol * hc);
   ~SparsityLayerProbe();
   virtual int communicateInitInfo();
   virtual int outputState(double timef);
   float getSparsity();
   double getLastUpdateTime();
protected:
   SparsityLayerProbe();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void ioParam_windowSize(enum ParamsIOFlag ioFlag);
   void ioParam_calcNNZ(enum ParamsIOFlag ioFlag);

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
};

}

#endif /* LAYERPROBE_HPP_ */
