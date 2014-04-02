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
   virtual int outputState(double timef);
   float getSparsity(){return sparsityVal;}
protected:
   SparsityLayerProbe();

private:
   float sparsityVal;
   int initSparsityLayerProbe_base();
};

}

#endif /* LAYERPROBE_HPP_ */
