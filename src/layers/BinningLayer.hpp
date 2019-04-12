/*
 *  Created on: Jan 15, 2014
 *      Author: Sheng Lundquist
 */

#ifndef BINNINGLAYER_HPP_
#define BINNINGLAYER_HPP_

#include "HyPerLayer.hpp"
#include "components/OriginalLayerNameParam.hpp"

namespace PV {

/**
 * A layer class to sort another layer's output activity into bins.
 * The number of features of the BinningLayer is the number of bins.
 * The number of features of the other layer must be equal to 1.
 * In the simplest case, with binSigma==0, the region [binMin, binMax] is divided into nf equal
 * intervals, labeled 0, 1, ..., nf-1.
 * If the input activity at the location (x,y) falls into the bin labeled k, then the BinningLayer
 * has A(x,y,k) = 1 and A(x,y,k') = 0 if k != k'. If any input activity is less than the binMin
 * parameter, it is put in the bin labeled 0; similarly, input activity greater than binMax is put
 * in the bin labeled nf-1.
 *
 * Other parameters can modify the behavior of the BinningLayer, as described
 * in the documentation for those parameters.
 */

class BinningLayer : public HyPerLayer {
  public:
   BinningLayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~BinningLayer();

  protected:
   BinningLayer();
   void initialize(const char *name, PVParams *params, Communicator const *comm);

   virtual void fillComponentTable() override;
   virtual LayerInputBuffer *createLayerInput() override;
   virtual ActivityComponent *createActivityComponent() override;
   virtual OriginalLayerNameParam *createOriginalLayerNameParam();
}; // class BinningLayer

} // namespace PV

#endif /* CLONELAYER_HPP_ */
