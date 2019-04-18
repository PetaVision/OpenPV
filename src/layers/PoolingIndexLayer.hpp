/*
 * PoolingIndexLayer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#ifndef POOLINGINDEXLAYER_HPP_
#define POOLINGINDEXLAYER_HPP_

#include "HyPerLayer.hpp"

namespace PV {

class PoolingIndexLayer : public HyPerLayer {
  public:
   PoolingIndexLayer(const char *name, HyPerCol *hc);
   virtual ~PoolingIndexLayer();
   bool activityIsSpiking() override { return false; }
   virtual int requireChannel(int channelNeeded, int *numChannelsResult) override;

  protected:
   PoolingIndexLayer();
   int initialize(const char *name, HyPerCol *hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_dataType(enum ParamsIOFlag ioFlag) override;
   virtual int resetGSynBuffers(double timef, double dt) override;

  private:
   int initialize_base();
}; // end of class PoolingIndexLayer

} // end namespace PV

#endif /* ANNLAYER_HPP_ */
