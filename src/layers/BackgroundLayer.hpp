/*
 * BackgroundLayer.hpp
 * Background layer clones a layer, adds 1 more feature in the 0 feature idx, and sets the activity
 * to the NOR of everything of that feature (none of the above category)
 *
 *  Created on: 4/16/15
 *  slundquist
 */

#ifndef BACKGROUNDLAYER_HPP_
#define BACKGROUNDLAYER_HPP_

#include "CloneVLayer.hpp"

namespace PV {

// CloneLayer can be used to implement Sigmoid junctions between spiking neurons
class BackgroundLayer : public CloneVLayer {
  public:
   BackgroundLayer(const char *name, HyPerCol *hc);
   virtual ~BackgroundLayer();
   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual int allocateV() override;
   virtual int updateState(double timef, double dt) override;
   virtual int setActivity() override;

  protected:
   BackgroundLayer();
   int initialize(const char *name, HyPerCol *hc);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   void ioParam_repFeatureNum(enum ParamsIOFlag ioFlag);

  private:
   int initialize_base();
   int repFeatureNum;
}; // class BackgroundLayer

} // namespace PV

#endif /* CLONELAYER_HPP_ */
