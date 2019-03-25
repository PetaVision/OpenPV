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

#include "HyPerLayer.hpp"
#include "components/OriginalLayerNameParam.hpp"

namespace PV {

/**
 * Background layer clones a layer, adds 1 more feature in the 0 feature idx, and sets the activity
 * to the NOR of everything of that feature (none of the above category)
 */
class BackgroundLayer : public HyPerLayer {
  public:
   BackgroundLayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~BackgroundLayer();

  protected:
   BackgroundLayer();
   void initialize(const char *name, PVParams *params, Communicator const *comm);

   virtual void createComponentTable(char const *description) override;
   virtual LayerInputBuffer *createLayerInput() override;
   virtual ActivityComponent *createActivityComponent() override;
   virtual OriginalLayerNameParam *createOriginalLayerNameParam();
}; // class BackgroundLayer

} // namespace PV

#endif /* CLONELAYER_HPP_ */
