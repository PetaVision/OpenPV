// InputVolumeLayer

#ifndef INPUTVOLUMELAYER_HPP__
#define INPUTVOLUMELAYER_HPP__

#include "HyPerLayer.hpp"
#include "components/ActivityComponent.hpp"
#include "components/LayerInputBuffer.hpp"
#include "components/LayerUpdateController.hpp"

namespace PV {

/**
 * A layer class for reading 
 */
class InputVolumeLayer : public HyPerLayer {
  public:
   InputVolumeLayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~InputVolumeLayer();

  protected:
   InputVolumeLayer() {}

   void initialize(const char *name, PVParams *params, Communicator const *comm);

   virtual LayerUpdateController *createLayerUpdateController() override;

   virtual LayerInputBuffer *createLayerInput() override;

   /**
    * Creates an InputVolumeActivityComponent object and adds it to the component table.
    * The InputVolumeActivityComponent object contains an InputVolumeActivityBuffer.
    */
   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif // INPUTVOLUMELAYER_HPP__
