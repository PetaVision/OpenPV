/*
 * DenseLayerOutputComponent.hpp
 *
 *  Created on: Dec 3, 2018
 *      Author: peteschultz
 */

#ifndef DENSELAYEROUTPUTCOMPONENT_HPP_
#define DENSELAYEROUTPUTCOMPONENT_HPP_

#include "components/LayerOutputComponent.hpp"

namespace PV {

/**
 * A subclass of LayerOutputComponent where the sparseLayer parameter is not used and the
 * output is always to a dense .pvp file.
 */
class DenseLayerOutputComponent : public LayerOutputComponent {
  protected:
   /**
    * List of parameters needed from the DenseLayerOutputComponent class
    * @name HyPerLayer Parameters
    * @{
    */

   /**
    * @brief sparseLayer: DenseLayerOutputComponent does not read sparseLayer.
    * The component behaves like a LayerOutputComponent with sparseLayer set to false,
    * and it is an error to set the sparseLayer parameter to true.
    */
   virtual void ioParam_sparseLayer(enum ParamsIOFlag ioFlag) override;

   /** @} */ // end of DenseLayerOutputComponent parameters

  public:
   DenseLayerOutputComponent(char const *name, PVParams *params, Communicator *comm);
   virtual ~DenseLayerOutputComponent();

  protected:
   DenseLayerOutputComponent();

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType() override;
}; // class DenseLayerOutputComponent

} // namespace PV

#endif // DENSELAYEROUTPUTCOMPONENT_HPP_
