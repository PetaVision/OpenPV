/*
 * PublisherComponent.hpp
 *
 *  Created on: Dec 4, 2018
 *      Author: peteschultz
 */

#ifndef SPARSELAYERFLAGPUBLISHERCOMPONENT_HPP_
#define SPARSELAYERFLAGPUBLISHERCOMPONENT_HPP_

#include "components/BasePublisherComponent.hpp"

#include "columns/Publisher.hpp"
#include "utils/Timer.hpp"

namespace PV {

/**
 * A component to output a layer's activity. It creates a .pvp file whose name is the
 * component's name followed by the suffix ".pvp" in the output directory.
 * It then responds to the LayerOutputStateMessage by appending a frame to the .pvp file
 * with the current activity state (retrieved from the publisher).
 * The boolean parameter sparseLayer controls whether the layer is sparse or dense.
 */
class PublisherComponent : public BasePublisherComponent {
  protected:
   /**
    * List of parameters needed from the PublisherComponent class
    * @name HyPerLayer Parameters
    * @{
    */

   /**
    * @brief sparseLayer: Specifies if the layer should be considered sparse for optimization and
    * output
    */
   virtual void ioParam_sparseLayer(enum ParamsIOFlag ioFlag);

   /** @} */ // end of PublisherComponent parameters

  public:
   PublisherComponent(char const *name, PVParams *params, Communicator *comm);
   virtual ~PublisherComponent();

   bool getSparseLayer() const { return mSparseLayer; }

  protected:
   PublisherComponent();

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
}; // class PublisherComponent

} // namespace PV

#endif // SPARSELAYERFLAGPUBLISHERCOMPONENT_HPP_
