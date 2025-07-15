/*
 * PublisherComponent.hpp
 *
 *  Created on: Dec 4, 2018
 *      Author: peteschultz
 */

#ifndef PUBLISHERCOMPONENT_HPP_
#define PUBLISHERCOMPONENT_HPP_

#include "components/BasePublisherComponent.hpp"

#include "columns/Publisher.hpp"

namespace PV {

/**
 * A derived class of BasePublisherComponent, that adds a Boolean parameter sparseLayerFlag.
 * If true, the Publisher will maintain a SparseEntry table of nonzero values as well as
 * the dense activity.
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
   virtual void ioParam_sparseLayer(ParamsIOSwitch ioSwitch);

   /** @} */ // end of PublisherComponent parameters

  public:
   PublisherComponent(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~PublisherComponent();

  protected:
   PublisherComponent();

   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;
}; // class PublisherComponent

} // namespace PV

#endif // PUBLISHERCOMPONENT_HPP_
