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
   virtual void ioParam_sparseLayer(enum ParamsIOFlag ioFlag);

   /** @} */ // end of PublisherComponent parameters

  public:
   PublisherComponent(char const *name, PVParams *params, Communicator const *comm);
   virtual ~PublisherComponent();

   bool getSparseLayer() const { return mSparseLayer; }

  protected:
   PublisherComponent();

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
}; // class PublisherComponent

} // namespace PV

#endif // SPARSELAYERFLAGPUBLISHERCOMPONENT_HPP_
