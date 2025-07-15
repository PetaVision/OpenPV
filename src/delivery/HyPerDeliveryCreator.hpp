/*
 * HyPerDeliveryCreator.hpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#ifndef HYPERDELIVERYCREATOR_HPP_
#define HYPERDELIVERYCREATOR_HPP_

#include "columns/BaseObject.hpp"
#include "delivery/HyPerDelivery.hpp"

namespace PV {

/**
 * A class to take the pvpatchAccumulateType, updateGSynFromPostPerspective, and receiveGpu
 * parameters; and determine the type of delivery object specified by those params.
 * The create method creates and returns a HyPerDelivery object of the appropriate derived class.
 * Note that it does not retain ownership of the object created by create(). The usual
 * use case is that a HyPerConn will create a HyPerDeliveryCreator, call its create() method,
 * and add the HyPerDeliveryCreator and HyPerDelivery objects to its table of components.
 */
class HyPerDeliveryCreator : public BaseObject {
  protected:
   /**
    * List of parameters needed from the HyPerDeliveryCreator class
    * @name HyPerDeliveryCreator Parameters
    * @{
    */

   /**
    * @brief receiveGpu: This parameter determines whether the created HyPerDelivery object
    * should use the GPU.
    */
   virtual void ioParam_receiveGpu(ParamsIOSwitch ioSwitch);

   /**
    * @brief pvpatchAccumulateType: Specifies the method to accumulate synaptic input
    * @details Possible choices are
    * - convolve: Accumulates through convolution
    * - stochastic: Accumulates through stochastic release
    *
    * Defaults to convolve.
    */
   virtual void ioParam_pvpatchAccumulateType(ParamsIOSwitch ioSwitch);

   /**
    * @brief updateGSynFromPostPerspective: Specifies if the connection should push from pre or pull
    * from post.
    * @details: If set to true, the connection loops over postsynaptic neurons, and each
    * post-neuron pulls from its receptive field. This avoids issues of collisions when
    * parallelizing, but is not able to take advantage of a sparse pre-layer.
    *
    * If false, the connection loops over presynaptic neurons, and each pre-neuron pushes to its
    * region of influence. This allows efficiency for sparse pre-layers, but requires extra memory
    * to manage potential collisions as multiple pre-neurons write to the same post-neuron.
    */
   virtual void ioParam_updateGSynFromPostPerspective(ParamsIOSwitch ioSwitch);
   /** @} */ // End of list of HyPerDeliveryCreator parameters.

  public:
   enum AccumulateType { UNDEFINED, CONVOLVE, STOCHASTIC };

   HyPerDeliveryCreator(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual ~HyPerDeliveryCreator();

   HyPerDelivery *create();

   AccumulateType getAccumulateType() const { return mAccumulateType; }

   bool getUpdateGSynFromPostPerspective() const { return mUpdateGSynFromPostPerspective; }

   bool getReceiveGpu() const { return mReceiveGpu; }

  protected:
   HyPerDeliveryCreator();

   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

   // Data members
  protected:
   AccumulateType mAccumulateType = CONVOLVE;

   std::string mAccumulateTypeString;
   bool mUpdateGSynFromPostPerspective = false;
   bool mReceiveGpu                    = false;

   HyPerDelivery *mDeliveryIntern = nullptr;
}; // end class HyPerDeliveryCreator

} // end namespace PV

#endif // HYPERDELIVERYCREATOR_HPP_
