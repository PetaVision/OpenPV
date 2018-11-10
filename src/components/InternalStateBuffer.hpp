/*
 * InternalStateBuffer.hpp
 *
 *  Created on: Sep 11, 2018
 *      Author: Pete Schultz
 */

#ifndef INTERNALSTATEBUFFER_HPP_
#define INTERNALSTATEBUFFER_HPP_

#include "components/RestrictedBuffer.hpp"
#include "initv/BaseInitV.hpp"

namespace PV {

/**
 * A component to contain the internal state (membrane potential) of a HyPerLayer.
 */
class InternalStateBuffer : public RestrictedBuffer {
  protected:
   /**
    * List of parameters needed from the InternalStateBuffer class
    * @name HyPerLayer Parameters
    * @{
    */

   /**
    * @brief initVType: Specifies how to initialize the V buffer.
    * @details Possible choices include
    * - @link ConstantV::ioParamsFillGroup ConstantV@endlink: Sets V to a constant value
    * - @link ZeroV::ioParamsFillGroup ZeroV@endlink: Sets V to zero
    * - @link UniformRandomV::ioParamsFillGroup UniformRandomV@endlink: Sets V with a uniform
    * distribution
    * - @link GaussianRandomV::ioParamsFillGroup GaussianRandomV@endlink: Sets V with a gaussian
    * distribution
    * - @link InitVFromFile::ioparamsFillGroup InitVFromFile@endlink: Sets V to specified pvp file
    *
    * Further parameters are needed depending on initialization type.
    */
   virtual void ioParam_InitVType(enum ParamsIOFlag ioFlag);

   /** @} */
  public:
   InternalStateBuffer(char const *name, HyPerCol *hc);

   virtual ~InternalStateBuffer();

  protected:
   InternalStateBuffer() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

  private:
  protected:
   BaseInitV *mInitVObject = nullptr;
   char *mInitVTypeString  = nullptr;
};

} // namespace PV

#endif // INTERNALSTATEBUFFER_HPP_
