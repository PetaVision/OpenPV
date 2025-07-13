/*
 * TestImageActivityBuffer.hpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#ifndef TESTIMAGEACTIVITYBUFFER_HPP_
#define TESTIMAGEACTIVITYBUFFER_HPP_

#include <components/ActivityBuffer.hpp>

namespace PV {

/**
 * TestImageActivityBuffer is the ActivityBuffer subclass for the TestImage layer.
 */
class TestImageActivityBuffer : public ActivityBuffer {
  protected:
   /**
    * List of parameters used by the TestImageActivityBuffer class
    * @name TestImage Parameters
    * @{
    */

   /**
    * @brief constantVal:
    * The activity of this layer is set to a constant value
    * input. If false, the retina treats the input like a HyPerLayer.
    */
   virtual void ioParam_constantVal(ParamsIOSwitch ioSwitch);

   /** @} */
  public:
   TestImageActivityBuffer(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual ~TestImageActivityBuffer();

   float getConstantVal() const { return mConstantVal; }

  protected:
   TestImageActivityBuffer() {}

   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   virtual void updateBufferCPU(double simTime, double deltaTime) override;

  private:
   float mConstantVal = 1.0f;
};

} // namespace PV

#endif // TESTIMAGEACTIVITYBUFFER_HPP_
