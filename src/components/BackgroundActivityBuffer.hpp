/*
 * BackgroundActivityBuffer.hpp
 *
 *  Created on: 4/16/15
 *  slundquist
 */

#ifndef BACKGROUNDACTIVITYBUFFER_HPP_
#define BACKGROUNDACTIVITYBUFFER_HPP_

#include "components/ActivityBuffer.hpp"
#include "components/ComponentBuffer.hpp"
#include "components/InternalStateBuffer.hpp"
#include "layers/HyPerLayer.hpp"

namespace PV {

/**
 * BackgroundActivityBuffer clones a layer, adds 1 more feature in the 0 feature idx, and sets the
 * activity
 * to the NOR of everything of that feature (none of the above category)
 */
class BackgroundActivityBuffer : public ActivityBuffer {
  protected:
   /**
    * List of parameters used by the BackgroundActivityBuffer class
    * @name BackgroundLayer Parameters
    * @{
    */

   /**
    * repFeatureNum
    */
   void ioParam_repFeatureNum(enum ParamsIOFlag ioFlag);
   /** @} */
  public:
   BackgroundActivityBuffer(char const *name, PVParams *params, Communicator const *comm);

   virtual ~BackgroundActivityBuffer();

  protected:
   BackgroundActivityBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   /**
    * BackgroundActivityBuffer does not have an InternalStateBuffer.
    * However, the original layer must have the same nx and ny as the current layer,
    * and the original layer must have exactly one feature.
    */
   void checkDimensions() const;

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   virtual void updateBufferCPU(double simTime, double deltaTime) override;

   float calcGaussian(float x, float sigma);

  protected:
   int mRepFeatureNum = 1;

   BasePublisherComponent *mOriginalData = nullptr;
};

} // namespace PV

#endif // BACKGROUNDACTIVITYBUFFER_HPP_
