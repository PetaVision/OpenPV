/*
 * BaseDelivery.hpp
 *
 *  Created on: Nov 17, 2017
 *      Author: Pete Schultz
 */

#ifndef BASEDELIVERY_HPP_
#define BASEDELIVERY_HPP_

#include "columns/BaseObject.hpp"
#include "components/ConnectionData.hpp"
#include "layers/HyPerLayer.hpp"

namespace PV {

class BaseDelivery : public BaseObject {
  protected:
   /**
    * List of parameters needed from the BaseDelivery class
    * @name BaseDelivery Parameters
    * @{
    */

   /**
    * @brief channelCode: Specifies which channel in the post layer this connection is attached to
    * @details Channels can be -1 for no update, or >= 0 for channel number. <br />
    * 0 is excitatory, 1 is inhibitory
    */
   virtual void ioParam_channelCode(enum ParamsIOFlag ioFlag);

   /**
    * @brief receiveGpu: If PetaVision was compiled with GPU acceleration and this flag is set to
    * true, the connection uses the GPU to update the postsynaptic layer's GSyn.
    * If compiled without GPU acceleration, it is an error to set this flag to true.
    */
   virtual void ioParam_receiveGpu(enum ParamsIOFlag ioFlag);
   /** @} */ // end of BaseDelivery parameters

  public:
   BaseDelivery(char const *name, HyPerCol *hc);

   virtual ~BaseDelivery() {}

   virtual void deliver() {}

   virtual void deliverUnitInput(float *recvBuffer) {}

   /**
    * A virtual method to indicate whether the presynaptic layer's input is ready to be delivered.
    */
   virtual bool isAllInputReady() { return true; }

   ChannelType getChannelCode() const { return mChannelCode; }
   bool getReceiveGpu() const { return mReceiveGpu; }
   HyPerLayer *getPreLayer() const { return mPreLayer; }
   HyPerLayer *getPostLayer() const { return mPostLayer; }

  protected:
   BaseDelivery() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

  protected:
   ChannelType mChannelCode = CHANNEL_EXC;
   bool mReceiveGpu         = false;

   ConnectionData *mConnectionData = nullptr;
   HyPerLayer *mPreLayer           = nullptr;
   HyPerLayer *mPostLayer          = nullptr;
   // Rather than the layers, should we store the buffers and the PVLayerLoc data?
};

} // namespace PV

#endif // BASEDELIVERY_HPP_
