/*
 * LayerInputDelivery.hpp
 *
 *  Created on: Sept 17, 2018
 *      Author: Pete Schultz
 */

#ifndef LAYERINPUTDELIVERY_HPP_
#define LAYERINPUTDELIVERY_HPP_

#include "columns/BaseObject.hpp"

namespace PV {

enum ChannelType {
   CHANNEL_EXC      = 0,
   CHANNEL_INH      = 1,
   CHANNEL_INHB     = 2,
   CHANNEL_GAP      = 3,
   CHANNEL_NORM     = 4,
   CHANNEL_NOUPDATE = -1
};

/**
 * The parent class of all delivery classes, to provide the minimal interface needed by
 * LayerInputBuffer. There are two parameters, channelCode and receiveGpu; and three
 * virtual methods, isAllInputReady(), deliver(), and deliverUnitInput().
 * Even the interaction with a ConnectionData component, to provide pre and post layers,
 * is provided by BaseDelivery class, which derives from LayerInputDelivery.
 */
class LayerInputDelivery : public BaseObject {
  protected:
   /**
    * List of parameters needed from the LayerInputDelivery class
    * @name LayerInputDelivery Parameters
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
   /** @} */ // end of LayerInputDelivery parameters

  public:
   LayerInputDelivery(char const *name, PVParams *params, Communicator const *comm);

   virtual ~LayerInputDelivery() {}

   virtual void deliver(float *destBuffer) {}

   virtual void deliverUnitInput(float *recvBuffer) {}

   /**
    * A virtual method to indicate whether the presynaptic layer's input is ready to be delivered.
    */
   virtual bool isAllInputReady() const { return true; }

   ChannelType getChannelCode() const { return mChannelCode; }
   bool getReceiveGpu() const { return mReceiveGpu; }

  protected:
   LayerInputDelivery() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

  protected:
   ChannelType mChannelCode = CHANNEL_EXC;
   bool mReceiveGpu         = false;
};

} // namespace PV

#endif // LAYERINPUTDELIVERY_HPP_
