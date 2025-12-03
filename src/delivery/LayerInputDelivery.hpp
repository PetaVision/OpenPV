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
   virtual void ioParam_channelCode(ParamsIOSwitch ioSwitch);

   /**
    * @brief receiveGpu: If PetaVision was compiled with GPU acceleration and this flag is set to
    * true, the connection uses the GPU to update the postsynaptic layer's GSyn.
    * If compiled without GPU acceleration, it is an error to set this flag to true.
    */
   virtual void ioParam_receiveGpu(ParamsIOSwitch ioSwitch);
   /** @} */ // end of LayerInputDelivery parameters

  public:
   LayerInputDelivery(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual ~LayerInputDelivery() {}

   virtual void deliver(float *destBuffer) {}

   virtual void deliverUnitInput(float *recvBuffer) {}

   /**
    * A virtual method to indicate whether the presynaptic layer's input is ready to be delivered.
    */
   virtual bool isAllInputReady() const { return true; }

   ChannelType getChannelCode() const { return mChannelCode; }
   bool getReceiveGpu() const { return mReceiveGpu; }
   MPI_Op getMPIReductionOp() const { return mMPIReductionOp; }
   float getReductionMultiplier() const { return mReductionMultiplier; }

  protected:
   LayerInputDelivery() {}

   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

  protected:
   ChannelType mChannelCode = CHANNEL_EXC;
   bool mReceiveGpu         = false;
   MPI_Op mMPIReductionOp;

   // ReductionMultiplier is a hack to work around a subtle problem:
   // If a connection's pre and post are both broadcast layers, there should
   // not be any MPI reduction when delivering to the post synaptic layer.
   // If a connection has nonbroadcast pre and broadcastpost, we do need to
   // do an MPI reduction. However, there is the possibility that a
   // broadcast pre and nonbroadcast pre could accumulate to the same channel
   // of the same post layer, and we need to add them properly.
   // The post layer does not have direct access to whether its pre layers
   // are broadcast layers or not, but it's the post layer's input buffer
   // component that loops over the connections that connect to it.
   // In order not to require additional MPI reductions, or to further
   // complicate the accumulation of multiple connections, or to further
   // complicate the class dependencies, we have each connection indicate
   // a multiplier to be applied before being added into the post layer's
   // GSyn. For most connections, this multiplier will be one. However,
   // for a connection whose pre- and post- are both broadcast layers,
   // the multiplier is 1/(Nrows*Ncols), where Nrows and Ncols are
   // the numbers of rows and columns of the MPI configuration.
   // The ReductionMultiplier is defined in LayerInputDelivery, where
   // it is accessible by the layer input buffer component. It is
   // set, however, by the derived classes once the broadcast-ness
   // of the pre- and post- layers are determined.
   float mReductionMultiplier = 1.0f;
};

} // namespace PV

#endif // LAYERINPUTDELIVERY_HPP_
