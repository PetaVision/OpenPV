/*
 * BaseDelivery.hpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#ifndef BASEDELIVERY_HPP_
#define BASEDELIVERY_HPP_

#include "columns/BaseObject.hpp"
#include "columns/Publisher.hpp"
#include "components/Weights.hpp"
#include "layers/HyPerLayer.hpp"

namespace PV {

/**
 * The base class for the delivery component of connections.
 * Derived classes should override the deliver method.
 *
 * After instantiation but before any calls to deliver(), the initializeDelays() method
 * should be called with the number of arbors.
 */
class BaseDelivery : public BaseObject {
  protected:
   /**
    * List of parameters needed from the BaseDelivery class
    * @name BaseDelivery Parameters
    * @{
    */

   /**
    * @brief channelCode: Specifies which channel in the post layer to deliver to.
    * @details Channels can be -1 for no update, or >= 0 for channel number. <br />
    * 0 is excitatory, 1 is inhibitory
    */
   virtual void ioParam_channelCode(enum ParamsIOFlag ioFlag);

   /**
    * @brief delay: Specifies delay(s) over which the post layer will receive data.
    * The delays are specified in the same units that the HyPerCol dt parameter is specified in.
    * The default is a single delay of zero. The array must either have a single entry, or have
    * a length equal to the connection's number of arbors.
    */
   virtual void ioParam_delay(enum ParamsIOFlag ioFlag);

   /**
    * @brief receiveGpu: If PetaVision was compiled with GPU acceleration and this flag is set to
    * true, the connection uses the GPU to update the postsynaptic layer's GSyn.
    * If compiled without GPU acceleration, it is an error to set this flag to true.
    */
   virtual void ioParam_receiveGpu(enum ParamsIOFlag ioFlag);
   /** @} */ // End of list of BaseDelivery parameters.

  public:
   // No public constructors; only derived classes can be instantiated.

   virtual ~BaseDelivery();

   /**
    * Sets the number of arbors. Must be called before allocateDataStructures, since some data
    * structures' sizes may depend on the number of arbors. It is an error to set the number of
    * arbors to a nonpositive value, or to call setNumArbors if NumArbors was already set to a
    * positive value. Since NumArbors is initially zero, setNumArbors can only be called once.
    */
   void setNumArbors(int numArbors);

   /**
    * Sets the pre- and post-synaptic layers. Must be called before allocateDataStructures, since
    * some data structures may need information from these layers.
    */
   void setPreAndPostLayers(HyPerLayer *preLayer, HyPerLayer *postLayer);

   /**
    * @brief The method that delivers the activity on the presynaptic layer to the postsynaptic
    * channel. weights is the object containing the synaptic strengths.
    * @details BaseDeliver is agnostic about whether the weights are organized from the
    * presynaptic or postsynaptic perspective. It is up to the calling function to make sure
    * that the particular derived class being called is consistent with the organization of the
    * weights argument.
    *
    * (HyPerConn accomplishes this with the updateGSynFromPostsynapticPerspective flag; if
    * the flag is true, the DeliveryObject is a class that uses the PostsynapticPerspective,
    * and it passes mPostWeights to the delivery object; if the flag is false, it creates a
    * subclass that uses the PresynapticPerspective, and passes mWeights to the delivery object.
    */
   virtual void deliver(Weights *weights);

   /**
    * @brief Similar to deliver(), except it does not use the presynaptic layer; instead
    * takes the presynaptic input to be a constant value of one. recvBuffer is a buffer
    * of size of the post layer's getNumNeuronsAllBatches().
    * If there is more than one arbor, each arbor is applied.
    */
   virtual void deliverUnitInput(Weights *weights, float *recvBuffer);

   /**
    * Returns the channel ID that the deliver method acts on
    */
   inline ChannelType getChannelCode() const { return mChannelCode; }

   /**
    * Returns the value of the receiveGpu parameter
    */
   bool getReceiveGpu() const { return mReceiveGpu; }

  protected:
   BaseDelivery();

   int initialize(char const *name, HyPerCol *hc);

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual int allocateDataStructures() override;

   /**
    * Called during allocateDataStructures. Converts the delays array, as read from the params file.
    * @details: Delays are specified in units of dt, but are rounded to be integer multiples of dt.
    * If delay is a scalar, all arbors of the connection have that value of delay.
    * If delay is an array, the length must match the number of arbors and the arbors are assigned
    * to the delays sequentially.
    */
   void initializeDelays();

   /**
    * Called by initializeDelays. Calculates the number of timesteps corresponding
    * to the given delay time, where timesteps have length deltaTime.
    * If delay is not exactly an integer multiple of zero, issues a warning and
    * returns round(delay/deltaTime).
    */
   int convertToNumberOfTimesteps(double delay, double deltaTime);

   /**
    * Type-safe method of translating an integer channel_code into
    * an allowed channel type.  If channel_code corresponds to a
    * recognized channel type, *channel_type is set accordingly and the
    * function returns successfully.  Otherwise, *channel_type is undefined
    * and the function returns PV_FAILURE.
    */
   static int decodeChannel(int channelCode, ChannelType *channelType) {
      int status = PV_SUCCESS;
      switch (channelCode) {
         case CHANNEL_EXC: *channelType      = CHANNEL_EXC; break;
         case CHANNEL_INH: *channelType      = CHANNEL_INH; break;
         case CHANNEL_INHB: *channelType     = CHANNEL_INHB; break;
         case CHANNEL_GAP: *channelType      = CHANNEL_GAP; break;
         case CHANNEL_NORM: *channelType     = CHANNEL_NORM; break;
         case CHANNEL_NOUPDATE: *channelType = CHANNEL_NOUPDATE; break;
         default: status                     = PV_FAILURE; break;
      }
      return status;
   }

   // Data members
  protected:
   ChannelType mChannelCode;

   std::vector<double> mDelayFromParams; // The array of delays, as read from the params file.
   std::vector<int> mDelay; // The array of delays, as a number of timesteps.
   // mDelayFromParams is in the same units as the HyPerCol's dt.
   // The size of mDelayFromParams is the number of entries in the delay array in the params file.
   // The size of mDelay is the number of arbors in the connection.
   // If mDelayFromParams.size() is zero (param was not specified), all delays in mDelay are zero.
   // If mDelayFromParams.size() is one, all delays in mDelay are round(delay/dt).
   // If mDelayFromParams.size() is the number of arbors, mDelay[a] = round(mDelayFromParams[a]/dt).
   // Any other size for mDelayFromParams is an error.

   bool mReceiveGpu = false;

   int mNumArbors         = 0;
   HyPerLayer *mPreLayer  = nullptr;
   HyPerLayer *mPostLayer = nullptr;
}; // end class BaseDelivery

} // end namespace PV

#endif // BASEDELIVERY_HPP_
