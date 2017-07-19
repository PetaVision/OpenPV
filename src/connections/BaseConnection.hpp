/*
 * BaseConnection.hpp
 *
 *  Created on Sep 19, 2014
 *      Author: Pete Schultz
 *
 *  The abstract base class for the connection hierarchy.
 *  Only derived classes can be instantiated.
 *  The purpose is so that there can be a pointer to a conn without having
 *  to specify the specialization in the pointer declaration.
 */

#ifndef BASECONNECTION_HPP_
#define BASECONNECTION_HPP_

#include "columns/BaseObject.hpp"
#include "columns/HyPerCol.hpp"
#include "io/PVParams.hpp"
#include "io/PrintStream.hpp"
#include "io/io.hpp"
#include "layers/HyPerLayer.hpp"

namespace PV {

class BaseConnectionProbe;

class BaseConnection : public BaseObject {

  public:
   /**
    * Destructor for BaseConnection
    */
   virtual ~BaseConnection();

   virtual int respond(std::shared_ptr<BaseMessage const> message) override;

   // manage the communicateInitInfo, allocateDataStructures, and initializeState stages.
   /**
    * communicateInitInfo is used to allow connections and layers to set params and related member
    * variables based on what other
    * layers or connections are doing.  (For example, CloneConn sets many parameters the same as its
    * originalConn.)
    * After a connection is constructed, it is not properly initialized until communicateInitInfo(),
    * allocateDataStructures(), and
    * initializeState() have been called.
    *
    * Return values:
    *    PV_POSTPONE means that communicateInitInfo() cannot be run until other layers'/connections'
    * own communicateInitInfo()
    *    have been run successfully.
    *
    *    PV_SUCCESS and PV_FAILURE have their usual meanings.
    *
    * communicateInitInfo() is called by passing a CommunicateInitInfoMessage to respond(), which is
    * usually done in HyPerCol::run.
    */
   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   /**
    * initializeState is used to set the initial values of the connection.
    *
    * initializeState() is typically called by passing an InitializeStateMessage to respond(), which
    * is usually done in HyPerCol::run.
    */
   virtual int initializeState() override final;
   // Not overridable. BaseConnection::initializeState() calls setInitialValues(), which is virtual.

   /**
    * A pure virtual function for writing the state of the connection to file(s) in the output
    * directory.
    * For example, HyPerConn writes the weights to a .pvp file with a schedule defined by
    * writeStep and initialWriteTime.
    */
   virtual int outputState(double timed) = 0;

   /**
    * A pure virtual function for updating the state of the connection.
    * timed is simulation time, and dt is the time increment between steps.
    */
   virtual int updateState(double timed, double dt) = 0;

   /**
    * A virtual function for performing any necessary updates after the normalizers are called.
    */
   virtual int finalizeUpdate(double timed, double dt) { return PV_SUCCESS; }

   /**
    * A pure virtual function for modifying the post-synaptic layer's GSyn buffer based on the
    * connection
    * and the presynaptic activity
    */
   virtual int deliver() = 0;

   /**
    * Called by HyPerCol::outputParams to output the params groups for probes whose ownership has
    * been transferred to this connection. (Does this need to be virtual?)
    */
   virtual int outputProbeParams();

   /**
    * Adds the given probe to the list of probes.
    */
   virtual int insertProbe(BaseConnectionProbe *p);

   /*
    * Returns the name of the connection's presynaptic layer.
    */
   inline const char *getPreLayerName() { return preLayerName; }

   /*
    * Returns the name of the connection's postsynaptic layer.
    */
   inline const char *getPostLayerName() { return postLayerName; }

   /*
    * Returns a pointer to the connection's presynaptic layer.
    */
   inline HyPerLayer *preSynapticLayer() { return pre; }

   /*
    * Returns a pointer to the connection's postsynaptic layer.
    */
   inline HyPerLayer *postSynapticLayer() { return post; }

   /*
    * Returns a pointer to the connection's presynaptic layer.
    */
   inline HyPerLayer *getPre() { return pre; }

   /*
    * Returns a pointer to the connection's postsynaptic layer.
    */
   inline HyPerLayer *getPost() { return post; }

   /*
    * Returns a pointer to the channel of the postsynaptic layer that the channel acts on.
    */
   inline ChannelType getChannel() { return channel; }

   inline int numberOfAxonalArborLists() { return numAxonalArborLists; }

   inline bool getPlasticityFlag() { return plasticityFlag; };

   /**
    * Returns the delay (in timesteps) belonging the given arbor.
    */
   inline int getDelay(int arbor) {
      return (arbor >= 0 && arbor < this->numberOfAxonalArborLists()) ? delays[arbor] : -1;
   }

   inline bool getConvertRateToSpikeCount() { return convertRateToSpikeCount; }
   inline bool getReceiveGpu() { return receiveGpu; }

   /**
    * Returns the number of probes that have been attached to this connection
    */
   int getNumProbes() { return numProbes; }

   /**
    * Returns the probe with the indicated position in the list of probes.
    * It does not do sanity checking on the value of i.
    */
   BaseConnectionProbe *getProbe(int i) { return probes[i]; }

  protected:
   /**
    * The constructor implicitly called by derived classes' constructors.
    * It calls initialize_base(), but not initialize().
    *
    * Note that BaseConnection has no public constructors; only derived
    * classes can be constructed directly.
    */
   BaseConnection();

   /**
    * The initialization routine.  It should be called during the
    * initialization routine of any derived class.
    *
    * It sets the name and parent HyPerCol to the indicated arguments,
    * and calls (via ioParams) the virtual ioParamsFillGroup method,
    * which reads params from the parent HyPerCol's params.
    */
   int initialize(const char *name, HyPerCol *hc);

   /**
    * Sets the pre- and post-synaptic layer names according to the parent HyPerCol's params.
    * Virtual to allow subclasses to infer layer names in other ways (for example, FeedbackConn
    * flips pre- and post- layers from originalConn).
    */
   virtual int setPreAndPostLayerNames();

   /**
    * Sets the presynaptic layer name to the given string.  It is an error to try to set
    * preLayerName
    * after it has already been set, or to call setPreLayerName() with a NULL argument.
    */
   void setPreLayerName(const char *preName);

   /**
    * Sets the postsynaptic layer name to the given string.  It is an error to try to set
    * postLayerName
    * after it has already been set, or to call setPostLayerName() with a NULL argument.
    */
   void setPostLayerName(const char *postName);

   /**
    * Sets the channel to the indicated argument.  It is an error to try to change channels
    * after communicateInitInfo() has completed successfully.
    */
   void setChannelType(ChannelType ch);

   /**
    * Sets the delay of the given arbor to the given amount.  delay is specified in the same units
    * that the parent HyPerCol's dt parameter is specified in.  Internally, the delay is set as
    * an integral number of timesteps, specifically round(delay/dt).
    */
   void setDelay(int arborId, double delay);

   /**
    * Sets the number of arbors to the indicated argument.  It is an error to try to change
    * numArbors
    * after communicateInitInfo() has completed successfully.
    */
   void setNumberOfAxonalArborLists(int numArbors);

   void setConvertRateToSpikeCount(bool convertRateToSpikeCountFlag);
#ifdef PV_USE_CUDA
   void setReceiveGpu();
#endif // PV_USE_CUDA

   /**
    * Called by BaseConnection::communicateInitInfo of the params did not set pre- and post- layers.
    */
   virtual int handleMissingPreAndPostLayerNames();

   /**
    * The default behavior of BaseConnection::handleMissingPreAndPostLayerNames.
    * It tries to parse the name argument of the connection in the form "PreLayerToPostLayer".
    * Then "PreLayer" put into *preLayerNamePtr and "PostLayer" is put into *postLayerNamePtr, and
    * PV_SUCCESS is returned.
    * If name does not contain the string "To", or if it contains it in more than one place, then
    * PV_FAILURE is returned
    * and *preLayerNamePtr and *postLayerNamePtr are not changed.
    * rank is the rank of the process under MPI; the root process will print a message to the error
    * stream if the routine fails; non-root process will not.
    * This routine uses malloc to fill *{pre,post}LayerNamePtr, so the routine calling this one is
    * responsible for freeing them.
    */
   static int inferPreAndPostFromConnName(
         const char *name,
         int rank,
         char **preLayerNamePtr,
         char **postLayerNamePtr);

   /**
    * Sets *preLayerNamePtr and *postLayerNamePtr according to the preLayerName and postLayerName
    * parameters in
    * the parameter group specified by the name and params arguments.
    */
   int getPreAndPostLayerNames(const char *name, char **preLayerNamePtr, char **postLayerNamePtr);

   /**
    * BaseConnection::ioParamsFillGroup reads/writes the parameters
    * preLayerName, postLayerName, channelCode, delay, numAxonalArbors, and convertRateToSpikeCount.
    *
    */
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   /**
    * List of parameters needed from the BaseConnection class
    * @name BaseConnection Parameters
    * @{
    */

   /**
    * @brief preLayerName: Specifies the connection's pre layer
    * @details Required parameter
    */
   virtual void ioParam_preLayerName(enum ParamsIOFlag ioFlag);

   /**
    * @brief postLayerName: Specifies the connection's post layer
    * @details Required parameter
    */
   virtual void ioParam_postLayerName(enum ParamsIOFlag ioFlag);

   /**
    * @brief channelCode: Specifies which channel in the post layer this connection is attached to
    * @details Channels can be -1 for no update, or >= 0 for channel number. <br />
    * 0 is excitatory, 1 is inhibitory
    */
   virtual void ioParam_channelCode(enum ParamsIOFlag ioFlag);

   /**
    * @brief delay: Specifies delay(s) which the post layer will receive data
    * @details: Delays are specified in units of dt, but are rounded to be integer multiples of dt.
    * If delay is a scalar, all arbors of the connection have that value of delay.
    * If delay is an array, the length must match the number of arbors and the arbors are assigned
    * the delays sequentially.
    */
   virtual void ioParam_delay(enum ParamsIOFlag ioFlag);

   /**
    * @brief numAxonalArbors: Specifies the number of arbors to use in this connection
    */
   virtual void ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag);

   /**
    * @brief plasticityFlag: Specifies if the weights will update
    */
   virtual void ioParam_plasticityFlag(enum ParamsIOFlag ioFlag);

   /**
    * @brief convertRateToSpikeCount: If true, presynaptic activity should be converted from a rate
    * to a count.
    * @details If this flag is true and the presynaptic layer is not spiking, the activity will be
    * interpreted
    * as a spike rate, and will be converted to a spike count when delivering activity to the
    * postsynaptic GSyn buffer.
    * If this flag is false, activity will not be converted.
    */
   virtual void ioParam_convertRateToSpikeCount(enum ParamsIOFlag ioFlag);

   /**
    * @brief receiveGpu: If PetaVision was compiled with GPU acceleration and this flag is set to
    * true, the connection uses the GPU to update the postsynaptic layer's GSyn.
    * If compiled without GPU acceleration, it is an error to set this flag to true.
    */
   virtual void ioParam_receiveGpu(enum ParamsIOFlag ioFlag);

   /**
    * @brief initializeFromCheckpointFlag: If set to true, initialize using checkpoint direcgtory
    * set in HyPerCol.
    * @details Checkpoint read directory must be set in HyPerCol to initialize from checkpoint.
    */
   virtual void ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag);

   /**
    * A pure virtual method for initializing the connection if we are neither
    * restarting from a checkpoint or initializing the connection from a checkpoint.
    * It should return PV_SUCCESS if successful, or PV_POSTPONE if it needs to wait for
    * other objects to set their initial values before it can set its own initial values.
    * (e.g. TransposeConn has to wait for original conn)
    */
   virtual int setInitialValues() = 0;

   int respondConnectionWriteParams(std::shared_ptr<ConnectionWriteParamsMessage const> message);

   int respondConnectionProbeWriteParams(
         std::shared_ptr<ConnectionProbeWriteParamsMessage const> message);

   int respondLayerProbeWriteParams(std::shared_ptr<LayerWriteParamsMessage const> message);

   int respondConnectionUpdate(std::shared_ptr<ConnectionUpdateMessage const> message) {
      return updateState(message->mTime, message->mDeltaT);
   }

   int
   respondConnectionFinalizeUpdate(std::shared_ptr<ConnectionFinalizeUpdateMessage const> message) {
      return finalizeUpdate(message->mTime, message->mDeltaT);
   }

   int respondConnectionOutput(std::shared_ptr<ConnectionOutputMessage const> message) {
      return outputState(message->mTime);
   }

   /**
    * A pure virtual method whose implementation returns true
    * if an the weights should update on the given timestep
    * and false if not.
    */
   virtual bool needUpdate(double time, double dt) = 0;

   /**
    * Allocates the delays array and calls setDelay() for each arbor.  fDelayArray is an array of
    * length size,
    * of delays, measured in the same units as the parent HyPerCol's dt.
    *
    * If size=0, all delays are set to zero.
    * If size=1, all delays are set to fDelayArray[0]
    * If size=numArbors, delays[k] is calculated from fDelayArray[k].
    * If size is any other value, it is an error.
    */
   virtual int initializeDelays(const float *fDelayArray, int size);

   /**
    * Returns the maximum value of the delay array, as a number of timesteps
    */
   int maxDelaySteps();

  private:
   /**
    * Called by the constructor, initialize_base() sets member variables
    * to safe values (e.g. pointers to NULL) and parameters to default values.
    */
   int initialize_base();

   // static methods
  public:
   /**
    * Type-safe method of translating an integer channel_code into
    * an allowed channel type.  If channel_code corresponds to a
    * recognized channel type, *channel_type is set accordingly and the
    * function returns successfully.  Otherwise, *channel_type is undefined
    * and the function returns PV_FAILURE.
    */
   static int decodeChannel(int channel_code, ChannelType *channel_type) {
      int status = PV_SUCCESS;
      switch (channel_code) {
         case CHANNEL_EXC: *channel_type      = CHANNEL_EXC; break;
         case CHANNEL_INH: *channel_type      = CHANNEL_INH; break;
         case CHANNEL_INHB: *channel_type     = CHANNEL_INHB; break;
         case CHANNEL_GAP: *channel_type      = CHANNEL_GAP; break;
         case CHANNEL_NORM: *channel_type     = CHANNEL_NORM; break;
         case CHANNEL_NOUPDATE: *channel_type = CHANNEL_NOUPDATE; break;
         default: status                      = PV_FAILURE; break;
      }
      return status;
   }

   int getDelayArraySize() { return delayArraySize; }

   // member variables
  protected:
   char *preLayerName;
   char *postLayerName;
   HyPerLayer *pre;
   HyPerLayer *post;
   ChannelType channel;
   int numAxonalArborLists; // number of axonal arbors from presynaptic layer
   bool plasticityFlag;
   bool convertRateToSpikeCount; // Whether to check if pre-layer is spiking and, if it is not,
   // scale activity by dt to convert it to a spike count
   bool receiveGpu; // Whether to use GPU acceleration in updating post's GSyn

   // If this flag is set and HyPerCol sets initializeFromCheckpointDir, load initiali state
   // from the initializeFromCheckpointDir directory.
   bool initializeFromCheckpointFlag = true;

   BaseConnectionProbe **probes; // probes used to output data
   int numProbes;

   bool initInfoCommunicatedFlag;
   bool dataStructuresAllocatedFlag;
   bool initialValuesSetFlag;

  private:
   int delayArraySize;
   int *delays; // delays[arborId] is the delay in timesteps (not units of dt) of the arborId'th
   // arbor
   float *fDelayArray; // delays[arborId] is the delay in units of dt of the arborId'th arbor
}; // end class BaseConnection

} // end namespace PV
#endif // BASECONNECTION_HPP_
