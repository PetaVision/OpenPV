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

#include <columns/BaseObject.hpp>
#include <columns/HyPerCol.hpp>
#include <io/io.h>
#include <io/PVParams.hpp>

namespace PV {

class HyPerLayer;
class BaseConnectionProbe;

class BaseConnection : public BaseObject {

public:
   /**
    * Destructor for BaseConnection
    */
   virtual ~BaseConnection();

   /**
    * Method for reading or writing the params from group in the parent HyPerCol's parameters.
    * The group from params is selected using the name of the connection.
    *
    * Note that ioParams is not virtual.  To add parameters in a derived class, override ioParamFillGroup.
    */
   int ioParams(enum ParamsIOFlag ioFlag);

   // manage the communicateInitInfo, allocateDataStructures, and initializeState stages.
   /**
    * communicateInitInfo is used to allow connections and layers to set params and related member variables based on what other
    * layers or connections are doing.  (For example, CloneConn sets many parameters the same as its originalConn.)
    * After a connection is constructed, it is not properly initialized until communicateInitInfo(), allocateDataStructures(), and
    * initializeState() have been called.
    *
    * Return values:
    *    PV_POSTPONE means that communicateInitInfo() cannot be run until other layers'/connections' own communicateInitInfo()
    *    have been run successfully.
    *
    *    PV_SUCCESS and PV_FAILURE have their usual meanings.
    *
    * communicateInitInfo() is typically called by the parent HyPerCol's run() method.
    */
   virtual int communicateInitInfo();

   /**
    * This method sets the flag returned by getInitInfoCommunicatedFlag() to true.
    * It is public so that the parent HyPerCol's run method can set it after receiving a successful communicateInitInfo command
    * (this behavior should probably be changed so that BaseConnection::communicateInitInfoWrapper, not the calling routine,
    * is responsible for setting the flag).
    */
   void setInitInfoCommunicatedFlag() { initInfoCommunicatedFlag = true; }// should be called only by HyPerCol::run

   /**
    * Returns true or false, depending on whether communicateInitInfo() has been called successfully.
    */
   bool getInitInfoCommunicatedFlag() { return initInfoCommunicatedFlag; }

   /**
    * allocateDataStructures is used to allocate blocks of memory whose size and arrangement depend on parameters.
    * (For example, HyPerConn allocates weight patches and data patches).
    * After a connection is constructed, it is not properly initialized until communicateInitInfo(), allocateDataStructures(), and
    * initializeState() have been called.
    *
    * Return values:
    *    PV_POSTPONE means that allocateDataStructures() cannot be run until other layers'/connections' own allocateDataStructures()
    *    have been run successfully.
    *
    *    PV_SUCCESS and PV_FAILURE have their usual meanings.
    *
    * allocateDataStructures() is typically called by the parent HyPerCol's run() method.
    */
   virtual int allocateDataStructures();// should be called only by HyPerCol::run

   /**
    * This method sets the flag returned by getDataStructuresAllocatedFlag() to true.
    * It is public so that the parent HyPerCol's run method can set it after receiving a successful allocateDataStructures command
    * (this behavior should probably be changed so that BaseConnection::allocateDataStructuresWrapper, not the calling routine,
    * is responsible for setting the flag).
    */
   void setDataStructuresAllocatedFlag() { dataStructuresAllocatedFlag = true; }// should be called only by HyPerCol::run

   /**
    * Returns true or false, depending on whether communicateInitInfo() has been called successfully.
    */
   bool getDataStructuresAllocatedFlag() { return dataStructuresAllocatedFlag; }

   /**
    * initializeState is used to set the initial values of the connection.
    * If the parent HyPerCol's checkpointReadFlag is set, it calls checkpointRead()
    * If not, but the connection's initializeFromCheckpointFlag is set, it calls readStateFromCheckpoint().
    * If neither of these flags is set, it calls setInitialValues.
    * Note that derived classes must implement the methods checkpointRead(), readStateFromCheckpoint(), and setInitialValues().
    *
    * After a connection is constructed, it is not properly initialized until communicateInitInfo(), allocateDataStructures(), and
    * initializeState() have been called.
    *
    * Return values:
    *    PV_POSTPONE means that initializeState() cannot be run until other layers'/connections' own initializeState()
    *    have been run successfully.
    *
    *    PV_SUCCESS and PV_FAILURE have their usual meanings.
    *
    * initializeState() is typically called by the parent HyPerCol's run() method.
    */
   int initializeState();

   /**
    * This method sets the flag returned by getInitialValuesSetFlag() to true.
    * It is public so that the parent HyPerCol's run method can set it after receiving a successful initializeState command
    * (this behavior should probably be changed so that BaseConnection::allocateDataStructuresWrapper, not the calling routine,
    * is responsible for setting the flag).
    */
   void setInitialValuesSetFlag() {initialValuesSetFlag = true;}// should be called only by HyPerCol::run

   /**
    * Returns true or false, depending on whether initializeState() has been called successfully.
    */
   bool getInitialValuesSetFlag() {return initialValuesSetFlag;}

   /**
    * A pure virtual function for writing the state of the connection to file(s) in the output directory.
    * For example, HyPerConn writes the weights to a .pvp file with a schedule defined by
    * writeStep and initialWriteTime.
    */
   virtual int outputState(double timed, bool last = false) = 0;

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
    * A pure virtual function for modifying the post-synaptic layer's GSyn buffer based on the connection
    * and the presynaptic activity
    */
   virtual int deliver() = 0;

   /**
    * A pure virtual function for reading the state of the connection from the directory specified in cpDir.
    * On exit, *timeptr is the time at which the checkpoint was written.
    * checkpointRead() should restore the state of the connection completely, so that restarting from a checkpoint
    * is equivalent to having the run continue.
    *
    */
   virtual int checkpointRead(const char * cpDir, double * timeptr) = 0;

   /**
    * A pure virtual function for writing the state of the connection to the directory specified in cpDir.
    * checkpointWrite() should save the complete state of the connection, so that restarting from a checkpoint
    * is equivalent to having the run continue.
    */
   virtual int checkpointWrite(const char * cpDir) = 0;

   /**
    * A pure virtual function for writing timing information.
    */
   virtual int writeTimers(FILE * stream) = 0;

   /**
    * Called by HyPerCol::outputParams to output the params groups for probes whose ownership has
    * been transferred to this connection. (Does this need to be virtual?)
    */
   virtual int outputProbeParams();

   /**
    * Adds the given probe to the list of probes.
    */
   virtual int insertProbe(BaseConnectionProbe* p);

   /**
    * Returns the connection's connId (assigned when added to its parent HyPerCol)
    */
   inline int getConnectionId() { return connId; }

   /*
    * Returns the name of the connection's presynaptic layer.
    */
   inline const char * getPreLayerName() {return preLayerName;}

   /*
    * Returns the name of the connection's postsynaptic layer.
    */
   inline const char * getPostLayerName() {return postLayerName;}

   /*
    * Returns a pointer to the connection's presynaptic layer.
    */
   inline HyPerLayer * preSynapticLayer() { return pre; }

   /*
    * Returns a pointer to the connection's postsynaptic layer.
    */
   inline HyPerLayer * postSynapticLayer() { return post; }

   /*
    * Returns a pointer to the connection's presynaptic layer.
    */
   inline HyPerLayer * getPre() { return pre; }

   /*
    * Returns a pointer to the connection's postsynaptic layer.
    */
   inline HyPerLayer * getPost() { return post; }

   /*
    * Returns a pointer to the channel of the postsynaptic layer that the channel acts on.
    */
   inline ChannelType getChannel() { return channel; }

   inline int numberOfAxonalArborLists() {
      return numAxonalArborLists;
   }

   inline bool getPlasticityFlag() {
      return plasticityFlag;
   };

   /**
    * Returns the delay (in timesteps) belonging the given arbor.
    */
   inline int getDelay(int arbor) { return (arbor >= 0 && arbor < this->numberOfAxonalArborLists()) ? delays[arbor] : -1; }

   inline bool getConvertRateToSpikeCount() { return convertRateToSpikeCount; }
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   inline bool getReceiveGpu() { return receiveGpu; }
#endif // defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)

   /**
    * Returns the number of probes that have been attached to this connection
    */
   int getNumProbes() { return numProbes; }

   /**
    * Returns the probe with the indicated position in the list of probes.
    * It does not do sanity checking on the value of i.
    */
   BaseConnectionProbe * getProbe(int i) { return probes[i]; }

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
   int initialize(const char * name, HyPerCol * hc);

   /**
    * Sets the pre- and post-synaptic layer names according to the parent HyPerCol's params.
    * Virtual to allow subclasses to infer layer names in other ways (for example, FeedbackConn
    * flips pre- and post- layers from originalConn).
    */
   virtual int setPreAndPostLayerNames();

   /**
    * Sets the presynaptic layer name to the given string.  It is an error to try to set preLayerName
    * after it has already been set, or to call setPreLayerName() with a NULL argument.
    */
   void setPreLayerName(const char * preName);

   /**
    * Sets the postsynaptic layer name to the given string.  It is an error to try to set postLayerName
    * after it has already been set, or to call setPostLayerName() with a NULL argument.
    */
   void setPostLayerName(const char * postName);

   /**
    * Sets the presynaptic layer to the given layer.  It is an error to try to set preLayer
    * after it has already been set, or to call setPreLayerName() with a NULL argument.
    */
   void setPreSynapticLayer(HyPerLayer * pre);

   /**
    * Sets the postsynaptic layer to the given layer.  It is an error to try to set postLayer
    * after it has already been set, or to call setPostLayerName() with a NULL argument.
    */
   void setPostSynapticLayer(HyPerLayer * post);

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
   void setDelay(int arborId, float delay);

   /**
    * Sets the number of arbors to the indicated argument.  It is an error to try to change numArbors
    * after communicateInitInfo() has completed successfully.
    */
   void setNumberOfAxonalArborLists(int numArbors);

   // preActivityIsNotRate was replaced with convertRateToSpikeCount on Dec 31, 2014.
   // /**
   //  * Sets the preActivityIsNotRate flag to the indicated argument.  It is an error to try to change
   //  * preActivityIsNotRate after communicateInitInfo() has completed successfully.
   //  */
   // void setPreActivityIsNotRate(bool preActivityIsNotRate);
   void setConvertRateToSpikeCount(bool convertRateToSpikeCountFlag);
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   void setReceiveGpu();
#endif // defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)

   /**
    * Called by BaseConnection::communicateInitInfo of the params did not set pre- and post- layers.
    */
   virtual int handleMissingPreAndPostLayerNames();

   /**
    * The default behavior of BaseConnection::handleMissingPreAndPostLayerNames.
    * It tries to parse the name argument of the connection in the form "PreLayerToPostLayer".
    * Then "PreLayer" put into *preLayerNamePtr and "PostLayer" is put into *postLayerNamePtr, and PV_SUCCESS is returned.
    * If name does not contain the string "To", or if it contains it in more than one place, then PV_FAILURE is returned
    * and *preLayerNamePtr and *postLayerNamePtr are not changed.
    * rank is the rank of the process under MPI; the root process will print a message to stderr if the routine fails; non-root process will not.
    * This routine uses malloc to fill *{pre,post}LayerNamePtr, so the routine calling this one is responsible for freeing them.
    */
   static int inferPreAndPostFromConnName(const char * name, int rank, char ** preLayerNamePtr, char ** postLayerNamePtr);


   /**
    * Sets *preLayerNamePtr and *postLayerNamePtr according to the preLayerName and postLayerName parameters in
    * the parameter group specified by the name and params arguments.
    */
   int getPreAndPostLayerNames(const char * name, char ** preLayerNamePtr, char ** postLayerNamePtr);

   /**
    * The virtual method for reading parameters from the parent HyPerCol's parameters, and writing to the output params file.
    *
    * BaseConnection::ioParamsFillGroup reads/writes the paremeters
    * preLayerName, postLayerName, channelCode, delay, numAxonalArbors, and convertRateToSpikeCount.
    *
    * Derived classes with additional parameters typically override ioParamsFillGroup to call the base class's ioParamsFillGroup
    * method and then call ioParam_[parametername] for each of their parameters.
    * The ioParam_[parametername] methods should call the parent HyPerCol's ioParamValue() and related methods,
    * to ensure that all parameters that get read also get written to the outputParams-generated file.
    */
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

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

   // preActivityIsNotRate was replaced with convertRateToSpikeCount on Dec 31, 2014.
   // /**
   //  * @brief preActivityIsNotRate: If true, pre activity is spike rate. If false, pre activity is value
   //  * @details The post synaptic layer needs to interpret pre synaptic activity as a spike rate
   //  * Other situations interpret as a value. This flag sets either one or the other.
   //  */
   // virtual void ioParam_preActivityIsNotRate(enum ParamsIOFlag ioFlag);
   /**
    * @brief convertRateToSpikeCount: If true, presynaptic activity should be converted from a rate to a count.
    * @details If this flag is true and the presynaptic layer is not spiking, the activity will be interpreted
    * as a spike rate, and will be converted to a spike count when delivering activity to the postsynaptic GSyn buffer.
    * If this flag is false, activity will not be converted.
    */
   virtual void ioParam_convertRateToSpikeCount(enum ParamsIOFlag ioFlag);

   /**
    * @brief receiveGpu: If PetaVision was compiled with GPU acceleration and this flag is set to true, the connection uses the GPU to update the postsynaptic layer's GSyn.
    * If compiled without GPU acceleration, it is an error to set this flag to true.
    */
   virtual void ioParam_receiveGpu(enum ParamsIOFlag ioFlag);

   /**
    * @brief initializeFromCheckpointFlag: If set to true, initialize using checkpoint direcgtory set in HyPerCol.
    * @details Checkpoint read directory must be set in HyPerCol to initialize from checkpoint.
    */
   virtual void ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag);

   /**
    * A pure virtual method that uses an existing checkpoint to
    * initialize the connection.  BaseConnection::initializeState calls it
    * when initializeFromCheckpointFlag is true.  A Subclass may also
    * call this method as part of the implementation of checkpointRead
    * (for example, HyPerConn does this).
    */
   virtual int readStateFromCheckpoint(const char * cpDir, double * timeptr) = 0;

   /**
    * A pure virtual method for initializing the connection if we are neither
    * restarting from a checkpoint or initializing the connection from a checkpoint.
    * It should return PV_SUCCESS if successful, or PV_POSTPONE if it needs to wait for
    * other objects to set their initial values before it can set its own initial values.
    * (e.g. TransposeConn has to wait for original conn)
    */
   virtual int setInitialValues() = 0;

   /**
    * A pure virtual method whose implementation returns true
    * if an the weights should update on the given timestep
    * and false if not.
    */
   virtual bool needUpdate(double time, double dt) = 0;

   /**
    * Allocates the delays array and calls setDelay() for each arbor.  fDelayArray is an array of length size,
    * of delays, measured in the same units as the parent HyPerCol's dt.
    *
    * If size=0, all delays are set to zero.
    * If size=1, all delays are set to fDelayArray[0]
    * If size=numArbors, delays[k] is calculated from fDelayArray[k].
    * If size is any other value, it is an error.
    */
   virtual int initializeDelays(const float * fDelayArray, int size);
   
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
   static int decodeChannel(int channel_code, ChannelType * channel_type) {
      int status = PV_SUCCESS;
      switch( channel_code ) {
      case CHANNEL_EXC:
         *channel_type = CHANNEL_EXC;
         break;
      case CHANNEL_INH:
         *channel_type = CHANNEL_INH;
         break;
      case CHANNEL_INHB:
         *channel_type = CHANNEL_INHB;
         break;
      case CHANNEL_GAP:
         *channel_type = CHANNEL_GAP;
         break;
      case CHANNEL_NORM:
         *channel_type = CHANNEL_NORM;
         break;
      case CHANNEL_NOUPDATE:
         *channel_type = CHANNEL_NOUPDATE;
         break;
      default:
         status = PV_FAILURE;
         break;
      }
      return status;
   }

   int getDelayArraySize(){return delayArraySize;}

// member variables
protected:
   int connId;
   char * preLayerName;
   char * postLayerName;
   HyPerLayer * pre;
   HyPerLayer * post;
   ChannelType channel;
   int numAxonalArborLists; // number of axonal arbors from presynaptic layer
   bool plasticityFlag;
   // bool preActivityIsNotRate; // preActivityIsNotRate was replaced with convertRateToSpikeCount on Dec 31, 2014.
   bool convertRateToSpikeCount; // Whether to check if pre-layer is spiking and, if it is not, scale activity by dt to convert it to a spike count
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   bool receiveGpu; // Whether to use GPU acceleration in updating post's GSyn
#endif // defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   bool initializeFromCheckpointFlag;

   BaseConnectionProbe** probes; // probes used to output data
   int numProbes;

   bool initInfoCommunicatedFlag;
   bool dataStructuresAllocatedFlag;
   bool initialValuesSetFlag;

private:
   int delayArraySize;
   int* delays; // delays[arborId] is the delay in timesteps (not units of dt) of the arborId'th arbor
   float * fDelayArray; // delays[arborId] is the delay in units of dt of the arborId'th arbor
}; // end class BaseConnection

}  // end namespace PV
#endif // BASECONNECTION_HPP_
