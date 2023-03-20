/*
 * BaseProbe.hpp
 *
 *      Author: slundquist
 */

#ifndef BASEPROBE_HPP_
#define BASEPROBE_HPP_

#include "cMakeHeader.h"

#include "checkpointing/Checkpointer.hpp"
#include "checkpointing/CheckpointingMessages.hpp"
#include "columns/BaseObject.hpp"
#include "columns/Communicator.hpp"
#include "columns/Messages.hpp"
#include "components/LayerUpdateController.hpp"
#include "include/pv_common.h"
#ifdef PV_USE_MPI
#include "io/MPIRecvStream.hpp"
#endif // PV_USE_MPI
#include "io/PVParams.hpp"
#include "io/PrintStream.hpp"
#include "observerpattern/Response.hpp"
#include "utils/Timer.hpp"
#include <memory>
#include <vector>

namespace PV {

// BaseProbe was deprecated on March 19, 2023. Derive probes from ProbeInterface instead,
// or if the desired probe behavior is too different from the interface provided by
// ProbeInterface (e.g. StatsProbe), derive the probe straight from BaseObject.
/**
 * An abstract base class for the common functionality of layer probes and
 * connection probes.
 */
class BaseProbe : public BaseObject {

   // Methods

  protected:
   /**
    * List of parameters for the BaseProbe class
    * @name BaseProbe Parameters
    * @{
    */

   /**
    * @brief targetName: the name of the object that the probe attaches to.
    * In LegacyLayerProbe, targetName is used to define the targetLayer, and in
    * BaseConnectionProbe, targetName is used to define the targetConn.
    */
   virtual void ioParam_targetName(enum ParamsIOFlag ioFlag);

   /**
    * @brief message: A string parameter that is typically included in the lines
    * output by the
    * outputState method
    */
   virtual void ioParam_message(enum ParamsIOFlag ioFlag);

   /**
    * @brief textOutputFlag: A boolean parameter that sets whether to generate an
    * output file.
    * Defaults to true.
    */
   virtual void ioParam_textOutputFlag(enum ParamsIOFlag ioFlag);

   /**
    * @brief probeOutputFile: If textOutputFlag is true, probeOutputFile
    * specifies the name of the file that the outputState method writes to.
    * If blank, the output is sent to the output stream.
    */
   virtual void ioParam_probeOutputFile(enum ParamsIOFlag ioFlag);

   /*
    * @brief statsFlag: Meaningful if textOutputFlag is true.
    * If statsFlag is false, outputState produces one output file for each
    * batch element. If statsFlag is true, outputState produces one output
    * file overall, which reports the min, max and average.
    */
   virtual void ioParam_statsFlag(enum ParamsIOFlag ioFlag);

   /**
    * @brief triggerLayerName: specifies the layer to check for triggering.
    * If triggerLayer is null or empty, the probe does not have a trigger
    * layer and updates every timestep.
    */
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag);

   /**
    * @brief triggerOffset: If triggerLayer is set, triggerOffset specifies the
    * time interval *before* the triggerLayer's nextUpdate time that needUpdate()
    * returns true.
    */
   virtual void ioParam_triggerOffset(enum ParamsIOFlag ioFlag);
   /** @} */
  public:
   virtual ~BaseProbe();

   /**
    * A virtual function called during HyPerCol::run, during the communicateInitInfo stage.
    * BaseProbe::communicateInitInfo sets up the triggering layer and attaches to the energy probe,
    * if either triggerFlag or energyProbe is set.
    */
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   /**
    * Called during HyPerCol::run, during the RegisterData stage.
    * BaseProbe::registerData sets up the output stream.
    * Derived classes that override this method should make sure to
    * call this method in their own registerData methods.
    */
   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   /**
    * Returns the number of value indices the probe can compute.
    * For BaseProbe, this is the parent HyPerCol's getNBatch(); this can be
    * overridden. A probe class can keep the ProbeValues vector empty, in
    * which case getValues() and getNumValues() are not fully implemented
    * for that probe.
    */
   int getNumValues() { return static_cast<int>(mProbeValues.size()); }

   /**
    * The public interface for calling the outputState method.
    * BaseConnection::outputStateWrapper calls outputState() if needUpdate()
    * returns true.
    * This behavior is intended to be general, but the method can be overridden
    * if needed.
    */
   virtual Response::Status outputStateWrapper(double simTime, double dt);
   virtual int writeTimer(PrintStream &stream) { return PV_SUCCESS; }

   /**
    * Returns the name of the targetName parameter for this probe.
    * LegacyLayerProbe uses targetName to specify the layer to attach to;
    * BaseConnectionProbe uses it to specify the connection to attach to.
    */
   const char *getTargetName() { return targetName; }

   /**
    * Returns the time that calcValues was last called.
    * BaseProbe updates the last update time in getValues(), based on the result of needRecalc.
    */
   double getLastUpdateTime() { return lastUpdateTime; }

   /**
    * getValues(double timevalue, double * values) sets the buffer 'values' with
    * the probe's
    * calculated values.
    * It assumes that the values buffer is large enough to hold getNumValues()
    * double-precision values.
    * If 'values' is NULL, the values are still updated internally if needed, but
    * those values are not returned.
    * Internally, getValues() calls calcValues() if needRecalc() is true.  It
    * then
    * copies the mProbeValues vector to the input argument buffer 'values'.
    * Derived classes should not override or hide this method.  Instead, they
    * should override
    * calcValues.
    */
   void getValues(double timevalue, double *valuesVector);
   /**
    * getValues(double timevalue, vector<double> * valuesVector) is a wrapper
    * around
    * getValues(double, double *) that uses C++ vectors.  It resizes valuesVector
    * to size getNumValues() and then fills the vector with the values returned
    * by getValues.
    */
   void getValues(double timevalue, std::vector<double> *valuesVector);

  protected:
   BaseProbe();

   /**
    * Calculates the global batch element corresponding to batch element 0 of the current process.
    */
   int calcGlobalBatchOffset();

   void initialize(const char *name, PVParams *params, Communicator const *comm);

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual void initMessageActionMap() override;

   /**
    * Called by registerData. If the MPIBlock row index and column index are
    * zero, this method sets a vector of PrintStreams whose size is the local
    * batch width. If probeOutputFile is being used, the elements of the vector
    * are FileStreams with filenames based on probeOutputFile: the global batch
    * index will be inserted in the probeOutputFile before the extension (or at
    * the end if there is no extension). If the MPIBlock row and column indices
    * are not both zero, the vector of PrintStreams will be empty - these
    * processes should communicate with the row=0,column=0 as needed.
    */
   virtual void initOutputStreams(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message);

   /**
    * Called by BaseProbe::initOutputStreams if StatsFlag is true.
    * The global root process sets up a single output stream, to the file specified
    * in ProbeOutputFile, or the log file if ProbeOutputFile is empty or null.
    */
   void
   initOutputStreamsStatsFlag(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message);

   /**
    * Called by BaseProbe::initOutputStreams if StatsFlag is false.
    * If the MPIBlock row index and column index are zero, this method sets a
    * vector of PrintStreams whose size is the local batch width. If
    * ProbeOutputFile is being used, the elements of the vector are FileStreams
    * with filenames based on probeOutputFile: the global batch index will be
    * inserted in the probeOutputFile before the extension (or at the end if
    * there is no extension). If the MPIBlock row and column indices are not
    * both zero, the vector of PrintStreams will be empty - these processes
    * should communicate with the row=0,column=0 process as needed.
    */
   void initOutputStreamsByBatchElement(
         std::shared_ptr<RegisterDataMessage<Checkpointer> const> message);

   /**
    * A pure virtual method for that should return true if the quantities being
    * measured by the probe have changed since the last time the quantities were
    * calculated.
    * Typically, an implementation of needRecalc() will check the lastUpdateTime
    * of the object being probed, and return true if that value is greater than
    * the lastUpdateTime member variable.
    * needRecalc() is called by getValues().
    * Note that there is a single needRecalc that applies to all getNumValues()
    * quantities.
    */
   virtual bool needRecalc(double timevalue) = 0;

   /**
    * A pure virtual method that should return the simulation time for the
    * values that calcValues() would compute if it were called instead.
    * The reason that this time might be different from the simuluation time at
    * which referenceUpdate was called, is that calcValues might be called
    * either before or after the update of the object the probe is attached to.
    *
    * The getValues() method calls this function after calling calcValues(),
    * and stores the result in the lastUpdateTime member variable.  Typically,
    * the implementation of needRecalc() will return true if lastUpdateTime is
    * less than the value returned by referenceUpdateTime, and false otherwise.
    */
   virtual double referenceUpdateTime(double simTime) const = 0;

   /**
    * A pure virtual method to calculate the values of the probe.  calcValues()
    * can assume that needRecalc() has been called and returned true.
    * It should write the computed values into the vector mProbeValues
    */
   virtual void calcValues(double timevalue) = 0;

   /**
    * If needRecalc() returns true, getValues(double) updates the mProbeValues
    * vector (by calling calcValues) and sets lastUpdateTime to the timevalue
    * input argument.
    */
   void getValues(double timevalue);

   /**
    * Returns a pointer to the message parameter.
    */
   const char *getMessage() { return msgstring; }

   /**
    * The method called by BaseProbe::initialize() to set the message used by
    * the probe's outputState method.
    */
   virtual int initMessage(const char *msg);

   /**
    * Returns a reference to the PrintStream for the given batch element
    */
   PrintStream &output(int b) { return *mOutputStreams.at(b); }

   /**
    * initNumValues() is called by initialize().
    * BaseProbe::initNumValues() sets size of the ProbeValues vector to the
    * parent HyPerCol's getNBatch(), by calling setNumValues(). Derived classes
    * can override initNumValues() to initialize the ProbeValues vector to a
    * different value.
    */
   virtual void initNumValues();

   /**
    * Sets the size of the ProbeValues vector (returned by getNumValues())
    * If the argument is negative, the mProbeValues vector is cleared
    * and getNumValues() will return zero. If the size is already positive
    * when setNumValues() is called with a positive value, the previous values
    * in the ProbeValues vector are not guaranteed to be preserved.
    */
   void setNumValues(int n);

   /**
    * Returns the probeOutputFilename parameter
    */
   char const *getProbeOutputFilename() { return mProbeOutputFilename; }

   /**
    * Returns a reference to the vector containing the probeValues.
    */
   std::vector<double> const &getProbeValues() const { return mProbeValues; }
   std::vector<double> &getProbeValues() { return mProbeValues; }

   /**
    * Returns the value of the textOutputFlag parameter
    */
   inline bool getTextOutputFlag() const { return textOutputFlag; }

   /**
    * Returns true if a probeOutputFile is being used.
    * Otherwise, returns false (indicating output is going to getOutputStream().
    */
   inline bool isWritingToFile() const { return mProbeOutputFilename and mProbeOutputFilename[0]; }

   /**
    * If there is a triggering layer, needUpdate returns true when the triggering
    * layer's
    * nextUpdateTime, modified by the probe's triggerOffset parameter, occurs;
    * otherwise false.
    * If there is not a triggering layer, needUpdate always returns true.
    * This behavior can be overridden if a probe uses some criterion other than
    * triggering
    * to choose when output its state.
    */
   virtual bool needUpdate(double simTime, double dt) const;

   /**
    * A pure virtual method for writing output to the output file when statsFlag is false.
    */
   virtual Response::Status outputState(double simTime, double deltaTime) = 0;

   /**
    * A pure virtual method for writing output to the output file when statsFlag is true.
    */
   virtual Response::Status outputStateStats(double simTime, double deltaTime) = 0;

   /**
    * calls flushOutputStream(), so that when checkpoints are written, the
    * output files are up to date.
    */
   virtual Response::Status prepareCheckpointWrite(double simTime) override;

   Response::Status
         respondProbeWriteParams(std::shared_ptr<ProbeWriteParamsMessage const>(message));

  private:
   int initialize_base();

   /**
    * This function first calls transferMPIOutput(), in case any MPISendStream objects
    * in the OutputStream vector have unsent data. Then, the nonroot processes send
    * their CurrentTag vectors to the root process over MPI, and the root process receives
    * messages unitl its CurrentTag vector has caught up with the nonroot processes'
    * CurrentTag vectors. Finally, the root process flushes the file streams associated
    * with all MPIRecvStream objects and all OutputStream objects.
    * This function is called when checkpoints are written and by the BaseProbe destructor.
    */
   void flushOutputStreams();

   /**
    * Increments the CurrentTag value associated with the given index.
    * Specifically, it adds one to mCurrentTag[index]; if the result reaches mTagLimit[index],
    * then mCurrentTag[index] wraps around to the value of mStartTag[index].
    */
   int incrementTag(int index);

   /**
    * Initializes CurrentTag, StartTag, and TagLimit to have vectorSize elements.
    * The first mLocalBatchWidth elements are BaseTag, BaseTag+spacing, BaseTag+2*spacing, etc.;
    * the remaining elements, if any, then repeat, beginning with BaseTag.
    * Here BaseTag is the static constant data member.
    * CurrentTag is initialized with the same values as StartTag.
    * TagLimit is initialized so that each element is the corresponding value of StartTag,
    * plus the value of the spacing argument.
    * The return value is the new value of the tag.
    */
   void initializeTagVectors(int vectorSize, int spacing);

   /**
    * Returns true if the Communicator has row 0, column 0, false otherwise.
    * Hence it is the base process for whichever batch elements live on the process.
    */
   bool isBatchBaseProc() const;

   /**
    * Returns true if the current process is the root process of the M-to-N communicator;
    * returns false otherwise.
    */
   bool isRootProc() const;

   /**
    * Calls the receive() method of the indicated MPIRecvStream, and increments the
    * corresponding element of the CurrentTag vector if a message was received.
    * Should only be called by the root process of the M-to-N communicator.
    */
   void receive(int batchProcessIndex, int localBatchIndex);

   /**
    * When this function member is called, nonroot processes send pending probe output using
    * MPISendStream::send(), and root processes check for probe output messages using
    * MPIRecvStream::receive(). These calls are non-blocking, so there may be unreceived
    * messages when this function returns. See the flushOutputStreams() function member if
    * it is necessary to make sure that all sent messages are received.
    */
   void transferMPIOutput();

   // Member variables
  protected:
   // A vector of PrintStreams, one for each batch element.
   std::vector<std::shared_ptr<PrintStream>> mOutputStreams;
   std::vector<MPIRecvStream> mMPIRecvStreams;

   bool triggerFlag;
   char *triggerLayerName;
   LayerUpdateController *mTriggerControl = nullptr;
   double triggerOffset;
   char *targetName;
   int mLocalBatchWidth = 1; // the value of loc->nbatch

  private:
   char *msgparams; // the message parameter in the params
   char *msgstring; // the string that gets printed by outputState ("" if message is empty or null;
                    // message + ":" if nonempty
   char *mProbeOutputFilename = nullptr;
   std::vector<double> mProbeValues;
   double lastUpdateTime; // The time of the last time calcValues was called.
   bool textOutputFlag;
   bool mStatsFlag = false; // Whether or not to take min, max or average over the batch

   // CurrentTag, StartTag, TagLimit allow for buffering of the probe output sent over MPI.
   // For root processes of the M-to-N communicator, which do I/O, the size of each of these
   // vectors is the local batch size times (MPIBlock->getBatchDimension() - 1).
   // (The minus one is because the root process doesn't use MPI for its own batch elements.)
   // For nonroot processes, which do not do I/O and must send its information to the root
   // process over MPI, the size of each of these vectors is the local batch size.
   // StartTag and TagLimit are set during initialization. They define ranges for each
   // batch element. Batch elements that live on the same process should not have overlapping
   // ranges, so that a given tag is always associated with one specific batch element.
   // Each time a nonroot process calls MPISendStream::send() for local batch element b,
   // it uses CurrentTag[b] as the tag argument, and then increments CurrentTag[b]; if the
   // result is TagLimit[b], it wraps around to StartTag[b].
   // The root process likewise maintains CurrentTag when it calls MPIRecvStream::receive(),
   // incrementing the tag when data is received. In this way, the root process does not have
   // to block, and printing the probe output for nonroot batch elements only has to be
   // synchronized when writing checkpoints or at the end of the run.
   std::vector<int> mCurrentTag;
   std::vector<int> mStartTag;
   std::vector<int> mTagLimit;

   // BaseTag and TagSpacing are used by initializeTagVectors() to initialize the CurrentTag,
   // StartTag, and TagLimit vectors.
   static int const mBaseTag    = 5000;
   static int const mTagSpacing = 10;

   // Each probe has a unique probe index. The static mNumProbes member keeps track of how many
   // indices have already been assigned, and is incremented each time a probe is added and its
   // probe index is assigned. The probe index is used by initializeTagVector() to make sure
   // different probes do not use the same tags.
   // Would it be better to have the HyPerCol assign a block of tags, the way random seeds are
   // handled?
   int mProbeIndex;
   static int mNumProbes;
   Timer *mInitialIOTimer     = nullptr;
   Timer *mInitialIOWaitTimer = nullptr;
   Timer *mIOTimer            = nullptr;
   Timer *mIOWaitTimer        = nullptr;
}; // class BaseProbe

} // namespace PV

#endif /* BASEPROBE_HPP_ */
