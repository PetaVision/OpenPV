/*
 * HyPerCol.hpp
 *
 *  Created on: Jul 30, 2008
 *      Author: Craig Rasmussen
 */

#ifndef HYPERCOL_HPP_
#define HYPERCOL_HPP_

#include "checkpointing/Checkpointer.hpp"
#include "columns/BaseObject.hpp"
#include "columns/Communicator.hpp"
#include "columns/Messages.hpp"
#include "columns/PV_Init.hpp"
#include "connections/BaseConnection.hpp"
#include "include/pv_types.h"
#include "io/PVParams.hpp"
#include "layers/HyPerLayer.hpp"
#include "observerpattern/Observer.hpp"
#include "observerpattern/ObserverTable.hpp"
#include "observerpattern/Subject.hpp"
#include "probes/ColProbe.hpp"
#include "utils/Clock.hpp"
#include "utils/Timer.hpp"
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <time.h>
#include <typeinfo>
#include <vector>

#ifdef PV_USE_CUDA
#include <arch/cuda/CudaDevice.hpp>
#endif

#include <vector>

namespace PV {

class ColProbe;
class BaseProbe;
class PVParams;
class NormalizeBase;
class PV_Init;

class HyPerCol : public Subject, Observer {

  private:
   /**
    * List of parameters needed from the HyPerCol class
    * @name HyPerCol Parameters
    * @{
    */

   /**
    * @brief mStartTime: The set starting time for the run
    */
   virtual void ioParam_startTime(enum ParamsIOFlag ioFlag);

   /**
    * @brief mStopTime: The set stopping time for the run
    */
   virtual void ioParam_stopTime(enum ParamsIOFlag ioFlag);

   /**
    * @brief dt: The default delta time to use.
    * @details This dt is used for advancing the run time.
    */
   virtual void ioParam_dt(enum ParamsIOFlag ioFlag);

   /**
    * @brief mProgressInterval: Specifies how often a progress report prints out
    * @details Units of dt
    */
   virtual void ioParam_progressInterval(enum ParamsIOFlag ioFlag);

   /**
    * @brief writeProgressToErr: Whether to print timestep progress to the error
    * stream instead of
    * the output stream
    */
   virtual void ioParam_writeProgressToErr(enum ParamsIOFlag ioFlag);

   /**
    * @brief mPrintParamsFilename: Specifies the output mParams filename.
    * @details Defaults to pv.params. Relative paths are relative to
    * the OutputPath.
    */
   virtual void ioParam_printParamsFilename(enum ParamsIOFlag ioFlag);

   /**
    * @brief randomSeed: The seed for the random number generator for
    * reproducability
    */
   virtual void ioParam_randomSeed(enum ParamsIOFlag ioFlag);

   /**
    * @brief nx: Specifies the size of the column
    */
   virtual void ioParam_nx(enum ParamsIOFlag ioFlag);

   /**
    * @brief ny: Specifies the size of the column
    */
   virtual void ioParam_ny(enum ParamsIOFlag ioFlag);

   /**
    * @brief ny: Specifies the batch size of the column
    */
   virtual void ioParam_nBatch(enum ParamsIOFlag ioFlag);

   /**
    * @brief errorOnNotANumber: Specifies if the run should check on each
    * timestep for nans in activity.
    */
   virtual void ioParam_errorOnNotANumber(enum ParamsIOFlag ioFlag);

  public:
   HyPerCol(const char *name, PV_Init *initObj);
   virtual ~HyPerCol();

   // Public functions

   virtual int respond(std::shared_ptr<BaseMessage const> message) override;
   BaseConnection *getConnFromName(const char *connectionName);
   BaseProbe *getBaseProbeFromName(const char *probeName);
   ColProbe *getColProbeFromName(const char *probeName);
   HyPerLayer *getLayerFromName(const char *layerName);

   /**
    * Adds an object (layer, connection, etc.) to the hierarchy.
    * Exits with an error if adding the object failed.
    * The usual reason for failing to add the object is that the name is the same
    * as that of an earlier added object.
    * Currently, addLayer, addConnection, and addBaseProbe call addObject;
    * therefore it is usually not necessary to call addObject.
    */
   void addObject(BaseObject *obj);
   int addBaseProbe(BaseProbe *p);
   int addConnection(BaseConnection *conn);
   int addNormalizer(NormalizeBase *normalizer);
   int addLayer(HyPerLayer *l);
   void advanceTimeLoop(Clock &runClock, int const runClockStartingStep);
   int advanceTime(double time);
   void nonblockingLayerUpdate(
         std::shared_ptr<LayerRecvSynapticInputMessage const> recvMessage,
         std::shared_ptr<LayerUpdateStateMessage const> updateMessage);
   int insertProbe(ColProbe *p);
   int outputState(double time);
   int processParams(char const *path);
   int ioParamsFinishGroup(enum ParamsIOFlag);
   int ioParamsStartGroup(enum ParamsIOFlag ioFlag, const char *group_name);

   /**
    * This function tells each added object to perform the tasks necessary
    * before calling advanceTimeLoop.
    * Specifically, if mReadyFlag is not set, performs the CommunicateInitInfo,
    * AllocateDataStructures, and RegisterData stages, and outputs the
    * generated params file, and sets the mReadyFlag If mReadyFlag is set, does
    * nothing, so that the above stages are not performed more than once.
    * This method is called by the run() method.
    */
   void allocateColumn();
   int run() { return run(mStartTime, mStopTime, mDeltaTime); }
   int run(double mStartTime, double mStopTime, double dt);
   NormalizeBase *getNormalizerFromName(const char *normalizerName);

// Sep 26, 2016: HyPerCol methods for parameter input/output have been moved to
// PVParams.

#ifdef PV_USE_CUDA
   void addGpuGroup(BaseConnection *conn, int gpuGroupIdx);
#endif // PV_USE_CUDA

   // Getters and setters

   BaseConnection *getConnection(int which) { return mConnections.at(which); }
   BaseProbe *getBaseProbe(int which) { return mBaseProbes.at(which); }
   bool getVerifyWrites() { return mCheckpointer->doesVerifyWrites(); }
   bool getDefaultInitializeFromCheckpointFlag() {
      return mCheckpointer->getDefaultInitializeFromCheckpointFlag();
   }
   bool getCheckpointReadFlag() const { return mCheckpointReadFlag; }
   bool getCheckpointWriteFlag() const { return mCheckpointer->getCheckpointWriteFlag(); }
   char const *getLastCheckpointDir() const { return mCheckpointer->getLastCheckpointDir(); }
   bool getWriteTimescales() const { return mWriteTimescales; }
   const char *getName() { return mName; }
   const char *getOutputPath() { return mCheckpointer->getOutputPath().c_str(); }
   const char *getInitializeFromCheckpointDir() const {
      return mCheckpointer->getInitializeFromCheckpointDir();
   }
   const char *getPrintParamsFilename() const { return mPrintParamsFilename; }
   ColProbe *getColProbe(int which) { return mColProbes.at(which); }
   double getDeltaTime() const { return mDeltaTime; }
   // Sep 26, 2016: Adaptive timestep routines and member variables have been
   // moved to
   // AdaptiveTimeScaleProbe.
   double simulationTime() const { return mSimTime; }
   double getStartTime() const { return mStartTime; }
   double getStopTime() const { return mStopTime; }
   HyPerLayer *getLayer(int which) { return mLayers.at(which); }
   int globalRank() { return mCommunicator->globalCommRank(); }
   int columnId() { return mCommunicator->commRank(); }
   int getNxGlobal() { return mNumXGlobal; }
   int getNyGlobal() { return mNumYGlobal; }
   int getNBatch() { return mNumBatch; }
   int getNBatchGlobal() { return mNumBatchGlobal; }
   int getNumThreads() const { return mNumThreads; }
   int numberOfLayers() const { return mLayers.size(); }
   int numberOfConnections() const { return mConnections.size(); }
   int numberOfNormalizers() const { return mNormalizers.size(); }
   int numberOfProbes() const { return mColProbes.size(); }
   int numberOfBaseProbes() const { return mBaseProbes.size(); }
   int numberOfBorderRegions() const { return MAX_NEIGHBORS; }
   int numberOfColumns() { return mCommunicator->commSize(); }
   int numberOfGlobalColumns() { return mCommunicator->globalCommSize(); }
   int commColumn() { return mCommunicator->commColumn(); }
   int commRow() { return mCommunicator->commRow(); }
   int commBatch() { return mCommunicator->commBatch(); }
   int numCommColumns() { return mCommunicator->numCommColumns(); }
   int numCommRows() { return mCommunicator->numCommRows(); }
   int numCommBatches() { return mCommunicator->numCommBatches(); }
   Communicator *getCommunicator() const { return mCommunicator; }
   NormalizeBase *getNormalizer(int which) { return mNormalizers.at(which); }
   PV_Init *getPV_InitObj() const { return mPVInitObj; }
   FileStream *getPrintParamsStream() const { return mPrintParamsStream; }
   PVParams *parameters() const { return mParams; }
   long int getInitialStep() const { return mInitialStep; }
   long int getFinalStep() const { return mFinalStep; }
   unsigned int getRandomSeed() { return mRandomSeed; }
   unsigned int seedRandomFromWallClock();

   // A hack to allow test_cocirc, test_gauss2d, and test_post_weights to send a
   // CommunicateInitInfoMessage.
   std::map<std::string, Observer *> *copyObjectMap() {
      auto objectMap = new std::map<std::string, Observer *>;
      *objectMap     = mObjectHierarchy.getObjectMap();
      return objectMap;
   }

  private:
   int getAutoGPUDevice();

#ifdef PV_USE_CUDA
  public:
   BaseConnection *getGpuGroupConn(int gpuGroupIdx) { return mGpuGroupConns.at(gpuGroupIdx); }
   PVCuda::CudaDevice *getDevice() { return mCudaDevice; }
#endif

   // Private functions

  private:
   void setDescription();
   int initialize_base();
   int initialize(const char *name, PV_Init *initObj);
   int ioParams(enum ParamsIOFlag ioFlag);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   int checkDirExists(const char *dirname, struct stat *pathstat);
   inline void notify(std::vector<std::shared_ptr<BaseMessage const>> messages) {
      Subject::notify(mObjectHierarchy, messages, getCommunicator()->commRank() == 0 /*printFlag*/);
   }
   inline void notify(std::shared_ptr<BaseMessage const> message) {
      notify(std::vector<std::shared_ptr<BaseMessage const>>{message});
   }
   int respondPrepareCheckpointWrite(PrepareCheckpointWriteMessage const *message);
#ifdef PV_USE_CUDA
   void initializeCUDA(std::string const &in_device);
   int finalizeCUDA();
#endif // PV_USE_CUDA
   int normalizeWeights();
   int outputParams(char const *path);
   int outputParamsHeadComments(FileStream *fileStream, char const *commentToken);
   int calcTimeScaleTrue();
   /**
    * Sets the mNumThreads member variable based on whether PV_USE_OPENMP is set
    * and the NumThreads argument in the ConfigFile (-t option if using the
    * command line).  If printMessagesFlag is true, it may print to the output
    * and/or error stream.
    * If printMessagesFlag is false, these messages are suppressed.
    */
   int setNumThreads(bool printMessagesFlag);

   // Private variables

  private:
   std::vector<BaseConnection *> mConnections; // BaseConnection  ** mConnections;
   std::vector<BaseProbe *> mBaseProbes; // Why is this Base and not just
   // mProbes? //BaseProbe ** mBaseProbes;
   ObserverTable mObjectHierarchy;
   bool mErrorOnNotANumber; // If true, check each layer's activity buffer for
   // not-a-numbers and
   // exit with an error if any appear
   bool mCheckpointReadFlag; // whether to load from a checkpoint directory
   bool mReadyFlag; // Initially false; set to true when communicateInitInfo,
   // allocateDataStructures, and setInitialValues stages are completed
   bool mParamsProcessedFlag; // Initially false; set to true when processParams
   // is called.
   bool mWriteTimeScaleFieldnames; // determines whether fieldnames are written to
   // HyPerCol_timescales file
   bool mWriteProgressToErr; // Whether to write progress step to standard error
   // (True) or standard
   // output (False) (default is output)
   bool mOwnsCommunicator; // True if icComm was created by initialize, false if
   // passed in the
   // constructor
   bool mWriteTimescales;
   char *mName;
   char *mPrintParamsFilename; // filename for outputting the mParams, including
   // defaults and
   // excluding unread mParams
   std::vector<ColProbe *> mColProbes; // ColProbe ** mColProbes;
   double mStartTime;
   double mSimTime;
   double mStopTime; // time to stop time
   double mDeltaTime; // time step interval
   double mProgressInterval; // Output progress after mSimTime increases by this
   // amount.
   double mNextProgressTime; // Next time to output a progress message
   // Sep 26, 2016: Adaptive timestep routines and member variables have been
   // moved to
   // AdaptiveTimeScaleProbe.
   std::vector<HyPerLayer *> mLayers; // HyPerLayer ** mLayers;
   int mNumPhases;
   int mNumXGlobal;
   int mNumYGlobal;
   int mNumBatch;
   int mNumBatchGlobal;
   // mFilenamesContainLayerNames and mFilenamesContainConnectionNames were
   // removed Aug 12, 2016.
   int mOrigStdOut;
   int mOrigStdErr;
   int mNumThreads;
   int *mLayerStatus;
   int *mConnectionStatus;
   Communicator *mCommunicator; // manages communication between HyPerColumns};

   Checkpointer *mCheckpointer = nullptr; // manages checkpointing and, eventually,
   // will manage outputState output.
   long int mInitialStep;
   long int mCurrentStep;
   long int mFinalStep;
   std::vector<NormalizeBase *> mNormalizers; // NormalizeBase ** mNormalizers; // Objects for
   // normalizing mConnections or groups of mConnections
   PV_Init *mPVInitObj;
   FileStream *mPrintParamsStream; // file pointer associated with mPrintParamsFilename
   FileStream *mLuaPrintParamsStream; // file pointer associated with the output lua file
   PVParams *mParams; // manages input parameters
   size_t mLayerArraySize;
   size_t mConnectionArraySize;
   size_t mNormalizerArraySize;
   std::ofstream mTimeScaleStream;
   std::vector<HyPerLayer *> mRecvLayerBuffer;
   std::vector<HyPerLayer *> mUpdateLayerBufferGpu;
   std::vector<HyPerLayer *> mUpdateLayerBuffer;
   Timer *mRunTimer;
   std::vector<Timer *> mPhaseRecvTimers; // Timer ** mPhaseRecvTimers;
   unsigned int mRandomSeed;
#ifdef PV_USE_CUDA
   // The list of GPU group showing which connection's buffer to use
   std::vector<BaseConnection *> mGpuGroupConns; // BaseConnection** mGpuGroupConns;
   int mNumGpuGroup;
   PVCuda::CudaDevice *mCudaDevice; // object for running kernels on OpenCL device
#endif

}; // class HyPerCol

HyPerCol *createHyPerCol(PV_Init *pv_initObj);

} // namespace PV

#endif /* HYPERCOL_HPP_ */
