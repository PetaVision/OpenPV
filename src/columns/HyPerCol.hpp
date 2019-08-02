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
#include "columns/ParamsInterface.hpp"
#include "include/pv_types.h"
#include "io/PVParams.hpp"
#include "observerpattern/Observer.hpp"
#include "observerpattern/ObserverTable.hpp"
#include "observerpattern/Subject.hpp"
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

namespace PV {

class HyPerCol : public Subject, public ParamsInterface {

  private:
   /**
    * List of parameters needed from the HyPerCol class
    * @name HyPerCol Parameters
    * @{
    */

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
   HyPerCol(PV_Init *initObj);
   virtual ~HyPerCol();

   // Public functions

   /**
    * Returns the object in the hierarchy with the given name, if any exists.
    * Returns the null pointer if the string does not match any object.
    * It is up to the calling function to determine if the returned object
    * has the appropriate type.
    */
   Observer *getObjectFromName(std::string const &objectName) const;

   /**
    * Returns the object in the object hierarchy vector immediately following
    * the object passed as an argument. To get the first object,
    * pass the null pointer. If the last object is passed, the null
    * pointer is returned. If a non-null pointer is passed but is not
    * in the object hierarchy vector, an exception is thrown.
    */
   Observer *getNextObject(Observer const *currentObject) const;

   static void expandRecursive(ObserverTable *objectTable, ObserverTable const *table);

   ObserverTable getAllObjectsFlat();

   void advanceTimeLoop(Clock &runClock, int const runClockStartingStep);
   int advanceTime(double time);
   void nonblockingLayerUpdate(std::shared_ptr<LayerUpdateStateMessage const> updateMessage);
   void nonblockingLayerUpdate(
         std::shared_ptr<LayerRecvSynapticInputMessage const> recvMessage,
         std::shared_ptr<LayerUpdateStateMessage const> updateMessage);
   int processParams(char const *path);

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
   int run() { return run(mStopTime, mDeltaTime); }
   int run(double stopTime, double dt);

   // Getters and setters

   bool getVerifyWrites() { return mCheckpointer->doesVerifyWrites(); }
   bool getCheckpointWriteFlag() const { return mCheckpointer->getCheckpointWriteFlag(); }
   char const *getLastCheckpointDir() const { return mCheckpointer->getLastCheckpointDir(); }
   bool getWriteTimescales() const { return mWriteTimescales; }
   const char *getOutputPath() { return mCheckpointer->getOutputPath().c_str(); }
   const char *getPrintParamsFilename() const { return mPrintParamsFilename; }
   double getDeltaTime() const { return mDeltaTime; }
   double simulationTime() const { return mSimTime; }
   double getStopTime() const { return mStopTime; }
   int globalRank() { return mCommunicator->globalCommRank(); }
   int columnId() { return mCommunicator->commRank(); }
   int getNxGlobal() { return mNumXGlobal; }
   int getNyGlobal() { return mNumYGlobal; }
   int getNBatchGlobal() { return mNumBatchGlobal; }
   int getNumThreads() const { return mNumThreads; }
   int numberOfColumns() { return mCommunicator->commSize(); }
   int numberOfGlobalColumns() { return mCommunicator->globalCommSize(); }
   int commColumn() { return mCommunicator->commColumn(); }
   int commRow() { return mCommunicator->commRow(); }
   int commBatch() { return mCommunicator->commBatch(); }
   int numCommColumns() { return mCommunicator->numCommColumns(); }
   int numCommRows() { return mCommunicator->numCommRows(); }
   int numCommBatches() { return mCommunicator->numCommBatches(); }
   Communicator *getCommunicator() const { return mCommunicator; }
   PV_Init *getPV_InitObj() const { return mPVInitObj; }
   long int getFinalStep() const { return mFinalStep; }
   unsigned int getRandomSeed() { return mRandomSeed; }
   unsigned int seedRandomFromWallClock();

#ifdef PV_USE_CUDA
   PVCuda::CudaDevice *getDevice() { return mCudaDevice; }
#endif

   // Private functions

  private:
   int getAutoGPUDevice();
   int initialize_base();
   int initialize(PV_Init *initObj);
   virtual void initMessageActionMap() override;
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void fillComponentTable() override;
   void addComponent(BaseObject *component);
   inline void notifyLoop(std::vector<std::shared_ptr<BaseMessage const>> messages) {
      bool printFlag = getCommunicator()->globalCommRank() == 0;
      Subject::notifyLoop(messages, printFlag, getDescription());
   }
   inline void notifyLoop(std::shared_ptr<BaseMessage const> message) {
      notifyLoop(std::vector<std::shared_ptr<BaseMessage const>>{message});
   }
   Response::Status respondWriteParamsFile(std::shared_ptr<WriteParamsFileMessage const> message);
#ifdef PV_USE_CUDA
   void initializeCUDA(std::string const &in_device);
   int finalizeCUDA();
#endif // PV_USE_CUDA

   virtual Response::Status writeParamsFile(std::shared_ptr<WriteParamsFileMessage const> message);
   void outputParams(char const *path);
   void outputParamsHeadComments(FileStream *fileStream, char const *commentToken);
   /**
    * Sets the mNumThreads member variable based on whether PV_USE_OPENMP is set
    * and the NumThreads argument in the ConfigFile (-t option if using the
    * command line).
    */
   int setNumThreads();

   // Private variables

  private:
   bool mErrorOnNotANumber; // If true, check each layer's activity buffer for
   // not-a-numbers and
   // exit with an error if any appear
   bool mCheckpointReadFlag; // whether to load from a checkpoint directory
   bool mReadyFlag; // Initially false; set to true when communicateInitInfo,
   // allocateDataStructures, and initializeState stages are completed
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
   char *mPrintParamsFilename; // filename for outputting the mParams, including
   // defaults and
   // excluding unread mParams
   double mSimTime;
   double mStopTime; // time to stop time
   double mDeltaTime; // time step interval
   double mProgressInterval; // Output progress after mSimTime increases by this
   // amount.
   double mNextProgressTime; // Next time to output a progress message
   int mNumPhases;
   int mNumXGlobal;
   int mNumYGlobal;
   int mNumBatchGlobal;
   int mOrigStdOut;
   int mOrigStdErr;
   int mNumThreads;
   int *mLayerStatus;
   int *mConnectionStatus;
   Communicator *mCommunicator; // manages communication between HyPerColumns};

   Checkpointer *mCheckpointer = nullptr; // manages checkpointing and outputState output
   long int mCurrentStep;
   long int mFinalStep;
   PV_Init *mPVInitObj;
   PVParams *mParams; // manages input parameters
   size_t mLayerArraySize;
   size_t mConnectionArraySize;
   std::ofstream mTimeScaleStream;
   Timer *mRunTimer;
   std::vector<Timer *> mPhaseRecvTimers; // Timer ** mPhaseRecvTimers;
   unsigned int mRandomSeed;
#ifdef PV_USE_CUDA
   PVCuda::CudaDevice *mCudaDevice; // object for running kernels on OpenCL device
#endif

}; // class HyPerCol

// July 7, 2017: Functionality of createHyPerCol() moved into HyPerCol::initialize()

} // namespace PV

#endif /* HYPERCOL_HPP_ */
