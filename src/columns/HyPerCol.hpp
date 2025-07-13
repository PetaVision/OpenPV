/*
 * HyPerCol.hpp
 *
 *  Created on: Jul 30, 2008
 *      Author: Craig Rasmussen
 */

#ifndef HYPERCOL_HPP_
#define HYPERCOL_HPP_

#include "cMakeHeader.h"

#include "checkpointing/Checkpointer.hpp"
#include "checkpointing/CheckpointingMessages.hpp"
#include "columns/BaseObject.hpp"
#include "columns/Communicator.hpp"
#include "columns/Messages.hpp"
#include "columns/PV_Init.hpp"
#include "columns/ParamsInterface.hpp"
#include "io/FileStream.hpp"
#include "params/ParamsIO.hpp"
#include "params/PVParams.hpp"
#include "observerpattern/BaseMessage.hpp"
#include "observerpattern/Observer.hpp"
#include "observerpattern/ObserverTable.hpp"
#include "observerpattern/Response.hpp"
#include "observerpattern/Subject.hpp"
#include "utils/Clock.hpp"
#include "utils/Timer.hpp"
#include <fstream>
#include <memory>
#include <string>
#include <time.h>
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
   virtual void ioParam_stopTime(ParamsIOSwitch ioSwitch);

   /**
    * @brief dt: The default delta time to use.
    * @details This dt is used for advancing the run time.
    */
   virtual void ioParam_dt(ParamsIOSwitch ioSwitch);

   /**
    * @brief mProgressInterval: Specifies how often a progress report prints out
    * @details Units of dt
    */
   virtual void ioParam_progressInterval(ParamsIOSwitch ioSwitch);

   /**
    * @brief writeProgressToErr: Whether to print timestep progress to the error
    * stream instead of
    * the output stream
    */
   virtual void ioParam_writeProgressToErr(ParamsIOSwitch ioSwitch);

   /**
    * @brief printParamsFilename: Specifies the output mParams filename.
    * @details Defaults to pv.params. Relative paths are relative to
    * the OutputPath.
    */
   virtual void ioParam_printParamsFilename(ParamsIOSwitch ioSwitch);

   /**
    * @brief randomSeed: The seed for the random number generator for
    * reproducability
    */
   virtual void ioParam_randomSeed(ParamsIOSwitch ioSwitch);

   /**
    * @brief nx: Specifies the size of the column
    */
   virtual void ioParam_nx(ParamsIOSwitch ioSwitch);

   /**
    * @brief ny: Specifies the size of the column
    */
   virtual void ioParam_ny(ParamsIOSwitch ioSwitch);

   /**
    * @brief ny: Specifies the batch size of the column
    */
   virtual void ioParam_nBatch(ParamsIOSwitch ioSwitch);

   /**
    * @brief errorOnUnusedParam: If true, it is a fatal error if the params file contains
    * timestep for nans in activity. Default is false, in which case a warning is issued.
    */
   virtual void ioParam_errorOnUnusedParam(ParamsIOSwitch ioSwitch);

   /**
    * @brief errorOnNotANumber: Specifies if the run should check on each
    * timestep for not-a-number values in activity.
    */
   virtual void ioParam_errorOnNotANumber(ParamsIOSwitch ioSwitch);

   /**
    * @brief outputPath: Specifies the absolute or relative output path of the
    * run
    */
   virtual void ioParam_outputPath(ParamsIOSwitch ioSwitch);

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
   void processParams(std::string const &path);

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
   char const *getLastCheckpointDir() const { return mCheckpointer->getLastCheckpointDir().c_str(); }
   char const *getOutputPath() { return mOutputPath.c_str(); }
   std::string const &getPrintParamsFilename() const { return mPrintParamsFilename; }
   double getDeltaTime() const { return mDeltaTime; }
   double simulationTime() const { return mSimTime; }
   double getStopTime() const { return mStopTime; }
   int globalRank() { return mCommunicator->globalCommRank(); }
   int columnId() { return mCommunicator->commRank(); }
   int getNxGlobal() const { return mNumXGlobal; }
   int getNyGlobal() const { return mNumYGlobal; }
   int getNBatchGlobal() const { return mNumBatchGlobal; }
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

  protected:
   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   // Private functions

  private:
   int getAutoGPUDevice();
   int initialize(PV_Init *initObj);
   virtual void initMessageActionMap() override;
   int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;
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
   bool mErrorOnUnusedParam = false; // If true, error out if a params file param goes unused
   bool mErrorOnNotANumber  = false; // If true, check each layer's activity buffer for
   // not-a-numbers and exit with an error if any appear
   bool mCheckpointReadFlag = false; // whether to load from a checkpoint directory
   bool mReadyFlag          = false; // Initially false; set to true when communicateInitInfo,
   // allocateDataStructures, and initializeState stages are completed
   bool mParamsProcessedFlag = false; // Set to true when processParams() is called.
   bool mWriteProgressToErr  = false; // Whether to write progress step to standard error
   // (True) or standard output (False) (default is output)

   // filename for outputting the mParams, including defaults and excluding unread mParams
   std::string mPrintParamsFilename;

   std::string mOutputPath;
   double mSimTime;
   double mStopTime         = 0.0; // time to stop time
   double mDeltaTime        = mDefaultDeltaTime; // time step interval
   double mProgressInterval = 1.0; // Output progress after mSimTime increases by this amount.
   double mNextProgressTime; // Next time to output a progress message
   int mNumPhases = 0;
   std::vector<int> mIdleCounts; // How many times each phase had to wait for data to arrive
   int mNumXGlobal             = 0;
   int mNumYGlobal             = 0;
   int mNumBatchGlobal         = 1;
   int mNumThreads             = 1;
   Communicator *mCommunicator = nullptr; // manages communication between MPI processes;

   Checkpointer *mCheckpointer = nullptr; // manages checkpointing and outputState output
   long int mCurrentStep;
   long int mFinalStep;
   PV_Init *mPVInitObj;
   PVParams *mParams = nullptr; // manages input parameters
   std::ofstream mTimeScaleStream;
   Timer *mBuildAndRunTimer = nullptr;
   Timer *mBuildTimer       = nullptr;
   Timer *mRunTimer         = nullptr;
   unsigned int mRandomSeed = 0U;
#ifdef PV_USE_CUDA
   PVCuda::CudaDevice *mCudaDevice = nullptr; // object for running kernels on OpenCL device
#endif

   static double const mDefaultDeltaTime;
}; // class HyPerCol

} // namespace PV

#endif /* HYPERCOL_HPP_ */
