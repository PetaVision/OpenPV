/*
 * HyPerCol.hpp
 *
 *  Created on: Jul 30, 2008
 *      Author: Craig Rasmussen
 */

#ifndef HYPERCOL_HPP_
#define HYPERCOL_HPP_

#include <columns/InterColComm.hpp>
#include <columns/BaseObject.hpp>
#include <columns/Messages.hpp>
#include <columns/ObjectHierarchy.hpp>
#include <layers/HyPerLayer.hpp>
#include <connections/BaseConnection.hpp>
#include <io/PVParams.hpp>
#include <include/pv_types.h>
#include <columns/PV_Init.hpp>
#include <utils/Timer.hpp>
#include <io/ColProbe.hpp>
#include <time.h>
#include <typeinfo>
#include <sys/stat.h>
#include <fstream>
#include <string>
#include <sstream>

#ifdef PV_USE_CUDA
#  include <arch/cuda/CudaDevice.hpp>
#endif

#include <vector>

namespace PV {

enum CheckpointWriteTriggerMode {
   CPWRITE_TRIGGER_STEP,
   CPWRITE_TRIGGER_TIME,
   CPWRITE_TRIGGER_CLOCK
};

enum TimeUnit {
   CLOCK_SECOND,
   CLOCK_MINUTE,
   CLOCK_HOUR,
   CLOCK_DAY
};

class ColProbe;
class BaseProbe;
class PVParams;
class NormalizeBase;
class PV_Init;

class HyPerCol {

private:
   /** 
    * List of parameters needed from the HyPerCol class
    * @name HyPerCol Parameters
    * @{
    */

   /**
    * @brief startTime: The set starting time for the run
    */
   virtual void ioParam_startTime(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief stopTime: The set stopping time for the run
    */
   virtual void ioParam_stopTime(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief dt: The default delta time to use.
    * @details This dt is used for advancing the run time.
    */
   virtual void ioParam_dt(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief dtAdaptController: The name of a ColProbe to use for controlling the adaptive timestep.
    * The ColProbe's vectorSize (returned by getVectorSize()) must be the same as the HyPerCol's nBatch parameter.
    */
   virtual void ioParam_dtAdaptController(enum ParamsIOFlag ioFlag);
   
   /**
     * @brief dtAdaptFlag: Deprecated.  If mDtAdaptController is set to a ColProbe, then dtAdaptFlag is implied to be true.
     * If mDtAdaptController is null, dtAdaptFlag is implied to be false.
     * It is an error for dtAdaptFlag to be inconsistent with mDtAdaptController.
     */
   virtual void ioParam_dtAdaptFlag(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief : determines whether a time step adaptation method based on an expotential approximation of the energy is used, requires a mDtAdaptController
    */
   virtual void ioParam_useAdaptMethodExp1stOrder(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief dtAdaptTriggerLayerName: The name of a HyPerLayer that resets the adaptive time step scheme when it triggers.
    */
   virtual void ioParam_dtAdaptTriggerLayerName(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief triggerOffset: If triggerLayer is set, triggers \<triggerOffset\> timesteps before target trigger
    * @details Defaults to 0
    */
   virtual void ioParam_dtAdaptTriggerOffset(enum ParamsIOFlag ioFlag);

   /**
    * @brief dtScaleMax: If mDtAdaptController is set, specifies the maximum timescale allowed
    */
   virtual void ioParam_dtScaleMax(enum ParamsIOFlag ioFlag);

   /**
    * @brief dtScaleMax2: If mDtAdaptController is set, specifies the maximum dtScaleMax allowed (this is a 2nd maximum that adapts much more slowly)
    */
   virtual void ioParam_dtScaleMax2(enum ParamsIOFlag ioFlag);

   /**
    * @brief dtScaleMin: If mDtAdaptController is set, specifies the default timescale
    * @details The parameter name is misleading, since dtAdapt can drop below timescale min
    */
   virtual void ioParam_dtScaleMin(enum ParamsIOFlag ioFlag);

   /**
    * @brief dtChangeMax: If mDtAdaptController is set, specifies the upper limit of adaptive dt based on error
    * @details dt will only adapt if the percent change in error is between dtChangeMin and dtChangeMax
    */
   virtual void ioParam_dtChangeMax(enum ParamsIOFlag ioFlag);

   /**
    * @brief dtChangeMin: If mDtAdaptController is set, specifies the lower limit of adaptive dt based on error
    * @details dt will only adapt if the percent change in error is between dtChangeMin and dtChangeMax.
    * Defaults to 0
    */
   virtual void ioParam_dtChangeMin(enum ParamsIOFlag ioFlag);

   /**
    * @brief dtMinToleratedTimeScale: If mDtAdaptController is set, specifies the minimum value dt can drop to before exiting
    * @details Program will exit if timeScale drops below this value
    */
   virtual void ioParam_dtMinToleratedTimeScale(enum ParamsIOFlag ioFlag);

   /**
    * @brief writeTimeScaleFieldnames: A flag to determine if fieldnames are written to the HyPerCol_timescales file, if false, file is written as comma separated list
    */
   virtual void ioParam_writeTimeScaleFieldnames(enum ParamsIOFlag ioFlag);

   /**
    * @brief progressInterval: Specifies how often a progress report prints out
    * @details Units of dt
    */
   virtual void ioParam_progressInterval(enum ParamsIOFlag ioFlag);

   /**
    * @brief writeProgressToError: Whether to print timestep progress to the error stream instead of the output stream
    */
   virtual void ioParam_writeProgressToErr(enum ParamsIOFlag ioFlag);

   /**
    * @brief verifyWrites: If true, calls to PV_fwrite are checked by opening the file in read mode
    * and reading back the data and comparing it to the data just written.
    */
   virtual void ioParam_verifyWrites(enum ParamsIOFlag ioFlag);

   /**
    * @brief outputPath: Specifies the absolute or relative output path of the run
    */
   virtual void ioParam_outputPath(enum ParamsIOFlag ioFlag);

   /**
    * @brief printParamsFilename: Specifies the output params filename.
    * @details Defaults to pv.params. The output params file will be put into outputPath.
    */
   virtual void ioParam_printParamsFilename(enum ParamsIOFlag ioFlag);

   /**
    * @brief randomSeed: The seed for the random number generator for reproducability
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
    * @brief filenamesContainLayerNames: Specifies if layer names gets printed out to output connection pvp files
    * @details Options are 0, 1, or 2.
    * - 0: filenames have form a5.pvp
    * - 1: filenames form a5_NameOfLayer.pvp
    * - 2: filenames form NameOfLayer.pvp
    */
   virtual void ioParam_filenamesContainLayerNames(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief filenamesContainConnectionNames: Specifies if connection names gets printed out to output connection pvp files
    * @details Options are 0, 1, or 2.
    * - 0: filenames have form w5.pvp
    * - 1: filenames have form w5_NameOfConnection.pvp
    * - 2: filenames have form NameOfConnection.pvp
    */
   virtual void ioParam_filenamesContainConnectionNames(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief initializeFromChckpointDir: Sets directory for layers and connection to initialize from.
    */
   virtual void ioParam_initializeFromCheckpointDir(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief defaultInitializeFromCheckpointFlag: Flag to set the default for layers and connections.
    * @details Sets the default for layers and connections to use for initialize from checkpoint
    * based off of initializeFromCheckpointDir. Only used if initializeFromCheckpointDir is set.
    */
   virtual void ioParam_defaultInitializeFromCheckpointFlag(enum ParamsIOFlag ioFlag);

   /**
    * @brief checkpointRead is obsolete.  Instead use -c foo/Checkpoint100 on the command line.
    */
   virtual void ioParam_checkpointRead(enum ParamsIOFlag ioFlag);

   /**
    * @brief checkpointWrite: Flag to determine if the run writes checkpoints. 
    */
   virtual void ioParam_checkpointWrite(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief checkpointWriteDir: If checkpointWrite is set, specifies the output checkpoint directory.
    */
   virtual void ioParam_checkpointWriteDir(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief checkpointWriteTriggerMode: If checkpointWrite is set, specifies the method to checkpoint. 
    * @details Possible choices include
    * - step: Checkpoint off of timesteps
    * - time: Checkpoint off of simulation time
    * - clock: Checkpoint off of clock time. Not implemented yet.
    */
   virtual void ioParam_checkpointWriteTriggerMode(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief checkpointWriteStepInterval: If checkpointWrite on step, specifies the number of steps between checkpoints.
    */
   virtual void ioParam_checkpointWriteStepInterval(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief checkpointWriteTimeInteval: If checkpointWrite on time, specifies the amount of simulation time between checkpoints.
    */
   virtual void ioParam_checkpointWriteTimeInterval(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief checkpointWriteClockInteval: If checkpointWrite on clock, specifies the amount of clock time between checkpoints.  The units are specified using the parameter checkpointWriteClockUnit
    */
   virtual void ioParam_checkpointWriteClockInterval(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief checkpointWriteClockInteval: If checkpointWrite on clock, specifies the units used in checkpointWriteClockInterval.
    */
   virtual void ioParam_checkpointWriteClockUnit(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief deleteOlderCheckpoints: If checkpointWrite, specifies if the run should delete older checkpoints when writing new ones.
    */
   virtual void ioParam_deleteOlderCheckpoints(enum ParamsIOFlag ioFlag);

   /**
    * @brief numCheckpointsKept: If mDeleteOlderCheckpoints is set,
    * keep this many checkpoints before deleting the checkpoint.
    * Default is 1 (delete a checkpoint when a newer checkpoint is written.)
    */
   virtual void ioParam_numCheckpointsKept(enum ParamsIOFlag ioFlag);

   /**
    * @brief supressLastOutput: If checkpointWrite, specifies if the run should supress the final written checkpoint for the end of the run.
    */
   virtual void ioParam_suppressLastOutput(enum ParamsIOFlag ioFlag);

   /**
    * If checkpointWrite is true and this flag is true, connections' checkpointWrite method will only be called for connections with plasticityFlag=false.
    */
   virtual void ioParam_suppressNonplasticCheckpoints(enum ParamsIOFlag ioFlag);

   /**
    * @brief If checkpointWrite is true, checkpointIndexWidth specifies the minimum width for the step number appearing in the checkpoint directory.
    * @details If the step number needs fewer digits than checkpointIndexWidth, it is padded with zeroes.  If the step number needs more, the full
    * step number is still printed.  Hence, setting checkpointWrite to zero means that there are never any padded zeroes.
    * If set to a negative number, the width will be inferred from startTime, stopTime and dt.
    * The default value is -1 (infer the width).
    */
   virtual void ioParam_checkpointIndexWidth(enum ParamsIOFlag ioFlag);

   /**
    * @brief writeTimescales: If mDtAdaptController is set, specifies if the timescales should be written
    * @details The timescales get written to outputPath/HyPerCol_timescales.txt.
    */
   virtual void ioParam_writeTimescales(enum ParamsIOFlag ioFlag); 

   /**
    * @brief errorOnNotANumber: Specifies if the run should check on each timestep for nans in activity.
    */
   virtual void ioParam_errorOnNotANumber(enum ParamsIOFlag ioFlag);
   /** @} */

public:
   HyPerCol(const char * name, PV_Init* initObj);
   virtual ~HyPerCol();

   // Public functions

   BaseConnection* getConnFromName(const char* connectionName);
   BaseProbe* getBaseProbeFromName(const char* probeName);
   char* pathInCheckpoint(const char* cpDir, const char* objectName, const char* suffix);
   ColProbe* getColProbeFromName(const char* probeName);
   HyPerLayer* getLayerFromName(const char* layerName);

   /**
    * Adds an object (layer, connection, etc.) to the hierarchy.
    * Return value is true if the object was successfully added, and false otherwise.
    * The usual reason for failing to add the object is that the name is the same
    * as that of an earlier added object.
    */
   bool addObject(BaseObject * obj) { return mObjectHierarchy.addObject(obj); }
   int addBaseProbe(BaseProbe* p);
   int addConnection(BaseConnection* conn);
   int addNormalizer(NormalizeBase* normalizer);
   int addLayer(HyPerLayer* l);
   int advanceTime(double time);
   int ensureDirExists(const char* dirname);
   int exitRunLoop(bool exitOnFinish);
   int insertProbe(ColProbe* p);
   int loadState();
   int outputState(double time);
   int processParams(char const* path);
   int ioParamsFinishGroup(enum ParamsIOFlag);
   int ioParamsStartGroup(enum ParamsIOFlag ioFlag, const char* group_name);
   template <typename T>
   int readArrayFromFile(const char* cp_dir, const char* group_name, const char* val_name, T* val, size_t count, T default_value=(T) 0);
   template <typename T>
   int readScalarFromFile(const char* cp_dir, const char* group_name, const char* val_name, T* val, T default_value=(T) 0);
   int run() { return run(startTime, stopTime, deltaTimeBase); }
   int run(double startTime, double stopTime, double dt);
   template <typename T>
   int writeArrayToFile(const char* cp_dir, const char* group_name, const char* val_name, T*  val, size_t count);
   template <typename T>
   int writeScalarToFile(const char* cp_dir, const char* group_name, const char* val_name, T val);
   NormalizeBase* getNormalizerFromName(const char* normalizerName);
   template <typename T>
   void ioParamValueRequired(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, T * val);
   template <typename T>
   void ioParamValue(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, T * val, T defaultValue, bool warnIfAbsent=true);
   void ioParamString(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, char ** value, const char * defaultValue, bool warnIfAbsent=true);
   void ioParamStringRequired(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, char ** value);
   template <typename T>
   void ioParamArray(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, T ** value, int * arraysize);
   template <typename T>
   void writeParam(const char* param_name, T value);
   template <typename T>
   void writeParamArray(const char* param_name, const T* array, int arraysize);
   void writeParamString(const char* param_name, const char* svalue);

#ifdef PV_USE_CUDA
   BaseConnection* getGpuGroupConn(int gpuGroupIdx);
   int finalizeThreads();
   void addGpuGroup(BaseConnection* conn, int gpuGroupIdx);
#endif //PV_USE_CUDA

   // Getters and setters

   BaseConnection* getConnection(int which)  { return mConnections[which]; }
   BaseProbe* getBaseProbe(int which) { return mBaseProbes[which]; }
   bool getDtAdaptFlag() const { pvWarn() << "getDtAdaptFlag() is deprecated.\n" ; return usingAdaptiveTimeScale(); }  // getDtAdaptFlag() was deprecated Jul 7, 2016, in favor if usingAdaptiveTimeScale().
   bool getVerifyWrites() { return mVerifyWrites; }
   bool warmStartup() const { return mWarmStart; }
   bool usingAdaptiveTimeScale() const { return mDtAdaptController != nullptr; }
   bool getUseAdaptMethodExp1stOrder() const { return mUseAdaptMethodExp1stOrder; }
   bool getDefaultInitializeFromCheckpointFlag() { return mDefaultInitializeFromCheckpointFlag; }
   bool getCheckpointReadFlag() const { return mCheckpointReadFlag; }
   bool getCheckpointWriteFlag() const { return mCheckpointWriteFlag; }
   bool getSuppressLastOutputFlag() const {return mSuppressLastOutput; }
   bool getSuppressNonplasticCheckpoints() const { return mSuppressNonplasticCheckpoints; }
   bool getWriteTimescales() const { return mWriteTimescales; }
   const char * getName() { return mName; }
   const char * getSrcPath() { return srcPath; }
   const char * getOutputPath() { return outputPath; }
   const char * getInitializeFromCheckpointDir() const { return initializeFromCheckpointDir; }
   const char * getCheckpointReadDir() const { return mCheckpointReadDir; }
   const char * getPrintParamsFilename() const { return printParamsFilename; }
   ColProbe * getColProbe(int which) { return colProbes[which]; }
   double getDeltaTime() const { return deltaTime; }
   double getDeltaTimeBase() const { return deltaTimeBase; }
   double getTimeScale(int batch) const { pvAssert(batch >= 0 && batch < nbatch); return timeScale[batch]; }
   double getTimeScaleMax(int batch) const { pvAssert(batch >= 0 && batch < nbatch); return timeScaleMax[batch]; }
   double getTimeScaleMax() const { return timeScaleMaxBase; }
   double getTimeScaleMax2(int batch) const { assert(batch >= 0 && batch < nbatch); return timeScaleMax2[batch]; }
   double getTimeScaleMax2() const { return timeScaleMax2Base; }
   double getTimeScaleMin() const { return timeScaleMin; }
   double getChangeTimeScaleMax() const { return changeTimeScaleMax; }
   double getChangeTimeScaleMin() const { return changeTimeScaleMin; }
   double simulationTime() const { return simTime; }
   double getStartTime() const { return startTime; }
   double getStopTime() const { return stopTime; }
   double* getTimeScale() const { return timeScale; }
   double* getTimeScaleMaxPtr() const { return timeScaleMax; }
   double* getTimeScaleMax2Ptr() const { return timeScaleMax2; }
   HyPerLayer * getLayer(int which)       {return layers[which];}
   int globalRank() { return icComm->globalCommRank(); }
   int columnId() { return icComm->commRank(); }
   int getNxGlobal() { return nxGlobal; }
   int getNyGlobal() { return nyGlobal; }
   int getNBatch() { return nbatch; }
   int getNBatchGlobal() { return nbatchGlobal; }
   int getNumThreads() const { return numThreads;}
   int includeLayerName() const { return filenamesContainLayerNames;}
   int includeConnectionName() const { return filenamesContainConnectionNames;}
   int numberOfLayers() const { return numLayers;}
   int numberOfConnections() const { return mConnections.size();}
   int numberOfNormalizers() const { return numNormalizers;}
   int numberOfProbes() const {return numColProbes;}
   int numberOfBaseProbes() const {return numBaseProbes;}
   int numberOfBorderRegions() const {return MAX_NEIGHBORS;}
   int numberOfColumns() { return icComm->commSize(); }
   int numberOfGlobalColumns() { return icComm->globalCommSize(); }
   int commColumn() { return icComm->commColumn(); }
   int commRow() { return icComm->commRow(); }
   int commBatch() { return icComm->commBatch(); }
   int numCommColumns() { return icComm->numCommColumns(); }
   int numCommRows() { return icComm->numCommRows(); }
   int numCommBatches() { return icComm->numCommBatches(); }
   InterColComm * icCommunicator() const { return icComm; }
   NormalizeBase * getNormalizer(int which) { return normalizers[which]; }
   PV_Init * getPV_InitObj() const { return pv_initObj; }
   PV_Stream * getPrintParamsStream() const { return printParamsStream; }
   PVParams * parameters() const { return params; }
   long int getInitialStep() const { return initialStep; }
   long int getFinalStep() const { return finalStep; }
   long int getCurrentStep() const { return currentStep; }
   unsigned int getSeed() { return random_seed; }
   unsigned int getObjectSeed(int count) { unsigned long seed = random_seed_obj; random_seed_obj += count; return seed; }
   unsigned int getRandomSeed();
private:
   int getAutoGPUDevice();

#ifdef PV_USE_CUDA
public:
   PVCuda::CudaDevice * getDevice() { return cudaDevice; }
#endif

   // Private functions
 
private:
   bool advanceCPWriteTime();
   double* adaptTimeScale();
   double* adaptTimeScaleExp1stOrder();
   int initializeThreads(char const * in_device);
   int initialize_base();
   int initialize(const char * name, PV_Init* initObj);
   int ioParams(enum ParamsIOFlag ioFlag);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   int checkDirExists(const char * dirname, struct stat * pathstat);
   void notify(BaseMessage const * message);
   int doInitializationStage(int (HyPerCol::*layerInitializationStage)(int), int (HyPerCol::*connInitializationStage)(int), const char * stageName);
   int layerCommunicateInitInfo(int l);
   int connCommunicateInitInfo(int c);
   int layerAllocateDataStructures(int l);
   int connAllocateDataStructures(int c);
   int layerSetInitialValues(int l);
   int connSetInitialValues(int c);
   int normalizeWeights();
   int initPublishers();
   int checkpointRead();
   int checkpointWrite(const char * cpDir);
   int writeTimers(std::ostream& stream);
   int outputParams(char const * path);
   int outputParamsHeadComments(FILE* fp, char const * commentToken);
   int calcTimeScaleTrue();
   /**
    * Sets the numThreads member variable based on whether PV_USE_OPENMP is set
    * and the -t option in the PV_Arguments object.
    * If printMessagesFlag is true, it may print to the output and/or error stream.
    * If printMessagesFlag is false, these messages are suppressed.
    */
   int setNumThreads(bool printMessagesFlag);
   void initDtAdaptControlProbe();

   // Private variables

private:
   ObjectHierarchy mObjectHierarchy;
    std::vector<BaseConnection *> mConnections;
   BaseProbe ** mBaseProbes;
   bool mErrorOnNotANumber;        // If true, check each layer's activity buffer for not-a-numbers and exit with an error if any appear
   bool mDefaultInitializeFromCheckpointFlag ; // Each Layer and connection can individually set its own initializeFromCheckpointFlag.  This sets the default value for those flags.
   bool mWarmStart;             // whether to start from a checkpoint
   bool mCheckpointReadFlag;    // whether to load from a checkpoint directory
   bool mCheckpointWriteFlag;   // whether to write from a checkpoint directory
   bool mDeleteOlderCheckpoints; // If true, whenever a checkpoint other than the first is written, the preceding checkpoint is deleted. Default is false.
   bool mSuppressLastOutput; // If mCheckpointWriteFlag is false and this flag is false, on exit a checkpoint is sent to the {outputPath}/Last directory.
                            // If mCheckpointWriteFlag is false and this flag is true, no checkpoint is done on exit.
                            // The flag has no effect if mCheckpointWriteFlag is true (in which case a checkpoint is written on exit to the next directory in mCheckpointWriteDir
   bool mSuppressNonplasticCheckpoints; // If mSuppressNonplasticCheckpoints is true, only weights with plasticityFlag true will be checkpointed.  If false, all weights will be checkpointed.
   bool mReadyFlag;          // Initially false; set to true when communicateInitInfo, allocateDataStructures, and setInitialValues stages are completed
   bool mParamsProcessedFlag; // Initially false; set to true when processParams is called.
   bool mUseAdaptMethodExp1stOrder = true; // specifies whether exponential approximation to energy function decay is used to adapt time scale, requires dtAdaptControlProbe != NULL
   bool mWriteTimeScaleFieldnames;      // determines whether fieldnames are written to HyPerCol_timescales file
   bool mWriteProgressToError;// Whether to write progress step to standard error (True) or standard output (False) (default is output)
   bool mVerifyWrites;     // Flag to indicate whether calls to PV_fwrite do a readback check
   bool mOwnsParams; // True if params was created from params file by initialize, false if params was passed in the constructor
   bool mOwnsInterColComm; // True if icComm was created by initialize, false if passed in the constructor
   bool mWriteTimescales;
   char* mCheckpointReadDir;   // name of the directory to read an initializing checkpoint from
   char* mCheckpointReadDirBase;   // name of the directory containing checkpoint read from (used by deprecated params-based method for loading from checkpoint)
   char* mCheckpointWriteDir; // name of the directory to write checkpoints to
   char* mCheckpointWriteTriggerModeString;
   char* mCheckpointWriteClockUnit; // If checkpoint mode is clock, the string that specifies the units.  "seconds", "minutes", "hours", or "days".
   char* mDtAdaptController;       // If nonblank, the name of a ColProbe whose getValues() method is called to control timeScale
   char* mName;
   char* srcPath;        // path to PetaVision src directory (used to compile OpenCL kernels)
   char* outputPath;     // path to output file directory
   char* printParamsFilename; // filename for outputting the params, including defaults and excluding unread params
   char* dtAdaptTriggerLayerName;
   char * initializeFromCheckpointDir; // If nonempty, layers and connections can load from this directory as in checkpointRead, by setting their initializeFromCheckpointFlag parameter, but the run still starts at simTime=startTime
   ColProbe * dtAdaptControlProbe; // The probe pointed to by mDtAdaptController, dtAdaptControlProbe->getValues() is used to control timeScale.  If blank, use the original method
   ColProbe ** colProbes;
   double startTime;
   double simTime;          // current time in milliseconds
   double stopTime;         // time to stop time
   double deltaTime;        // time step interval
   double cpWriteTimeInterval;
   double nextCPWriteTime;
   double cpWriteClockInterval; // If checkpoint mode is clock, the clock time between checkpoints, in the units specified by checkpointWriteClockUnit
   double dtAdaptTriggerOffset;
   double deltaTimeBase;    // base time step interval if mDtAdaptController is used; timeScale is applied to this value
   double timeScaleMaxBase;     // default value of maximum value of timeScale 
   double timeScaleMax2Base;     // default value of maximum value of timeScaleMax 
   double timeScaleMin;     // minimum value of timeScale (not really a minimum, actually sets starting/iniital value of deltaTime)
   double changeTimeScaleMax;     // maximum change in value of timeScale (prevents deltaTime from growing too quickly)
   double changeTimeScaleMin;     // typically 0 or negative, maximum DECREASE in timeScale allowed before resetting timeScale -> timeScaleMin
   double dtMinToleratedTimeScale;// Exits with an error if any layer returns a timeScale between zero and this amount
   double progressInterval; // Output progress after simTime increases by this amount.
   double nextProgressTime; // Next time to output a progress message
   double* timeScale;        // scale factor for deltaTimeBase, deltaTime = timeScale*deltaTimeBase
   double* timeScaleTrue;    // true timeScale returned by min(HyPerLayer::getTimeScale) before MIN/MAX/CHANGE constraints applied
   double* oldTimeScale;        // old value of timeScale
   double* oldTimeScaleTrue;    // old value of timeScaleTrue
   double* deltaTimeAdapt;    // Actual deltaTimeAdapt buffer passed to updateState
   double* timeScaleMax;     // maximum value of timeScale (prevents deltaTime from growing too large)
   double* timeScaleMax2;     // maximum value of timeScaleMax (prevents timeScaleMax from growing too large)
   enum CheckpointWriteTriggerMode checkpointWriteTriggerMode;
   HyPerLayer * dtAdaptTriggerLayer;
   HyPerLayer ** layers;
   int numLayers;
   int numPhases;
   int numNormalizers;
   int checkpointSignal;      // whether the process should checkpoint in response to an external signal
   int numCheckpointsKept; // If mDeleteOlderCheckpoints is true, does not delete a checkpoint until the specified number of more recent checkpoints have been written.  Default is 2.
   int oldCheckpointDirectoriesIndex; // A pointer to the oldest checkpoint in the oldCheckpointDirectories vector.
   int checkpointIndexWidth; // minimum width of the step number field in the name of a checkpoint directory; if needed the step number is padded with zeros.
   int nxGlobal;
   int nyGlobal;
   int nbatch;
   int nbatchGlobal;
   int numColProbes;
   int numBaseProbes;
   int filenamesContainLayerNames; // Controls the form of layers' clayer->activeFP
                                   // Value 0: layers have form a5.pvp
                                   // Value 1: layers have form a5_NameOfLayer.pvp
                                   // Value 2: layers have form NameOfLayer.pvp
   int filenamesContainConnectionNames; // Similar to filenamesContainLayerNames, but for connections
   int origStdOut;
   int origStdErr;
   int numThreads;
   int * layerStatus;
   int * connectionStatus;
   InterColComm * icComm; // manages communication between HyPerColumns};
   long int cpReadDirIndex;  // checkpoint number within mCheckpointReadDir to read
   long int cpWriteStepInterval;
   long int nextCPWriteStep;
   long int currentStep;
   long int initialStep;
   long int finalStep;
   NormalizeBase ** normalizers; // Objects for normalizing connections or groups of connections
   PV_Init * pv_initObj;
   PV_Stream * printParamsStream; // file pointer associated with printParamsFilename
   PV_Stream * luaPrintParamsStream; // file pointer associated with the output lua file
   PVParams     * params; // manages input parameters
   size_t layerArraySize;
   size_t connectionArraySize;
   size_t normalizerArraySize;
   std::vector<std::string> oldCheckpointDirectories; // A ring buffer of existing checkpoints, used if mDeleteOlderCheckpoints is true.
   std::ofstream timeScaleStream;
   std::vector<HyPerLayer*> recvLayerBuffer;
   std::vector<HyPerLayer*> updateLayerBufferGpu;
   std::vector<HyPerLayer*> updateLayerBuffer;
   Timer * runTimer;
   Timer * checkpointTimer;
   Timer ** phaseRecvTimers;
   time_t cpWriteClockSeconds; // If checkpoint mode is clock, the clock time between checkpoints, in seconds
   time_t nextCPWriteClock;
   unsigned int random_seed;
   unsigned int random_seed_obj;  // Objects that need to generate random numbers should request a seed from
                                  // the HyPerCol, saying how many they need (across all processes in an MPI run).
                                  // random_seed_obj is incremented by the number requested, so that everything
                                  // that needs a random seed gets a unique seed, and things are reproducible.
#ifdef PV_USE_CUDA
   //The list of GPU group showing which connection's buffer to use
   BaseConnection** gpuGroupConns;
   int numGpuGroup;
   PVCuda::CudaDevice * cudaDevice;    // object for running kernels on OpenCL device
#endif

}; // class HyPerCol

HyPerCol * createHyPerCol(PV_Init * pv_initObj);

} // namespace PV

#endif /* HYPERCOL_HPP_ */
