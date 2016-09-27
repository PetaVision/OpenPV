/*
 * HyPerCol.cpp
 *
 *  Created on: Jul 30, 2008
 *      Author: Craig Rasmussen
 */

#define TIMER_ON
#define DEFAULT_OUTPUT_PATH "output/"
#define DEFAULT_DELTA_T 1.0 //time step size (msec)
#define DEFAULT_NUMSTEPS 1

#include "HyPerCol.hpp"
#include "columns/Factory.hpp"
#include "columns/RandomSeed.hpp"
#include "columns/Communicator.hpp"
#include "normalizers/NormalizeBase.hpp"
#include "utils/Clock.hpp"
#include "io/io.hpp"

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <float.h>
#include <time.h>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <fts.h>
#include <fstream>
#include <time.h>
#include <csignal>
#include <limits>
#include <libgen.h>
#ifdef PV_USE_CUDA
#include <map>
#endif // PV_USE_CUDA

namespace PV {

HyPerCol::HyPerCol(const char * mName, PV_Init * initObj) {
   initialize_base();
   initialize(mName, initObj);
}

HyPerCol::~HyPerCol() {
#ifdef PV_USE_CUDA
   finalizeThreads();
#endif // PV_USE_CUDA
   writeTimers(getOutputStream());
   int rank = globalRank(); // Need to save so that we know whether we're the process that does I/O, even after deleting mCommunicator.

   for(auto iterator = mConnections.begin(); iterator != mConnections.end();) {
      delete *iterator;
      iterator = mConnections.erase(iterator);
   }
   for(auto iterator = mNormalizers.begin(); iterator != mNormalizers.end();) {
      delete *iterator;
      iterator = mNormalizers.erase(iterator);
   }   
   for(auto iterator = mPhaseRecvTimers.begin(); iterator != mPhaseRecvTimers.end();) { 
      delete *iterator;
      iterator = mPhaseRecvTimers.erase(iterator);
   }
   for(auto iterator = mLayers.begin(); iterator != mLayers.end();) {
      delete *iterator;
      iterator = mLayers.erase(iterator);
   }

   // mColProbes[i] should not be deleted; it points to an entry in mBaseProbes and will
   // be deleted when mBaseProbes is deleted
   for(auto iterator = mBaseProbes.begin(); iterator != mBaseProbes.end();) {
      delete *iterator;
      iterator = mBaseProbes.erase(iterator);
   }
   mColProbes.clear();
   
   //mCommunicator->clearPublishers();
   delete mRunTimer;
   delete mCheckpointTimer;
   //TODO: Change these old C strings into std::string
   free(mPrintParamsFilename);
   free(mOutputPath);
   free(mInitializeFromCheckpointDir);
   if (mCheckpointWriteFlag) {
      free(mCheckpointWriteDir); 
      mCheckpointWriteDir = nullptr;
      free(mCheckpointWriteTriggerModeString); 
      mCheckpointWriteTriggerModeString = nullptr;
   }
   if (mCheckpointReadFlag) {
      free(mCheckpointReadDir); 
      mCheckpointReadDir = nullptr;
      free(mCheckpointReadDirBase); mCheckpointReadDirBase = nullptr;
   }
}


int HyPerCol::initialize_base() {
   // Initialize all member variables to safe values.  They will be set to their actual values in initialize()
   mWarmStart = false;
   mReadyFlag = false;
   mParamsProcessedFlag = false;
   mCurrentStep = 0;
   mNumPhases = 0;
   mCheckpointReadFlag = false;
   mCheckpointWriteFlag = false;
   mCheckpointReadDir = nullptr;
   mCheckpointReadDirBase = nullptr;
   mCpReadDirIndex = -1L;
   mCheckpointWriteDir = nullptr;
   mCheckpointWriteTriggerMode = CPWRITE_TRIGGER_STEP;
   mCpWriteStepInterval = -1L;
   mNextCpWriteStep = 0L;
   mCpWriteTimeInterval = -1.0;
   mNextCpWriteTime = 0.0;
   mCpWriteClockInterval = -1.0;
   mDeleteOlderCheckpoints = false;
   mNumCheckpointsKept = 2;
   mOldCheckpointDirectoriesIndex = 0;
   mDefaultInitializeFromCheckpointFlag = false;
   mSuppressLastOutput = false;
   mSuppressNonplasticCheckpoints = false;
   mCheckpointIndexWidth = -1; // defaults to automatically determine index width
   mSimTime = 0.0;
   mStartTime = 0.0;
   mStopTime = 0.0;
   mDeltaTime = DEFAULT_DELTA_T;
   mWriteTimeScaleFieldnames = true;
   // Sep 26, 2016: Adaptive timestep routines and member variables have been moved to AdaptiveTimeScaleProbe.
   mProgressInterval = 1.0;
   mWriteProgressToErr = false;
   mOrigStdOut = -1;
   mOrigStdErr = -1;
   mLayers.clear();
   mConnections.clear();
   mNormalizers.clear(); //Pretty sure these aren't necessary
   mLayerStatus = nullptr;
   mConnectionStatus = nullptr;
   mOutputPath = nullptr;
   mPrintParamsFilename = nullptr;
   mPrintParamsStream = nullptr;
   mLuaPrintParamsStream = nullptr;
   mNumXGlobal = 0;
   mNumYGlobal = 0;
   mNumBatch = 1;
   mNumBatchGlobal = 1;
   mOwnsCommunicator = true;
   mParams = nullptr;
   mCommunicator = nullptr;
   mRunTimer = nullptr;
   mCheckpointTimer = nullptr;
   mPhaseRecvTimers.clear();
   mColProbes.clear();
   mBaseProbes.clear();
   mRandomSeed = 0U;
   //mRandomSeedObj = 0U;
   mErrorOnNotANumber = false;
   mNumThreads = 1;
   mRecvLayerBuffer.clear();
   mVerifyWrites = true; // Default for reading back and verifying when calling PV_fwrite
#ifdef PV_USE_CUDA
   mCudaDevice = nullptr;
   //mGpuGroupConns = nullptr;
   mGpuGroupConns.clear();
#endif
   return PV_SUCCESS;
}

int HyPerCol::initialize(const char * name, PV_Init* initObj)
{
   mPVInitObj = initObj;
   mCommunicator = mPVInitObj->getCommunicator();
   mParams = mPVInitObj->getParams();
   if(mParams == nullptr) {
      if (mCommunicator->globalCommRank()==0) {
         pvErrorNoExit() << "HyPerCol::initialize: params have not been set." << std::endl;
         MPI_Barrier(mCommunicator->communicator());
      }
      exit(EXIT_FAILURE);
   }
   int rank = mCommunicator->globalCommRank();
   char const * gpu_devices = mPVInitObj->getGPUDevices();
   char * working_dir = expandLeadingTilde(mPVInitObj->getWorkingDir());
   mWarmStart = mPVInitObj->getRestartFlag();

#ifdef PVP_DEBUG
   if (mPVInitObj->getRequireReturnFlag()) {
      if( rank == 0 ) {
         fflush(stdout);
         printf("Hit enter to begin! ");
         fflush(stdout);
         int charhit = -1;
         while(charhit != '\n') { charhit = getc(stdin); }
      }
      MPI_Barrier(mCommunicator->globalCommunicator());
   }
#endif // PVP_DEBUG

   mName = strdup(name);
   mRunTimer = new Timer(mName, "column", "run    ");
   mCheckpointTimer = new Timer(mName, "column", "checkpoint ");
   // Commented out in conversion to std::vector
   //mLayers = (HyPerLayer **) malloc(mLayerArraySize * sizeof(HyPerLayer *));
   //mConnections = (BaseConnection **) malloc(mConnectionArraySize * sizeof(BaseConnection *));
   //mNormalizers = (NormalizeBase **) malloc(mNormalizerArraySize * sizeof(NormalizeBase *));

   // mNumThreads will not be set, or used until HyPerCol::run.
   // This means that threading cannot happen in the initialization or communicateInitInfo stages,
   // but that should not be a problem.
   char const * programName = mPVInitObj->getProgramName();

   if(working_dir && columnId()==0) {
      int status = chdir(working_dir);
      if(status) {
         pvError(chdirMessage);
         chdirMessage.printf("Unable to switch directory to \"%s\"\n", working_dir);
         chdirMessage.printf("chdir error: %s\n", strerror(errno));
      }
   }
   free(working_dir); working_dir = nullptr;

#ifdef PV_USE_MPI // Fail if there was a parsing error, but make sure nonroot processes don't kill the root process before the root process reaches the syntax error
   int parsedStatus;
   int rootproc = 0;
   if( globalRank() == rootproc ) { 
      parsedStatus = this->mParams->getParseStatus();
   }
   MPI_Bcast(&parsedStatus, 1, MPI_INT, rootproc, getCommunicator()->globalCommunicator());
#else
   int parsedStatus = this->mParams->getParseStatus();
#endif
   if( parsedStatus != 0 ) {
      exit(parsedStatus);
   }

   if (mPVInitObj->getOutputPath()) {
      mOutputPath = expandLeadingTilde(mPVInitObj->getOutputPath());
      pvErrorIf(mOutputPath == nullptr, "HyPerCol::initialize unable to copy output path.\n");
   }

   mRandomSeed = mPVInitObj->getRandomSeed();
   ioParams(PARAMS_IO_READ);
   mCheckpointSignal = 0;
   mSimTime = mStartTime;
   mInitialStep = (long int) nearbyint(mStartTime/mDeltaTime);
   mCurrentStep = mInitialStep;
   mFinalStep = (long int) nearbyint(mStopTime/mDeltaTime);
   mNextProgressTime = mStartTime + mProgressInterval;

   RandomSeed::instance()->initialize(mRandomSeed);

   if(mCheckpointWriteFlag) {
      switch (mCheckpointWriteTriggerMode) {
      case CPWRITE_TRIGGER_STEP:
         mNextCpWriteStep = mInitialStep;
         mNextCpWriteTime = mStartTime; // Should be unnecessary
         mCpWriteTimeInterval = -1;
         mCpWriteClockInterval = -1.0;
         break;
      case CPWRITE_TRIGGER_TIME:
         mNextCpWriteStep = mInitialStep; // Should be unnecessary
         mNextCpWriteTime = mStartTime;
         mCpWriteStepInterval = -1;
         mCpWriteClockInterval = -1.0;
         break;
      case CPWRITE_TRIGGER_CLOCK:
         mNextCpWriteClock = time(nullptr);
         mCpWriteTimeInterval = -1;
         mCpWriteStepInterval = -1;
         break;
      default:
         assert(0); // All cases of mCheckpointWriteTriggerMode should have been covered above.
         break;
      }
   }

   //mWarmStart is set if command line sets the -r option.  PV_Arguments should prevent -r and -c from being both set.
   char const * checkpoint_read_dir = mPVInitObj->getCheckpointReadDir();
   pvAssert(!(mWarmStart && checkpoint_read_dir));
   if (mWarmStart) {
      mCheckpointReadDir = (char *) pvCallocError(PV_PATH_MAX, sizeof(char),
            "%s error: unable to allocate memory for path to checkpoint read directory.\n", programName);
      if (columnId()==0) {
         struct stat statbuf;
         // Look for directory "Last" in mOutputPath directory
         std::string cpDirString = mOutputPath;
         cpDirString += "/";
         cpDirString += "Last";
         if (PV_stat(cpDirString.c_str(), &statbuf)==0) {
            if (statbuf.st_mode & S_IFDIR) {
               strncpy(mCheckpointReadDir, cpDirString.c_str(), PV_PATH_MAX);
                  pvErrorIf(mCheckpointReadDir[PV_PATH_MAX-1], "%s error: checkpoint read directory \"%s\" too long.\n", programName, cpDirString.c_str());
            }
            else {
               pvError().printf("%s error: checkpoint read directory \"%s\" is not a directory.\n", programName, cpDirString.c_str()); 
            }
         }
         else if (mCheckpointWriteFlag) {
            // Last directory didn't exist; now look for mCheckpointWriteDir
            assert(mCheckpointWriteDir);
            cpDirString = mCheckpointWriteDir;
            if (cpDirString.c_str()[cpDirString.length()-1] != '/') {
               cpDirString += "/";
            }
            int statstatus = PV_stat(cpDirString.c_str(), &statbuf);
            if (statstatus==0) {
               if (statbuf.st_mode & S_IFDIR) {
                  char *dirs[] = {mCheckpointWriteDir, nullptr};
                  FTS * fts = fts_open(dirs, FTS_LOGICAL, nullptr);
                  FTSENT * ftsent = fts_read(fts);
                  bool found = false;
                  long int cp_index = LONG_MIN;
                  for (ftsent = fts_children(fts, 0); ftsent!=nullptr; ftsent=ftsent->fts_link) {
                     if (ftsent->fts_statp->st_mode & S_IFDIR) {
                        long int x;
                        int k = sscanf(ftsent->fts_name, "Checkpoint%ld", &x);
                        if (x>cp_index) {
                           cp_index = x;
                           found = true;
                        }
                     }
                  }
                  pvErrorIf(!found, "%s: restarting but Last directory does not exist and checkpointWriteDir directory \"%s\" does not have any checkpoints\n", programName, mCheckpointWriteDir);
                  int pathlen=snprintf(mCheckpointReadDir, PV_PATH_MAX, "%sCheckpoint%ld", cpDirString.c_str(), cp_index);
                  pvErrorIf(pathlen > PV_PATH_MAX, "%s error: checkpoint read directory \"%s\" too long.\n", programName, cpDirString.c_str());
               }
               else {
                  pvError().printf("%s error: checkpoint read directory \"%s\" is not a directory.\n", programName, mCheckpointWriteDir);
               }
            }
            else if (errno == ENOENT) {
               pvError().printf("%s error: restarting but neither Last nor checkpointWriteDir directory \"%s\" exists.\n", programName, mCheckpointWriteDir); 
            }
         }
         else { 
            pvError().printf("%s: restarting but Last directory does not exist and checkpointWriteDir is not defined (checkpointWrite=false)\n", programName); 
         }
      }
      MPI_Bcast(mCheckpointReadDir, PV_PATH_MAX, MPI_CHAR, 0, mCommunicator->communicator());
   }
   if (checkpoint_read_dir) {
      char * origChkPtr = strdup(mPVInitObj->getCheckpointReadDir());
      char** splitCheckpoint = (char**)pvCalloc(mCommunicator->numCommBatches(), sizeof(char*));
      size_t count = 0;
      char * tmp = nullptr;
      tmp = strtok(origChkPtr, ":");
      while(tmp != nullptr) {
         splitCheckpoint[count] = strdup(tmp);
         count++;
         pvErrorIf(count > mCommunicator->numCommBatches(), "Checkpoint read parsing error: Too many colon seperated checkpoint read directories. Only specify %d checkpoint directories.\n", mCommunicator->numCommBatches());
         tmp = strtok(nullptr, ":");
      }
      //Make sure number matches up
      pvErrorIf(count != mCommunicator->numCommBatches(), "Checkpoint read parsing error: Not enough colon seperated checkpoint read directories. Running with %d batch MPIs but only %zu colon seperated checkpoint directories.\n", mCommunicator->numCommBatches(), count);

      //Grab this rank's actual mCheckpointReadDir and replace with mCheckpointReadDir
      mCheckpointReadDir = expandLeadingTilde(splitCheckpoint[mCommunicator->commBatch()]);
      pvAssert(mCheckpointReadDir);
      //Free all tmp memories
      for(int i = 0; i < mCommunicator->numCommBatches(); i++){
         free(splitCheckpoint[i]);
      }
      free(splitCheckpoint);
      free(origChkPtr);

      mCheckpointReadFlag = true;
      pvInfo().printf("Global Rank %d process setting checkpointReadDir to %s.\n", globalRank(), mCheckpointReadDir);
   }

   // run only on GPU for now
#ifdef PV_USE_CUDA
   //Default to auto assign gpus
   initializeThreads(gpu_devices);
#endif
   gpu_devices = nullptr;

   //Only print rank for comm rank 0
   if(globalRank() == 0){
#ifdef PV_USE_CUDA
      mCudaDevice->query_device_info();
#endif
   }

   // If mDeleteOlderCheckpoints is true, set up a ring buffer of checkpoint directory names.
   pvAssert(mOldCheckpointDirectories.size() == 0);
   mOldCheckpointDirectories.resize(mNumCheckpointsKept, "");
   mOldCheckpointDirectoriesIndex = 0;
   return PV_SUCCESS;
}

int HyPerCol::ioParams(enum ParamsIOFlag ioFlag) {
   ioParamsStartGroup(ioFlag, mName);
   ioParamsFillGroup(ioFlag);
   ioParamsFinishGroup(ioFlag);
   return PV_SUCCESS;
}

int HyPerCol::ioParamsStartGroup(enum ParamsIOFlag ioFlag, const char * group_name) {
   if (ioFlag == PARAMS_IO_WRITE && columnId()==0) {
      pvAssert(mPrintParamsStream);
      pvAssert(mLuaPrintParamsStream);
      const char * keyword = mParams->groupKeywordFromName(group_name);
      fprintf(mPrintParamsStream->fp, "\n");
      fprintf(mPrintParamsStream->fp, "%s \"%s\" = {\n", keyword, group_name);
      fprintf(mLuaPrintParamsStream->fp, "%s = {\n", group_name);
      fprintf(mLuaPrintParamsStream->fp, "groupType = \"%s\";\n", keyword);
   }
   return PV_SUCCESS;
}

int HyPerCol::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_startTime(ioFlag);
   ioParam_dt(ioFlag);
   ioParam_dtAdaptController(ioFlag);
   ioParam_dtAdaptFlag(ioFlag);
   ioParam_useAdaptMethodExp1stOrder(ioFlag);
   ioParam_writeTimeScaleFieldnames(ioFlag);
   ioParam_dtAdaptTriggerLayerName(ioFlag);
   ioParam_dtAdaptTriggerOffset(ioFlag);
   ioParam_dtScaleMax(ioFlag);
   ioParam_dtScaleMin(ioFlag);
   ioParam_dtChangeMax(ioFlag);
   ioParam_dtChangeMin(ioFlag);
   ioParam_dtMinToleratedTimeScale(ioFlag);
   ioParam_stopTime(ioFlag);
   ioParam_progressInterval(ioFlag);
   ioParam_writeProgressToErr(ioFlag);
   ioParam_verifyWrites(ioFlag);
   ioParam_outputPath(ioFlag);
   ioParam_printParamsFilename(ioFlag);
   ioParam_randomSeed(ioFlag);
   ioParam_nx(ioFlag);
   ioParam_ny(ioFlag);
   ioParam_nBatch(ioFlag);
   ioParam_filenamesContainLayerNames(ioFlag);
   ioParam_filenamesContainConnectionNames(ioFlag);
   ioParam_initializeFromCheckpointDir(ioFlag);
   ioParam_defaultInitializeFromCheckpointFlag(ioFlag);
   ioParam_checkpointRead(ioFlag); // checkpointRead is obsolete as of June 27, 2016.
   ioParam_checkpointWrite(ioFlag);
   ioParam_checkpointWriteDir(ioFlag);
   ioParam_checkpointWriteTriggerMode(ioFlag);
   ioParam_checkpointWriteStepInterval(ioFlag);
   ioParam_checkpointWriteTimeInterval(ioFlag);
   ioParam_checkpointWriteClockInterval(ioFlag);
   ioParam_checkpointWriteClockUnit(ioFlag);
   ioParam_deleteOlderCheckpoints(ioFlag);
   ioParam_numCheckpointsKept(ioFlag);
   ioParam_suppressLastOutput(ioFlag);
   ioParam_suppressNonplasticCheckpoints(ioFlag);
   ioParam_checkpointIndexWidth(ioFlag);
   ioParam_writeTimescales(ioFlag);
   ioParam_errorOnNotANumber(ioFlag);

   // Aug 18, 2016.  Several HyPerCol parameters have moved to AdaptiveTimeScaleProbe.
   // The ioParam_* method for those parameters sets the mObsoleteParameterFound flag
   // if the parameter is present in the HyPerCol group.  Exit with an error here
   // if any of those parameters were found.
   if (mObsoleteParameterFound) {
      if (getCommunicator()->commRank()==0) {
         pvErrorNoExit() << "Exiting due to obsolete HyPerCol parameters in the params file.\n";
      }
      MPI_Barrier(getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}

int HyPerCol::ioParamsFinishGroup(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_WRITE && columnId()==0) {
      pvAssert(mPrintParamsStream);
      pvAssert(mLuaPrintParamsStream);
      fprintf(mPrintParamsStream->fp, "};\n");
      fprintf(mLuaPrintParamsStream->fp, "};\n\n");
   }
   return PV_SUCCESS;
}

// Sep 26, 2016: HyPerCol methods for parameter input/output have been moved to PVParams.

void HyPerCol::ioParam_startTime(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, mName, "startTime", &mStartTime, mStartTime);
}

void HyPerCol::ioParam_dt(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, mName, "dt", &mDeltaTime, mDeltaTime);
   // mDeltaTimeBase = mDeltaTime;  // use param value as base
}

void HyPerCol::ioParam_dtAdaptController(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && mParams->stringPresent(mName, "dtAdaptController")) {
      if (columnId()==0) {
         pvErrorNoExit() << "The dtAdaptController parameter is obsolete.  Use the AdaptiveTimeScaleProbe targetName parameter.\n";
      }
      mObsoleteParameterFound = true;
   }
}

void HyPerCol::ioParam_dtAdaptFlag(enum ParamsIOFlag ioFlag) {
   // dtAdaptFlag was deprecated Feb 1, 2016 and marked obsolete Aug 18, 2016.
   if (ioFlag==PARAMS_IO_READ && mParams->present(mName, "dtAdaptFlag")) {
      if (columnId()==0) {
         pvErrorNoExit() << "The dtAdaptFlag parameter is obsolete.  Define an AdaptiveTimeScaleProbe\n";
      }
   mObsoleteParameterFound = true;
   }
}

// Several HyPerCol parameters were moved to AdaptiveTimeScaleProbe and are therefore obsolete as HyPerCol Parameters, Aug 18, 2016.
void HyPerCol::paramMovedToColumnEnergyProbe(enum ParamsIOFlag ioFlag, char const * paramName) {
   if (ioFlag==PARAMS_IO_READ && mParams->present(mName, paramName)) {
      if (columnId()==0) {
         pvErrorNoExit() << "The " << paramName << " parameter is now part of AdaptiveTimeScaleProbe.\n";
      }
      mObsoleteParameterFound = true;
   }
}

void HyPerCol::ioParam_writeTimescales(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && mParams->present(mName, "dtAdaptTriggerOffset")) {
      if (columnId()==0) {
         pvErrorNoExit() << "The dtAdaptTriggerOffset parameter is obsolete.  Use the AdaptiveTimeScaleProbe writeTimeScales parameter (note capital S).\n";
      }
      mObsoleteParameterFound = true;
   }
}

void HyPerCol::ioParam_writeTimeScaleFieldnames(enum ParamsIOFlag ioFlag) {
   paramMovedToColumnEnergyProbe(ioFlag, "writeTimeScaleFieldnames");
}

void HyPerCol::ioParam_useAdaptMethodExp1stOrder(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && mParams->present(mName, "useAdaptMethodExp1stOrder")) {
      if (columnId()==0) {
         pvErrorNoExit() << "The useAdaptMethodExp1stOrder parameter is obsolete.  Adapting the timestep always uses the Exp1stOrder method.\n";
      }
      mObsoleteParameterFound = true;
   }
}

void HyPerCol::ioParam_dtAdaptTriggerLayerName(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && mParams->stringPresent(mName, "dtAdaptTriggerLayerName")) {
      if (columnId()==0) {
         pvErrorNoExit() << "The dtAdaptTriggerLayerName parameter obsolete.  Use the AdaptiveTimeScaleProbe triggerLayerName parameter.\n";
      }
      mObsoleteParameterFound = true;
   }
}

void HyPerCol::ioParam_dtAdaptTriggerOffset(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && mParams->present(mName, "dtAdaptTriggerOffset")) {
      if (columnId()==0) {
         pvErrorNoExit() << "The dtAdaptTriggerOffset parameter is obsolete.  Use the AdaptiveTimeScaleProbe triggerOffset parameter\n";
      }
      mObsoleteParameterFound = true;
   }
}

void HyPerCol::ioParam_dtScaleMax(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && mParams->present(mName, "dtScaleMax")) {
      if (columnId()==0) {
         pvErrorNoExit() << "The dtScaleMax parameter is obsolete.  Use the AdaptiveTimeScaleProbe baseMax parameter\n";
      }
      mObsoleteParameterFound = true;
   }
}

void HyPerCol::ioParam_dtScaleMax2(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && mParams->present(mName, "dtAdaptTriggerOffset")) {
      if (columnId()==0) {
         pvErrorNoExit() << "The dtScaleMax2 parameter has been removed.\n";
      }
      mObsoleteParameterFound = true;
   }
}

void HyPerCol::ioParam_dtScaleMin(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && mParams->present(mName, "dtScaleMin")) {
      if (columnId()==0) {
         pvErrorNoExit() << "The dtScaleMin parameter is obsolete.  Use the AdaptiveTimeScaleProbe baseMin parameter\n";
      }
      mObsoleteParameterFound = true;
   }
}

void HyPerCol::ioParam_dtMinToleratedTimeScale(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && mParams->present(mName, "dtScaleMin")) {
      if (columnId()==0) {
         pvErrorNoExit() << "The dtMinToleratedTimeScale parameter has been removed.\n";
      }
      mObsoleteParameterFound = true;
   }
}

void HyPerCol::ioParam_dtChangeMax(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && mParams->present(mName, "dtChangeMax")) {
      if (columnId()==0) {
         pvErrorNoExit() << "The dtChangeMax parameter is obsolete.  Use the AdaptiveTimeScaleProbe tauFactor parameter\n";
      }
      mObsoleteParameterFound = true;
   }
}

void HyPerCol::ioParam_dtChangeMin(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && mParams->present(mName, "dtChangeMax")) {
      if (columnId()==0) {
         pvErrorNoExit() << "The dtChangeMin parameter is obsolete.  Use the AdaptiveTimeScaleProbe growthFactor parameter\n";
      }
      mObsoleteParameterFound = true;
   }
}

void HyPerCol::ioParam_stopTime(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && !mParams->present(mName, "stopTime") && mParams->present(mName, "numSteps")) {
      assert(!mParams->presentAndNotBeenRead(mName, "startTime"));
      assert(!mParams->presentAndNotBeenRead(mName, "dt"));
      long int numSteps = mParams->value(mName, "numSteps");
      mStopTime = mStartTime + numSteps * mDeltaTime;
      if (globalRank()==0) {
         pvError() << "numSteps is obsolete.  Use startTime, stopTime and dt instead.\n" <<
               "    stopTime set to " << mStopTime << "\n";
      }
      MPI_Barrier(getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   // numSteps was deprecated Dec 12, 2013 and marked obsolete Jun 27, 2016
   // After a reasonable fade time, remove the above if-statement and keep the ioParamValue call below.
   parameters()->ioParamValue(ioFlag, mName, "stopTime", &mStopTime, mStopTime);
}

void HyPerCol::ioParam_progressInterval(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && !mParams->present(mName, "progressInterval") && mParams->present(mName, "progressStep")) {
      long int progressStep = (long int) mParams->value(mName, "progressStep");
      mProgressInterval = progressStep/mDeltaTime;
      if (globalRank()==0) {
         pvErrorNoExit() << "progressStep is obsolete.  Use progressInterval instead.\n";
      }
      MPI_Barrier(getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   // progressStep was deprecated Dec 18, 2013
   // After a reasonable fade time, remove the above if-statement and keep the ioParamValue call below.
   parameters()->ioParamValue(ioFlag, mName, "progressInterval", &mProgressInterval, mProgressInterval);
}

void HyPerCol::ioParam_writeProgressToErr(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, mName, "writeProgressToErr", &mWriteProgressToErr, mWriteProgressToErr);
}

void HyPerCol::ioParam_verifyWrites(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, mName, "verifyWrites", &mVerifyWrites, mVerifyWrites);
}

void HyPerCol::ioParam_outputPath(enum ParamsIOFlag ioFlag) {
   // mOutputPath can be set on the command line.
   switch(ioFlag) {
   case PARAMS_IO_READ:
      if (mOutputPath==nullptr) {
         if( mParams->stringPresent(mName, "outputPath") ) {
            const char* strval = mParams->stringValue(mName, "outputPath");
            pvAssert(strval);
            mOutputPath = strdup(strval);
            pvAssert(mOutputPath != nullptr);
         }
         else {
            mOutputPath = strdup(DEFAULT_OUTPUT_PATH);
            pvAssert(mOutputPath != nullptr);
            pvWarn().printf("Output path specified neither in command line nor in params file.\n"
                   "Output path set to default \"%s\"\n", DEFAULT_OUTPUT_PATH);
         }
      }
      break;
   case PARAMS_IO_WRITE:
      parameters()->writeParamString("outputPath", mOutputPath);
      break;
   default: break;
   }
}

void HyPerCol::ioParam_printParamsFilename(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamString(ioFlag, mName, "printParamsFilename", &mPrintParamsFilename, "pv.params");
   if (mPrintParamsFilename==nullptr || mPrintParamsFilename[0]=='\0') {
      if (columnId()==0) {
         pvErrorNoExit().printf("printParamsFilename cannot be null or the empty string.\n");
      }
      MPI_Barrier(getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
}

void HyPerCol::ioParam_randomSeed(enum ParamsIOFlag ioFlag) {
   switch(ioFlag) {
   // randomSeed can be set on the command line, from the params file, or from the system clock
   case PARAMS_IO_READ:
      // set random seed if it wasn't set in the command line
      // bool seedfromclock = false;
      if( !mRandomSeed ) {
         if( mParams->present(mName, "randomSeed") ) {
            mRandomSeed = (unsigned long) mParams->value(mName, "randomSeed");
         }
         else {
            mRandomSeed = seedRandomFromWallClock();
         }
      }
      if (mRandomSeed < RandomSeed::minSeed) {
         pvError().printf("Error: random seed %u is too small. Use a seed of at least 10000000.\n", mRandomSeed);
      }
      break;
   case PARAMS_IO_WRITE:
      parameters()->writeParam("randomSeed", mRandomSeed);
      break;
   default:
      assert(0);
      break;
   }
}

void HyPerCol::ioParam_nx(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValueRequired(ioFlag, mName, "nx", &mNumXGlobal);
}

void HyPerCol::ioParam_ny(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValueRequired(ioFlag, mName, "ny", &mNumYGlobal);
}

void HyPerCol::ioParam_nBatch(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, mName, "nbatch", &mNumBatchGlobal, mNumBatchGlobal);
   //Make sure numCommBatches is a multiple of mNumBatch specified in the params file
   pvErrorIf(mNumBatchGlobal % mCommunicator->numCommBatches() != 0, 
         "The total number of batches (%d) must be a multiple of the batch width (%d)\n", mNumBatchGlobal, mCommunicator->numCommBatches());
   mNumBatch = mNumBatchGlobal / mCommunicator->numCommBatches();
}

void HyPerCol::ioParam_filenamesContainLayerNames(enum ParamsIOFlag ioFlag) {
   //filenamesContainConnectionNames was marked obsolete Aug 12, 2016.
   if (parameters()->present(mName, "filenamesContainLayerNames")) {
      double fccnValue = parameters()->value(mName, "filenamesContainLayerNames");
      std::string msg("The HyPerCol parameter \"filenamesContainLayerNames\" is obsolete.\n");
      msg.append("Layer output pvp files have the format \"NameOfConnection.pvp\"\n");
      msg.append("(corresponding to filenamesContainLayerNames=2).\n");
      if (fccnValue==2) { pvWarn() << msg; } else { pvError() << msg; }
   }
}

void HyPerCol::ioParam_filenamesContainConnectionNames(enum ParamsIOFlag ioFlag) {
   //filenamesContainConnectionNames was marked obsolete Aug 12, 2016.
   if (parameters()->present(mName, "filenamesContainConnectionNames")) {
      double fccnValue = parameters()->value(mName, "filenamesContainConnectionNames");
      std::string msg("The HyPerCol parameter \"filenamesContainConnectionNames\" is obsolete.\n");
      msg.append("Connection output pvp files have the format \"NameOfConnection.pvp\"\n");
      msg.append("(corresponding to filenamesContainConnectionNames=2).\n");
      if (fccnValue==2) { pvWarn() << msg; } else { pvError() << msg; }
   }
}

void HyPerCol::ioParam_initializeFromCheckpointDir(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamString(ioFlag, mName, "initializeFromCheckpointDir", &mInitializeFromCheckpointDir, "", true);
}

void HyPerCol::ioParam_defaultInitializeFromCheckpointFlag(enum ParamsIOFlag ioFlag) {
   assert(!mParams->presentAndNotBeenRead(mName, "initializeFromCheckpointDir"));
   if (mInitializeFromCheckpointDir != nullptr && mInitializeFromCheckpointDir[0] != '\0') {
      parameters()->ioParamValue(ioFlag, mName, "defaultInitializeFromCheckpointFlag", &mDefaultInitializeFromCheckpointFlag, mDefaultInitializeFromCheckpointFlag, true);
   }
}

// Error out if someone uses obsolete checkpointRead flag in params.
// After a reasonable fade time, this function can be removed.
void HyPerCol::ioParam_checkpointRead(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && mParams->stringPresent(mName, "checkpointRead")) {
      if (columnId()==0) {
         pvErrorNoExit() << "The checkpointRead params file parameter is obsolete." <<
               "  Instead, set the checkpoint directory on the command line.\n";
      }
      MPI_Barrier(getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
}

// athresher, July 20th
// Removed ioParam_checkpointRead(). It was marked obsolete.

void HyPerCol::ioParam_checkpointWrite(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, mName, "checkpointWrite", &mCheckpointWriteFlag, false);
}

void HyPerCol::ioParam_checkpointWriteDir(enum ParamsIOFlag ioFlag) {
   pvAssert(!mParams->presentAndNotBeenRead(mName, "checkpointWrite"));
   if (mCheckpointWriteFlag) {
      parameters()->ioParamStringRequired(ioFlag, mName, "checkpointWriteDir", &mCheckpointWriteDir); 
   }
   else { 
      mCheckpointWriteDir = nullptr; 
   }
}

void HyPerCol::ioParam_checkpointWriteTriggerMode(enum ParamsIOFlag ioFlag ) {
   pvAssert(!mParams->presentAndNotBeenRead(mName, "checkpointWrite"));
   if (mCheckpointWriteFlag) {
      parameters()->ioParamString(ioFlag, mName, "checkpointWriteTriggerMode", &mCheckpointWriteTriggerModeString, "step");
      if (ioFlag==PARAMS_IO_READ) {
         pvAssert(mCheckpointWriteTriggerModeString);
         if (!strcmp(mCheckpointWriteTriggerModeString, "step") || !strcmp(mCheckpointWriteTriggerModeString, "Step") || !strcmp(mCheckpointWriteTriggerModeString, "STEP")) {
            mCheckpointWriteTriggerMode = CPWRITE_TRIGGER_STEP;
         }
         else if (!strcmp(mCheckpointWriteTriggerModeString, "time") || !strcmp(mCheckpointWriteTriggerModeString, "Time") || !strcmp(mCheckpointWriteTriggerModeString, "TIME")) {
            mCheckpointWriteTriggerMode = CPWRITE_TRIGGER_TIME;
         }
         else if (!strcmp(mCheckpointWriteTriggerModeString, "clock") || !strcmp(mCheckpointWriteTriggerModeString, "Clock") || !strcmp(mCheckpointWriteTriggerModeString, "CLOCK")) {
            mCheckpointWriteTriggerMode = CPWRITE_TRIGGER_CLOCK;
         }
         else {
            if (globalRank()==0) {
               pvErrorNoExit().printf("HyPerCol \"%s\": checkpointWriteTriggerMode \"%s\" is not recognized.\n", mName, mCheckpointWriteTriggerModeString);
            }
            MPI_Barrier(getCommunicator()->globalCommunicator());
            exit(EXIT_FAILURE);
         }
      }
   }
}

void HyPerCol::ioParam_checkpointWriteStepInterval(enum ParamsIOFlag ioFlag) {
   assert(!mParams->presentAndNotBeenRead(mName, "checkpointWrite"));
   if (mCheckpointWriteFlag) {
      pvAssert(!mParams->presentAndNotBeenRead(mName, "checkpointWriteTriggerMode"));
      if(mCheckpointWriteTriggerMode == CPWRITE_TRIGGER_STEP) {
         parameters()->ioParamValue(ioFlag, mName, "checkpointWriteStepInterval", &mCpWriteStepInterval, 1L);
      }
   }
}

void HyPerCol::ioParam_checkpointWriteTimeInterval(enum ParamsIOFlag ioFlag) {
   assert(!mParams->presentAndNotBeenRead(mName, "checkpointWrite"));
   if (mCheckpointWriteFlag) {
      pvAssert(!mParams->presentAndNotBeenRead(mName, "checkpointWriteTriggerMode"));
      if(mCheckpointWriteTriggerMode == CPWRITE_TRIGGER_TIME) {
         parameters()->ioParamValue(ioFlag, mName, "checkpointWriteTimeInterval", &mCpWriteTimeInterval, mDeltaTime);
      }
   }
}

void HyPerCol::ioParam_checkpointWriteClockInterval(enum ParamsIOFlag ioFlag) {
   assert(!mParams->presentAndNotBeenRead(mName, "checkpointWrite"));
   if (mCheckpointWriteFlag) {
      pvAssert(!mParams->presentAndNotBeenRead(mName, "checkpointWriteTriggerMode"));
      if(mCheckpointWriteTriggerMode == CPWRITE_TRIGGER_CLOCK) {
         parameters()->ioParamValueRequired(ioFlag, mName, "checkpointWriteClockInterval", &mCpWriteClockInterval);
      }
   }
}

void HyPerCol::ioParam_checkpointWriteClockUnit(enum ParamsIOFlag ioFlag) {
   pvAssert(!mParams->presentAndNotBeenRead(mName, "checkpointWrite"));
   if (mCheckpointWriteFlag) {
      pvAssert(!mParams->presentAndNotBeenRead(mName, "checkpointWriteTriggerMode"));
      if(mCheckpointWriteTriggerMode == CPWRITE_TRIGGER_CLOCK) {
         assert(!mParams->presentAndNotBeenRead(mName, "checkpointWriteTriggerClockInterval"));
         parameters()->ioParamString(ioFlag, mName, "checkpointWriteClockUnit", &mCheckpointWriteClockUnit, "seconds");
         if (ioFlag==PARAMS_IO_READ) {
            pvAssert(mCheckpointWriteClockUnit);
            for (size_t n=0; n<strlen(mCheckpointWriteClockUnit); n++) {
               mCheckpointWriteClockUnit[n] = tolower(mCheckpointWriteClockUnit[n]);
            }
            if (!strcmp(mCheckpointWriteClockUnit, "second") || !strcmp(mCheckpointWriteClockUnit, "seconds") || !strcmp(mCheckpointWriteClockUnit, "sec") || !strcmp(mCheckpointWriteClockUnit, "s")) {
               free(mCheckpointWriteClockUnit);
               mCheckpointWriteClockUnit=strdup("seconds");
               mCpWriteClockSeconds = (time_t) mCpWriteClockInterval;
            }
            else if (!strcmp(mCheckpointWriteClockUnit, "minute") || !strcmp(mCheckpointWriteClockUnit, "minutes") || !strcmp(mCheckpointWriteClockUnit, "min") || !strcmp(mCheckpointWriteClockUnit, "m")) {
               free(mCheckpointWriteClockUnit);
               mCheckpointWriteClockUnit=strdup("minutes");
               mCpWriteClockSeconds = (time_t) (60.0 * mCpWriteTimeInterval);
            }
            else if (!strcmp(mCheckpointWriteClockUnit, "hour") || !strcmp(mCheckpointWriteClockUnit, "hours") || !strcmp(mCheckpointWriteClockUnit, "hr") || !strcmp(mCheckpointWriteClockUnit, "h")) {
               free(mCheckpointWriteClockUnit);
               mCheckpointWriteClockUnit=strdup("hours");
               mCpWriteClockSeconds = (time_t) (3600.0 * mCpWriteTimeInterval);
            }
            else if (!strcmp(mCheckpointWriteClockUnit, "day") || !strcmp(mCheckpointWriteClockUnit, "days")) {
               free(mCheckpointWriteClockUnit);
               mCheckpointWriteClockUnit=strdup("days");
               mCpWriteClockSeconds = (time_t) (86400.0 * mCpWriteTimeInterval);
            }
            else {
               if (globalRank()==0) {
                  pvErrorNoExit().printf("checkpointWriteClockUnit \"%s\" is unrecognized.  Use \"seconds\", \"minutes\", \"hours\", or \"days\".\n", mCheckpointWriteClockUnit);
               }
               MPI_Barrier(getCommunicator()->globalCommunicator());
               exit(EXIT_FAILURE);
            }
            pvErrorIf(mCheckpointWriteClockUnit == nullptr, "Error in global rank %d process converting checkpointWriteClockUnit: %s\n", globalRank(), strerror(errno));
         }
      }
   }
}

void HyPerCol::ioParam_deleteOlderCheckpoints(enum ParamsIOFlag ioFlag) {
   assert(!mParams->presentAndNotBeenRead(mName, "checkpointWrite"));
   if (mCheckpointWriteFlag) {
      parameters()->ioParamValue(ioFlag, mName, "deleteOlderCheckpoints", &mDeleteOlderCheckpoints, false/*default value*/);
   }
}

void HyPerCol::ioParam_numCheckpointsKept(enum ParamsIOFlag ioFlag) {
   pvAssert(!mParams->presentAndNotBeenRead(mName, "checkpointWrite"));
   if (mCheckpointWriteFlag) {
      pvAssert(!mParams->presentAndNotBeenRead(mName, "deleteOlderCheckpoints"));
      if (mDeleteOlderCheckpoints) {
         parameters()->ioParamValue(ioFlag, mName, "numCheckpointsKept", &mNumCheckpointsKept, 1);
         if (ioFlag==PARAMS_IO_READ && mNumCheckpointsKept <= 0) {
            if (columnId()==0) {
               pvErrorNoExit() << "HyPerCol \"" << mName << "\": numCheckpointsKept must be positive (value was " << mNumCheckpointsKept << ")" << std::endl;
            }
            MPI_Barrier(mCommunicator->communicator());
            exit(EXIT_FAILURE);
         }
      }
   }
}

void HyPerCol::ioParam_suppressLastOutput(enum ParamsIOFlag ioFlag) {
   assert(!mParams->presentAndNotBeenRead(mName, "checkpointWrite"));
   if (!mCheckpointWriteFlag) {
      parameters()->ioParamValue(ioFlag, mName, "suppressLastOutput", &mSuppressLastOutput, false/*default value*/);
   }
}

void HyPerCol::ioParam_suppressNonplasticCheckpoints(enum ParamsIOFlag ioFlag) {
   assert(!mParams->presentAndNotBeenRead(mName, "checkpointWrite"));
   if (mCheckpointWriteFlag) {
      parameters()->ioParamValue(ioFlag, mName, "suppressNonplasticCheckpoints", &mSuppressNonplasticCheckpoints, mSuppressNonplasticCheckpoints);
   }
}

void HyPerCol::ioParam_checkpointIndexWidth(enum ParamsIOFlag ioFlag) {
   assert(!mParams->presentAndNotBeenRead(mName, "checkpointWrite"));
   if (mCheckpointWriteFlag) {
      parameters()->ioParamValue(ioFlag, mName, "checkpointIndexWidth", &mCheckpointIndexWidth, mCheckpointIndexWidth);
   }
}

void HyPerCol::ioParam_errorOnNotANumber(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, mName, "errorOnNotANumber", &mErrorOnNotANumber, mErrorOnNotANumber);
}

// Sep 26, 2016: HyPerCol methods for parameter input/output have been moved to PVParams.
// Sep 27, 2016: ensureDirExists has been moved to fileio.cpp.

int HyPerCol::addLayer(HyPerLayer * layer)
{
   addObject(layer);
   mLayers.push_back(layer);
   if(layer->getPhase() >= mNumPhases) mNumPhases = layer->getPhase() + 1;
   return mLayers.size() - 1;
}

int HyPerCol::addConnection(BaseConnection * conn)
{
   addObject(conn);
   mConnections.push_back(conn);
   return mConnections.size() - 1;
}

int HyPerCol::addNormalizer(NormalizeBase * normalizer) {
   mNormalizers.push_back(normalizer);
   return PV_SUCCESS; //Why does this return success when the other add functions return an index?
}

  // typically called by buildandrun via HyPerCol::run()
int HyPerCol::run(double start_time, double stop_time, double dt)
{
   mStartTime = start_time;
   mStopTime = stop_time;
   mDeltaTime = dt;

   int status = PV_SUCCESS;
   if (!mReadyFlag) {
      pvAssert(mPrintParamsFilename && mPrintParamsFilename[0]);
      std::string printParamsFileString("");
      if (mPrintParamsFilename[0] != '/') {
         printParamsFileString += mOutputPath;
         printParamsFileString += "/";
      }
      printParamsFileString += mPrintParamsFilename;

      setNumThreads(false);
      //When we call processParams, the communicateInitInfo stage will run, which can put out a lot of messages.
      //So if there's a problem with the -t option setting, the error message can be hard to find.
      //Instead of printing the error messages here, we will call setNumThreads a second time after
      //processParams(), and only then print messages.

      // processParams function does communicateInitInfo stage, sets up adaptive time step, and prints params
      status = processParams(printParamsFileString.c_str());
      MPI_Barrier(getCommunicator()->communicator());
      
      pvErrorIf(status != PV_SUCCESS,  "HyPerCol \"%s\" failed to run.\n", mName); 
      if (mPVInitObj->getDryRunFlag()) { return PV_SUCCESS; }
      int thread_status = setNumThreads(true/*now, print messages related to setting number of threads*/);
      MPI_Barrier(mCommunicator->globalCommunicator());
      if (thread_status !=PV_SUCCESS) {
         exit(EXIT_FAILURE);
      }

#ifdef PV_USE_OPENMP_THREADS
      pvAssert(mNumThreads > 0); // setNumThreads should fail if it sets mNumThreads less than or equal to zero
      omp_set_num_threads(mNumThreads);
#endif // PV_USE_OPENMP_THREADS

      // initDtAdaptControlProbe(); // Handling adaptive timesteps moved to AdaptiveTimeScaleProbe Aug 18, 2016.

      notify(std::make_shared<AllocateDataMessage>());

      mPhaseRecvTimers.clear();
      for(int phase = 0; phase < mNumPhases; phase++) {
         char tmpStr[10];
         sprintf(tmpStr, "phRecv%d", phase);
         mPhaseRecvTimers.push_back(new Timer(mName, "column", tmpStr));
      }

   #ifdef DEBUG_OUTPUT
      if (columnId() == 0) {
         pvInfo().printf("[0]: HyPerCol: running...\n");
         pvInfo().flush();
      }
   #endif

      // Initialize either by loading from checkpoint, or calling initializeState
      // This needs to happen after initPublishers so that we can initialize the values in the data stores,
      // and before the mLayers' publish calls so that the data in border regions gets copied correctly.
      if ( mCheckpointReadFlag ) {
         checkpointRead();
      }

      notify(std::make_shared<InitializeStateMessage>());

      // Initial normalization moved here to facilitate normalizations of groups of HyPerConns
      normalizeWeights();
      notify(std::make_shared<ConnectionFinalizeUpdateMessage>(mSimTime, mDeltaTime));

      // publish initial conditions
      for(int phase = 0; phase < mNumPhases; phase++){
         notify(std::make_shared<LayerPublishMessage>(phase, mSimTime));
      }

      // wait for all published data to arrive and update active indices;
      for (int phase=0; phase<mNumPhases; phase++) {
         notify(std::make_shared<LayerUpdateActiveIndicesMessage>(phase));
      }

      // output initial conditions
      if (!mCheckpointReadFlag) {
         notify(std::make_shared<ConnectionOutputMessage>(mSimTime));
         for (int phase=0; phase<mNumPhases; phase++) {
            notify(std::make_shared<LayerOutputStateMessage>(phase, mSimTime));
         }
      }
      mReadyFlag = true;
   }

#ifdef TIMER_ON
   Clock runClock;
   runClock.start_clock();
#endif
   // time loop
   //
   long int step = 0;
   pvAssert(status == PV_SUCCESS);
   while (mSimTime < mStopTime - mDeltaTime/2.0) {
      // Should we move the if statement below into advanceTime()?
      // That way, the routine that polls for SIGUSR1 and sets mCheckpointSignal is the same
      // as the routine that acts on mCheckpointSignal and clears it, which seems clearer.  --pete July 7, 2015
      if( mCheckpointWriteFlag && (advanceCPWriteTime() || mCheckpointSignal) ) {
         // the order should be advanceCPWriteTime() || mCheckpointSignal so that advanceCPWriteTime() is called even if mCheckpointSignal is true.
         // that way advanceCPWriteTime's calculation of the next checkpoint time won't be thrown off.
         char cpDir[PV_PATH_MAX];
         int stepFieldWidth;
         if (mCheckpointIndexWidth >= 0) { stepFieldWidth = mCheckpointIndexWidth; }
         else { stepFieldWidth = (int) floor(log10((mStopTime - mStartTime)/mDeltaTime))+1; }
         int chars_printed = snprintf(cpDir, PV_PATH_MAX, "%s/Checkpoint%0*ld", mCheckpointWriteDir, stepFieldWidth, mCurrentStep);
         pvErrorIf(chars_printed >= PV_PATH_MAX && globalRank()==0, "HyPerCol::run error.  Checkpoint directory \"%s/Checkpoint%ld\" is too long.\n", mCheckpointWriteDir, mCurrentStep);
         if ( !mCheckpointReadFlag || strcmp(mCheckpointReadDir, cpDir) ) {
            /* Note: the strcmp isn't perfect, since there are multiple ways to specify a path that points to the same directory */
            if (globalRank()==0) {
               pvInfo().printf("Checkpointing, simTime = %f\n", simulationTime());
            }
            checkpointWrite(cpDir);
         }
         else {
            if (globalRank()==0) {
               pvInfo().printf("Skipping checkpoint at time %f, since this would clobber the checkpointRead checkpoint.\n", simulationTime());
            }
         }
         if (mCheckpointSignal) {
            pvInfo().printf("Global rank %d: checkpointing in response to SIGUSR1.\n", globalRank());
            mCheckpointSignal = 0;
         }
      }
      status = advanceTime(mSimTime);

      step += 1;
#ifdef TIMER_ON
      if (step == 10) {
         runClock.start_clock(); 
      }
#endif

   }  // end time loop

#ifdef DEBUG_OUTPUT
   if (columnId() == 0) {
      pvInfo().printf("[0]: HyPerCol::run done...\n");
      pvInfo().flush();
   }
#endif

   const bool exitOnFinish = false;
   exitRunLoop(exitOnFinish);

#ifdef TIMER_ON
   runClock.stop_clock();
   runClock.print_elapsed(getOutputStream());
#endif

   return status;
}

// Sep 26, 2016: Adaptive timestep routines and member variables have been moved to AdaptiveTimeScaleProbe.

// This routine sets the mNumThreads member variable.  It should only be called by the run() method,
// and only inside the !ready if-statement.
//TODO: Instead of using the printMessagesFlag, why not use the same flag that 
int HyPerCol::setNumThreads(bool printMessagesFlag) {
   bool printMsgs0 = printMessagesFlag && globalRank()==0;
   int thread_status = PV_SUCCESS;
   int num_threads = 0;
#ifdef PV_USE_OPENMP_THREADS
   int max_threads = mPVInitObj->getMaxThreads();
   int comm_size = mCommunicator->globalCommSize();
   if (printMsgs0) {
      pvInfo().printf("Maximum number of OpenMP threads%s is %d\nNumber of MPI processes is %d.\n",
            comm_size==1 ? "" : " (over all processes)", max_threads, comm_size);
   }
   if (mPVInitObj->getUseDefaultNumThreads()) {
      num_threads = max_threads/comm_size; // integer arithmetic
      if (num_threads == 0) {
         num_threads = 1;
         if (printMsgs0) {
            pvWarn().printf("Warning: more MPI processes than available threads.  Processors may be oversubscribed.\n");
         }
      }
   }
   else {
      num_threads = mPVInitObj->getNumThreads();
   }
   if (num_threads>0) {
      if (printMsgs0) {
         pvInfo().printf("Number of threads used is %d\n", num_threads);
      }
   }
   else if (num_threads==0) {
      thread_status = PV_FAILURE;
      if (printMsgs0) {
         pvErrorNoExit().printf("%s: number of threads must be positive (was set to zero)\n", mPVInitObj->getProgramName());
      }
   }
   else {
      assert(num_threads<0);
      thread_status = PV_FAILURE;
      if (printMsgs0) {
         pvErrorNoExit().printf("%s was compiled with PV_USE_OPENMP_THREADS; therefore the \"-t\" argument is required.\n", mPVInitObj->getProgramName());
      }
   }
#else // PV_USE_OPENMP_THREADS
   if (mPVInitObj->getUseDefaultNumThreads()) {
      num_threads = 1;
      if (printMsgs0) {
         pvInfo().printf("Number of threads used is 1 (Compiled without OpenMP.\n");
      }
   }
   else {
      num_threads = mPVInitObj->getNumThreads();
      if (num_threads < 0) {
         num_threads = 1; 
      }
      if (num_threads != 1) {
         thread_status = PV_FAILURE; 
      }
   }
   if (printMsgs0) {
      if (thread_status!=PV_SUCCESS) {
         pvErrorNoExit().printf("%s error: PetaVision must be compiled with OpenMP to run with threads.\n", mPVInitObj->getProgramName());
      }
   }
#endif // PV_USE_OPENMP_THREADS
   mNumThreads = num_threads;
   return thread_status;
}

int HyPerCol::processParams(char const * path) {
   if (!mParamsProcessedFlag) {
      auto const& objectMap = mObjectHierarchy.getObjectMap();
      notify(std::make_shared<CommunicateInitInfoMessage>(objectMap));
   }

   // Print a cleaned up version of params to the file given by printParamsFilename
   parameters()->warnUnread();
   std::string printParamsPath = "";
   if (path!=nullptr && path[0] != '\0') {
      outputParams(path);
   }
   else {
      if (globalRank()==0) {
         pvInfo().printf("HyPerCol \"%s\": path for printing parameters file was empty or null.\n", mName);
      }
   }
   mParamsProcessedFlag = true;
   return PV_SUCCESS;
}

int HyPerCol::normalizeWeights() {
   int status = PV_SUCCESS;
   for (int n = 0; n < mNormalizers.size(); n++) {
      NormalizeBase * normalizer = mNormalizers.at(n);
      if (normalizer) { status = normalizer->normalizeWeightsWrapper(); }
      if (status != PV_SUCCESS) {
         pvErrorNoExit().printf("Normalizer \"%s\" failed.\n", mNormalizers[n]->getName());
      }
   }
   return status;
}

// Sep 26, 2016: Adaptive timestep routines and member variables have been moved to AdaptiveTimeScaleProbe.

int HyPerCol::advanceTime(double sim_time)
{
   if (mSimTime >= mNextProgressTime) {
      mNextProgressTime += mProgressInterval;
      if (columnId() == 0) {
         std::ostream& progressStream = mWriteProgressToErr ? getErrorStream() : getOutputStream();
         time_t current_time;
         time(&current_time);
         progressStream << "   time==" << sim_time << "  " << ctime(&current_time); // ctime outputs an newline
      }
   }

   mRunTimer->start();

   // make sure mSimTime is updated even if HyPerCol isn't running time loop
   // triggerOffset might fail if mSimTime does not advance uniformly because
   // mSimTime could skip over tigger event
   // !!!TODO: fix trigger layer to compute mTimeScale so as not to allow bypassing trigger event
   mSimTime = sim_time + mDeltaTime;
   mCurrentStep++;

   notify(std::make_shared<AdaptTimestepMessage>());
   // Sep 26, 2016: Adaptive timestep routines and member variables have been moved to AdaptiveTimeScaleProbe.

   // At this point all activity from the previous time step has
   // been delivered to the data store.
   //

   int status = PV_SUCCESS;

   // update the connections (weights)
   //
   notify(std::make_shared<ConnectionUpdateMessage>(mSimTime, mDeltaTime));
   normalizeWeights();
   notify(std::make_shared<ConnectionFinalizeUpdateMessage>(mSimTime, mDeltaTime));
   notify(std::make_shared<ConnectionOutputMessage>(mSimTime));

   if (globalRank()==0) {
      int sigstatus = PV_SUCCESS;
      sigset_t pollusr1;

      sigstatus = sigpending(&pollusr1); assert(sigstatus==0);
      mCheckpointSignal = sigismember(&pollusr1, SIGUSR1); assert(mCheckpointSignal==0 || mCheckpointSignal==1);
      if (mCheckpointSignal) {
         sigstatus = sigemptyset(&pollusr1); assert(sigstatus==0);
         sigstatus = sigaddset(&pollusr1, SIGUSR1); assert(sigstatus==0);
         int result=0;
         sigwait(&pollusr1, &result);
         assert(result==SIGUSR1);
      }
   }
   // Balancing MPI_Recv is after the for-loop over phases.  Is this better than MPI_Bcast?  Should it be MPI_Isend?
   if (globalRank()==0) {
      for (int k=1; k<numberOfGlobalColumns(); k++) {
         MPI_Send(&mCheckpointSignal, 1/*count*/, MPI_INT, k/*destination*/, 99/*tag*/, mCommunicator->globalCommunicator());
      }
   }

   // Each layer's phase establishes a priority for updating
   for (int phase=0; phase<mNumPhases; phase++) {

      //Ordering needs to go recvGpu, if(recvGpu and upGpu)update, recvNoGpu, update rest
#ifdef PV_USE_CUDA
      notify({
         std::make_shared<LayerRecvSynapticInputMessage>(phase, mPhaseRecvTimers.at(phase), true/*recvGpuFlag*/, mSimTime, mDeltaTime),
         std::make_shared<LayerUpdateStateMessage>(phase, true/*recvGpuFlag*/, true/*updateGpuFlag*/, mSimTime, mDeltaTime)
      });

      notify({
         std::make_shared<LayerRecvSynapticInputMessage>(phase, mPhaseRecvTimers.at(phase), false/*recvGpuFlag*/, mSimTime, mDeltaTime),
         std::make_shared<LayerUpdateStateMessage>(phase, false/*recvGpuFlag*/, false/*updateGpuFlag*/, mSimTime, mDeltaTime)

      });

      getDevice()->syncDevice();

      //Update for receiving on cpu and updating on gpu
      notify(std::make_shared<LayerUpdateStateMessage>(phase, false/*recvOnGpuFlag*/, true/*updateOnGpuFlag*/, mSimTime, mDeltaTime));

      getDevice()->syncDevice();
      notify(std::make_shared<LayerCopyFromGpuMessage>(phase, mPhaseRecvTimers.at(phase)));

      //Update for gpu recv and non gpu update
      notify(std::make_shared<LayerUpdateStateMessage>(phase, true/*recvOnGpuFlag*/, false/*updateOnGpuFlag*/, mSimTime, mDeltaTime));
#else
      notify({
         std::make_shared<LayerRecvSynapticInputMessage>(phase, mPhaseRecvTimers.at(phase), mSimTime, mDeltaTime),
         std::make_shared<LayerUpdateStateMessage>(phase, mSimTime, mDeltaTime)
      });
#endif

      // Rotate DataStore ring buffers, copy activity buffer to DataStore, and do MPI exchange.
      notify(std::make_shared<LayerPublishMessage>(phase, mSimTime));

      // wait for all published data to arrive and call layer's outputState

      std::vector<std::shared_ptr<BaseMessage const> > messageVector = {
         std::make_shared<LayerUpdateActiveIndicesMessage>(phase),
         std::make_shared<LayerOutputStateMessage>(phase, mSimTime)
      };
      if (mErrorOnNotANumber) {
         messageVector.push_back(std::make_shared<LayerCheckNotANumberMessage>(phase));
      }
      notify(messageVector);
   }

   // Balancing MPI_Send is before the for-loop over phases.  Is this better than MPI_Bcast?
   if (globalRank()!=0) {
      MPI_Recv(&mCheckpointSignal, 1/*count*/, MPI_INT, 0/*source*/, 99/*tag*/, getCommunicator()->globalCommunicator(), MPI_STATUS_IGNORE);
   }

   mRunTimer->stop();

   outputState(mSimTime);


   return status;
}

bool HyPerCol::advanceCPWriteTime() {
   // returns true if nextCPWrite{Step,Time} has been advanced
   bool advanceCPTime;
   time_t now; // needed only by CPWRITE_TRIGGER_CLOCK, but can't declare variables inside a case
   switch (this->mCheckpointWriteTriggerMode) {
   case CPWRITE_TRIGGER_STEP:
      assert(mCpWriteStepInterval>0 && mCpWriteTimeInterval<0 && mCpWriteClockInterval<0.0);
      advanceCPTime = mCurrentStep >= mNextCpWriteStep;
      if( advanceCPTime ) {
         mNextCpWriteStep += mCpWriteStepInterval;
      }
      break;
   case CPWRITE_TRIGGER_TIME:
      assert(mCpWriteStepInterval<0 && mCpWriteTimeInterval>0 && mCpWriteClockInterval<0.0);
      advanceCPTime = mSimTime >= mNextCpWriteTime;
      if( advanceCPTime ) {
         mNextCpWriteTime += mCpWriteTimeInterval;
      }
      break;
   case CPWRITE_TRIGGER_CLOCK:
      assert(mCpWriteStepInterval<0 && mCpWriteTimeInterval<0 && mCpWriteClockInterval>0.0);
      now = time(nullptr);
      advanceCPTime = now >= mNextCpWriteClock;
      if (advanceCPTime) {
         if (globalRank()==0) {
            pvInfo().printf("Checkpoint triggered at %s", ctime(&now));
         }
         mNextCpWriteClock += mCpWriteClockSeconds;
         if (globalRank()==0) {
            pvInfo().printf("Next checkpoint trigger will be at %s", ctime(&mNextCpWriteClock));
         }
      }
      break;
   default:
      assert(0); // all possible cases are considered above.
      break;
   }
   return advanceCPTime;
}

int HyPerCol::checkpointRead() {
   struct timestamp_struct {
      double time; // time measured in units of dt
      long int step; // step number, usually time/dt
   };
   struct timestamp_struct timestamp;
   size_t timestamp_size = sizeof(struct timestamp_struct);
   assert(sizeof(struct timestamp_struct) == sizeof(long int) + sizeof(double));
   if( columnId()==0 ) {
      char timestamppath[PV_PATH_MAX];
      int chars_needed = snprintf(timestamppath, PV_PATH_MAX, "%s/timeinfo.bin", mCheckpointReadDir);
      if (chars_needed >= PV_PATH_MAX) {
         pvError().printf("HyPerCol::checkpointRead error: path \"%s/timeinfo.bin\" is too long.\n", mCheckpointReadDir);
      }
      PV_Stream * timestampfile = PV_fopen(timestamppath,"r",false/*mVerifyWrites*/);
      if (timestampfile == nullptr) {
         pvError().printf("HyPerCol::checkpointRead error: unable to open \"%s\" for reading.\n", timestamppath);
      }
      long int startpos = getPV_StreamFilepos(timestampfile);
      PV_fread(&timestamp,1,timestamp_size,timestampfile);
      long int endpos = getPV_StreamFilepos(timestampfile);
      assert(endpos-startpos==(int)timestamp_size);
      PV_fclose(timestampfile);
   }
   MPI_Bcast(&timestamp,(int) timestamp_size,MPI_CHAR,0,getCommunicator()->communicator());
   mSimTime = timestamp.time;
   mCurrentStep = timestamp.step;

   double t = mStartTime;
   for (long int k=mInitialStep; k<mCurrentStep; k++) {
      if (t >= mNextProgressTime) {
         mNextProgressTime += mProgressInterval;
      }
      t += mDeltaTime;
   }

   double checkTime = simulationTime();
   for (auto& p : mColProbes) {
      p->checkpointRead(mCheckpointReadDir, &checkTime);
   }
   // Sep 26, 2016: Adaptive timestep routines and member variables have been moved to AdaptiveTimeScaleProbe.

   if(mCheckpointWriteFlag) {
      char nextCheckpointPath[PV_PATH_MAX];
      int chars_needed;
      PV_Stream * nextCheckpointFile = nullptr;
      switch(mCheckpointWriteTriggerMode) {
      case CPWRITE_TRIGGER_STEP:
         readScalarFromFile(mCheckpointReadDir, mName, "nextCheckpointStep", &mNextCpWriteStep, mCurrentStep+mCpWriteStepInterval);
         break;
      case CPWRITE_TRIGGER_TIME:
         readScalarFromFile(mCheckpointReadDir, mName, "nextCheckpointTime", &mNextCpWriteTime, mSimTime+mCpWriteTimeInterval);
         break;
      case CPWRITE_TRIGGER_CLOCK:
         // Nothing to do in this case
         break;
      default:
         // All cases of mCheckpointWriteTriggerMode are handled above
         assert(0);
      }
   }
   return PV_SUCCESS;
}

int HyPerCol::writeTimers(std::ostream& stream){
   int rank=columnId();
   if (rank==0) {
      mRunTimer->fprint_time(stream);
      mCheckpointTimer->fprint_time(stream);
      for (auto c : mConnections) {
         c->writeTimers(stream);
      }
      for (int phase=0; phase < mPhaseRecvTimers.size(); phase++) {
         if(mPhaseRecvTimers.at(phase)) { mPhaseRecvTimers.at(phase)->fprint_time(stream); }
         for (int n = 0; n < mLayers.size(); n++) {
            if (mLayers.at(n) != nullptr) { //How would mLayers ever contain a null pointer?
               if(mLayers.at(n)->getPhase() != phase) continue;
               mLayers.at(n)->writeTimers(stream);
            }
         }
      }
   }
   return PV_SUCCESS;
}

int HyPerCol::checkpointWrite(const char * cpDir) {
   mCheckpointTimer->start();
   if (columnId()==0) {
      pvInfo().printf("Checkpointing to directory \"%s\" at simTime = %f\n", cpDir, mSimTime);
      struct stat timeinfostat;
      char timeinfofilename[PV_PATH_MAX];
      int chars_needed = snprintf(timeinfofilename, PV_PATH_MAX, "%s/timeinfo.bin", cpDir);
      if (chars_needed >= PV_PATH_MAX) {
         pvError().printf("HyPerCol::checkpointWrite error: path \"%s/timeinfo.bin\" is too long.\n", cpDir);
      }
      int statstatus = stat(timeinfofilename, &timeinfostat);
      if (statstatus == 0) {
         pvWarn().printf("Checkpoint directory \"%s\" has existing timeinfo.bin, which is now being deleted.\n", cpDir);
         int unlinkstatus = unlink(timeinfofilename);
         if (unlinkstatus != 0) {
            pvError().printf("Failure deleting \"%s\": %s\n", timeinfofilename, strerror(errno));
         }
      }
   }

   ensureDirExists(getCommunicator(), cpDir);
   for( int l=0; l<mLayers.size(); l++ ) {
      mLayers.at(l)->checkpointWrite(cpDir);
   }
   for( auto c : mConnections ) {
      if (c->getPlasticityFlag() || !mSuppressNonplasticCheckpoints) { c->checkpointWrite(cpDir); }
   }

   // Timers
   if (columnId()==0) {
      std::string timerpathstring = cpDir;
      timerpathstring += "/";
      timerpathstring += "timers.txt";

      //std::string timercsvstring = cpDir;
      //timercsvstring += "/";
      //timercsvstring += "timers.csv";

      const char * timerpath = timerpathstring.c_str();
      FileStream timerstream(timerpath, std::ios_base::out, getVerifyWrites());
      if (timerstream.outStream().fail()) {
         pvError().printf("Unable to open \"%s\" for checkpointing timer information: %s\n", timerpath, strerror(errno));
      }
      writeTimers(timerstream.outStream());

      // NOTE: If timercsvpath ever gets brought back to life, it needs to be converted to using ostreams instead of FILE*s.
      //const char * timercsvpath = timercsvstring.c_str();
      //PV_Stream * timercsvstream = PV_fopen(timercsvpath, "w", getVerifyWrites());
      //if (timercsvstream==nullptr) {
      //   pvError().printf("Unable to open \"%s\" for checkpointing timer information: %s\n", timercsvpath, strerror(errno));
      //}
      //writeCSV(timercsvstream->fp);
      //
      //PV_fclose(timercsvstream); timercsvstream = nullptr;
   }

   for (auto& p : mColProbes) {
      p->checkpointWrite(cpDir);
   }
   // Sep 26, 2016: Adaptive timestep routines and member variables have been moved to AdaptiveTimeScaleProbe.

   std::string checkpointedParamsFile = cpDir;
   checkpointedParamsFile += "/";
   checkpointedParamsFile += "pv.params";
   this->outputParams(checkpointedParamsFile.c_str());

   if (mCheckpointWriteFlag) {
      char nextCheckpointPath[PV_PATH_MAX];
      int chars_needed;
      PV_Stream * nextCheckpointFile = nullptr;
      switch(mCheckpointWriteTriggerMode) {
      case CPWRITE_TRIGGER_STEP:
         writeScalarToFile(cpDir, mName, "nextCheckpointStep", mNextCpWriteStep);
         break;
      case CPWRITE_TRIGGER_TIME:
         writeScalarToFile(cpDir, mName, "nextCheckpointTime", mNextCpWriteTime);
         break;
      case CPWRITE_TRIGGER_CLOCK:
         // Nothing to do in this case
         break;
      default:
         // All cases of mCheckpointWriteTriggerMode are handled above
         assert(0);
      }
   }


   // Note: timeinfo should be done at the end of the checkpointing, so that its presence serves as a flag that the checkpoint has completed.
   if( columnId()==0 ) {
      char timestamppath[PV_PATH_MAX];
      int chars_needed = snprintf(timestamppath, PV_PATH_MAX, "%s/timeinfo.bin", cpDir);
      assert(chars_needed < PV_PATH_MAX);
      PV_Stream * timestampfile = PV_fopen(timestamppath,"w", getVerifyWrites());
      assert(timestampfile);
      PV_fwrite(&mSimTime,1,sizeof(double),timestampfile);
      PV_fwrite(&mCurrentStep,1,sizeof(long int),timestampfile);
      PV_fclose(timestampfile);
      chars_needed = snprintf(timestamppath, PV_PATH_MAX, "%s/timeinfo.txt", cpDir);
      assert(chars_needed < PV_PATH_MAX);
      timestampfile = PV_fopen(timestamppath,"w", getVerifyWrites());
      assert(timestampfile);
      fprintf(timestampfile->fp,"time = %g\n", mSimTime);
      fprintf(timestampfile->fp,"timestep = %ld\n", mCurrentStep);
      PV_fclose(timestampfile);
   }


   if (mDeleteOlderCheckpoints) {
      pvAssert(mCheckpointWriteFlag); // checkpointWrite is called by exitRunLoop when mCheckpointWriteFlag is false; in this case mDeleteOlderCheckpoints should be false as well.
      char const * oldestCheckpointDir = mOldCheckpointDirectories[mOldCheckpointDirectoriesIndex].c_str();
      if (oldestCheckpointDir && oldestCheckpointDir[0]) {
         if (mCommunicator->commRank()==0) {
            struct stat lcp_stat;
            int statstatus = stat(oldestCheckpointDir, &lcp_stat);
            if ( statstatus!=0 || !(lcp_stat.st_mode & S_IFDIR) ) {
               if (statstatus==0) {
                  pvErrorNoExit().printf("Failed to delete older checkpoint: failed to stat \"%s\": %s.\n", oldestCheckpointDir, strerror(errno));
               }
               else {
                  pvErrorNoExit().printf("Deleting older checkpoint: \"%s\" exists but is not a directory.\n", oldestCheckpointDir);
               }
            }
            sync();
            std::string rmrf_string("");
            rmrf_string = rmrf_string + "rm -r '" + oldestCheckpointDir + "'";
            int rmrf_result = system(rmrf_string.c_str());
            if (rmrf_result != 0) {
               pvWarn().printf("unable to delete older checkpoint \"%s\": rm command returned %d\n",
                     oldestCheckpointDir, WEXITSTATUS(rmrf_result));
            }
         }
      }
      mOldCheckpointDirectories[mOldCheckpointDirectoriesIndex] = std::string(cpDir);
      mOldCheckpointDirectoriesIndex++;
      if (mOldCheckpointDirectoriesIndex==mNumCheckpointsKept) { mOldCheckpointDirectoriesIndex = 0; }
   }

   if (mCommunicator->commRank()==0) {
      pvInfo().printf("checkpointWrite complete. simTime = %f\n", mSimTime);
   }
   mCheckpointTimer->stop();
   return PV_SUCCESS;
}

int HyPerCol::outputParams(char const * path) {
   assert(path!=nullptr && path[0]!='\0');
   int status = PV_SUCCESS;
   int rank=mCommunicator->commRank();
   assert(mPrintParamsStream==nullptr);
   char printParamsPath[PV_PATH_MAX];
   char * tmp = strdup(path); // duplicate string since dirname() is allowed to modify its argument
   if (tmp==nullptr) {
      pvError().printf("HyPerCol::outputParams unable to allocate memory: %s\n", strerror(errno));
   }
   char * containingdir = dirname(tmp);
   status = ensureDirExists(getCommunicator(), containingdir); // must be called by all processes, even though only rank 0 creates the directory
   if (status != PV_SUCCESS) {
      pvErrorNoExit().printf("HyPerCol::outputParams unable to create directory \"%s\"\n", containingdir);
   }
   free(tmp);
   if(rank == 0){
      if( strlen(path)+4/*allow room for .lua at end, and string terminator*/ > (size_t) PV_PATH_MAX ) {
         pvWarn().printf("outputParams called with too long a filename.  Parameters will not be printed.\n");
         status = ENAMETOOLONG;
      }
      else {
         mPrintParamsStream = PV_fopen(path, "w", getVerifyWrites());
         if( mPrintParamsStream == nullptr ) {
            status = errno;
            pvErrorNoExit().printf("outputParams error opening \"%s\" for writing: %s\n", path, strerror(errno));
         }
         //Get new lua path
         char luapath [PV_PATH_MAX];
         strcpy(luapath, path);
         strcat(luapath, ".lua");
         mLuaPrintParamsStream = PV_fopen(luapath, "w", getVerifyWrites());
         if( mLuaPrintParamsStream == nullptr ) {
            status = errno;
            pvErrorNoExit().printf("outputParams failed to open \"%s\" for writing: %s\n", luapath, strerror(errno));
         }
      }
      assert(mPrintParamsStream != nullptr);
      assert(mLuaPrintParamsStream != nullptr);
      parameters()->setPrintParamsStream(mPrintParamsStream);
      parameters()->setPrintLuaStream(mLuaPrintParamsStream);

      //Params file output
      outputParamsHeadComments(mPrintParamsStream->fp, "//");

      //Lua file output
      outputParamsHeadComments(mLuaPrintParamsStream->fp, "--");
      //Load util module based on PVPath
      fprintf(mLuaPrintParamsStream->fp, "package.path = package.path .. \";\" .. \"" PV_DIR "/../parameterWrapper/?.lua\"\n");
      fprintf(mLuaPrintParamsStream->fp, "local pv = require \"PVModule\"\n\n");
      fprintf(mLuaPrintParamsStream->fp, "-- Base table variable to store\n");
      fprintf(mLuaPrintParamsStream->fp, "local pvParameters = {\n");
   }

   // Parent HyPerCol params
   status = ioParams(PARAMS_IO_WRITE);
   if( status != PV_SUCCESS ) {
      pvError().printf("outputParams: Error copying params to \"%s\"\n", printParamsPath);
   }

   // HyPerLayer params
   for (int l=0; l<mLayers.size(); l++) {
      HyPerLayer * layer = mLayers.at(l);
      status = layer->ioParams(PARAMS_IO_WRITE);
      if( status != PV_SUCCESS ) {
         pvError().printf("outputParams: Error copying params to \"%s\"\n", printParamsPath);
      }
   }

   // BaseConnection params
   for (auto c : mConnections) {
      status = c->ioParams(PARAMS_IO_WRITE);
      if( status != PV_SUCCESS ) {
         pvError().printf("outputParams: Error copying params to \"%s\"\n", printParamsPath);
      }
   }

   // Probe params

   // ColProbes
   for (int p=0; p<mColProbes.size(); p++) {
      mColProbes.at(p)->ioParams(PARAMS_IO_WRITE);
   }

   // LayerProbes
   for (int l=0; l<mLayers.size(); l++) {
      mLayers.at(l)->outputProbeParams();
   }

   // BaseConnectionProbes
   for (auto c : mConnections) {
      c->outputProbeParams();
   }

   if(rank == 0){
      fprintf(mLuaPrintParamsStream->fp, "} --End of pvParameters\n");
      fprintf(mLuaPrintParamsStream->fp, "\n-- Print out PetaVision approved parameter file to the console\n");
      fprintf(mLuaPrintParamsStream->fp, "paramsFileString = pv.createParamsFileString(pvParameters)\n");
      fprintf(mLuaPrintParamsStream->fp, "io.write(paramsFileString)\n");
   }

   if (mPrintParamsStream) {
      PV_fclose(mPrintParamsStream);
      mPrintParamsStream = nullptr;
      parameters()->setPrintParamsStream(mPrintParamsStream);
   }
   if (mLuaPrintParamsStream) {
      PV_fclose(mLuaPrintParamsStream);
      mLuaPrintParamsStream = nullptr;
      parameters()->setPrintLuaStream(mLuaPrintParamsStream);
   }
   return status;
}

int HyPerCol::outputParamsHeadComments(FILE* fp, char const * commentToken) {
   time_t t = time(nullptr);
   fprintf(fp, "%s PetaVision, " PV_REVISION "\n", commentToken);
   fprintf(fp, "%s Run time %s", commentToken, ctime(&t)); // newline is included in output of ctime
#ifdef PV_USE_MPI
   fprintf(fp, "%s Compiled with MPI and run using %d rows and %d columns.\n", commentToken, mCommunicator->numCommRows(), mCommunicator->numCommColumns());
#else // PV_USE_MPI
   fprintf(fp, "%s Compiled without MPI.\n", commentToken);
#endif // PV_USE_MPI
#ifdef PV_USE_CUDA
   fprintf(fp, "%s Compiled with CUDA.\n", commentToken);
#else
   fprintf(fp, "%s Compiled without CUDA.\n", commentToken);
#endif
#ifdef PV_USE_OPENMP_THREADS
   fprintf(fp, "%s Compiled with OpenMP parallel code", commentToken);
   if (mNumThreads>0) { fprintf(fp, " and run using %d threads.\n", mNumThreads); }
   else if (mNumThreads==0) { fprintf(fp, " but number of threads was set to zero (error).\n"); }
   else { fprintf(fp, " but the -t option was not specified.\n"); }
#else
   fprintf(fp, "%s Compiled without OpenMP parallel code", commentToken);
   if (mNumThreads==1) { fprintf(fp, ".\n"); }
   else if (mNumThreads==0) { fprintf(fp, " but number of threads was set to zero (error).\n"); }
   else { fprintf(fp, " but number of threads specified was %d instead of 1. (error).\n", mNumThreads); }
#endif // PV_USE_OPENMP_THREADS
   if (mCheckpointReadFlag) {
      fprintf(fp, "%s Started from checkpoint \"%s\"\n", commentToken, mCheckpointReadDir);
   }
   return PV_SUCCESS;
}

// Uses the arguments cpDir, objectName, and suffix to create a path of the form
// [cpDir]/[objectName][suffix]
// (the brackets are not in the created path, but the slash is)
// The string returned is allocated with malloc, and the calling routine is responsible for freeing the string.
char * HyPerCol::pathInCheckpoint(const char * cpDir, const char * objectName, const char * suffix) {
   assert(cpDir!=nullptr && suffix!=nullptr);
   size_t n = strlen(cpDir)+strlen("/")+strlen(objectName)+strlen(suffix)+(size_t) 1; // the +1 leaves room for the terminating null
   char * filename = (char *) malloc(n);
   if (filename==nullptr) {
      pvError().printf("Error: rank %d process unable to allocate filename \"%s/%s%s\": %s\n", columnId(), cpDir, objectName, suffix, strerror(errno));
   }
   int chars_needed = snprintf(filename, n, "%s/%s%s", cpDir, objectName, suffix);
   assert(chars_needed < n);
   return filename;
}

int HyPerCol::exitRunLoop(bool exitOnFinish)
{
   int status = 0;

   // output final state of layers and connections

   char cpDir[PV_PATH_MAX];
   if (mCheckpointWriteFlag || !mSuppressLastOutput) {
      int chars_printed;
      if (mCheckpointWriteFlag) {
         chars_printed = snprintf(cpDir, PV_PATH_MAX, "%s/Checkpoint%ld", mCheckpointWriteDir, mCurrentStep);
      }
      else {
         assert(!mSuppressLastOutput);
         chars_printed = snprintf(cpDir, PV_PATH_MAX, "%s/Last", mOutputPath);
      }
      if(chars_printed >= PV_PATH_MAX) {
         if (mCommunicator->commRank()==0) {
            pvError().printf("HyPerCol::run error.  Checkpoint directory \"%s/Checkpoint%ld\" is too long.\n", mCheckpointWriteDir, mCurrentStep);
         }
      }
      checkpointWrite(cpDir);
   }

   if (exitOnFinish) {
      delete this;
      exit(0);
   }

   return status;
}

int HyPerCol::getAutoGPUDevice(){
   int returnGpuIdx = -1;
#ifdef PV_USE_CUDA
   int mpiRank = mCommunicator->globalCommRank();
   int numMpi = mCommunicator->globalCommSize();
   char hostNameStr[PV_PATH_MAX];
   gethostname(hostNameStr, PV_PATH_MAX);
   size_t hostNameLen = strlen(hostNameStr) + 1; //+1 for null terminator

   //Each rank communicates which host it is on
   //Root process
   if(mpiRank == 0){
      //Allocate data structure for rank to host
      char rankToHost[numMpi][PV_PATH_MAX];
      assert(rankToHost);
      //Allocate data structure for rank to maxGpu
      int rankToMaxGpu[numMpi];
      //Allocate final data structure for rank to GPU index
      int rankToGpu[numMpi];
      assert(rankToGpu);

      for(int rank = 0; rank < numMpi; rank++){
         if(rank == 0){
            strcpy(rankToHost[rank], hostNameStr);
            rankToMaxGpu[rank] = PVCuda::CudaDevice::getNumDevices();
         }
         else{
            MPI_Recv(rankToHost[rank], PV_PATH_MAX, MPI_CHAR, rank, 0, mCommunicator->globalCommunicator(), MPI_STATUS_IGNORE);
            MPI_Recv(&(rankToMaxGpu[rank]), 1, MPI_INT, rank, 0, mCommunicator->globalCommunicator(), MPI_STATUS_IGNORE);
         }
      }

      //rankToHost now is an array such that the index is the rank, and the value is the host
      //Convert to a map of vectors, such that the key is the host name and the value
      //is a vector of mpi ranks that is running on that host
      std::map<std::string, std::vector<int> > hostMap;
      for(int rank = 0; rank < numMpi; rank++){
         hostMap[std::string(rankToHost[rank])].push_back(rank);
      }

      //Determine what gpus to use per mpi
      for (auto& host : hostMap) {
         std::vector<int> rankVec = host.second;
         int numRanksPerHost = rankVec.size();
         assert(numRanksPerHost > 0);
         //Grab maxGpus of current host
         int maxGpus = rankToMaxGpu[rankVec[0]];
         //Warnings for overloading/underloading gpus
         if(numRanksPerHost != maxGpus){
            pvWarn(assignGpuWarning);
            assignGpuWarning.printf("HyPerCol::getAutoGPUDevice: Host \"%s\" (rank[s] ", host.first.c_str());
            for(int v_i = 0; v_i < numRanksPerHost; v_i++){
               if(v_i != numRanksPerHost-1){
                  assignGpuWarning.printf("%d, ", rankVec[v_i]);
               }
               else{
                  assignGpuWarning.printf("%d", rankVec[v_i]);
               }
            }
            assignGpuWarning.printf(") is being %s, with %d mpi processes mapped to %d total GPU[s]\n",
                  numRanksPerHost < maxGpus ? "underloaded":"overloaded",
                  numRanksPerHost, maxGpus);
         }

         //Match a rank to a gpu
         for(int v_i = 0; v_i < numRanksPerHost; v_i++){
            rankToGpu[rankVec[v_i]] = v_i % maxGpus;
         }
      }

      //MPI sends to each process to specify which gpu the rank should use
      for(int rank = 0; rank < numMpi; rank++){
         pvInfo() << "Rank " << rank << " on host \"" << rankToHost[rank] << "\" (" << rankToMaxGpu[rank] << " GPU[s]) using GPU index " <<
            rankToGpu[rank] << "\n";
         if(rank == 0){
            returnGpuIdx = rankToGpu[rank];
         }
         else{
            MPI_Send(&(rankToGpu[rank]), 1, MPI_INT, rank, 0, mCommunicator->globalCommunicator());
         }
      }
   }
   //Non root process
   else{
      //Send host name
      MPI_Send(hostNameStr, hostNameLen, MPI_CHAR, 0, 0, mCommunicator->globalCommunicator());
      //Send max gpus for that host
      int maxGpu = PVCuda::CudaDevice::getNumDevices();
      MPI_Send(&maxGpu, 1, MPI_INT, 0, 0, mCommunicator->globalCommunicator());
      //Recv gpu idx
      MPI_Recv(&(returnGpuIdx), 1, MPI_INT, 0, 0, mCommunicator->globalCommunicator(), MPI_STATUS_IGNORE);
   }
   assert(returnGpuIdx >= 0 && returnGpuIdx < PVCuda::CudaDevice::getNumDevices());
#else
   //This function should never be called when not running with GPUs
   assert(false);
#endif
   return returnGpuIdx;
}

int HyPerCol::initializeThreads(char const * in_device)
{
   int numMpi = mCommunicator->globalCommSize();
   int device;

   //default value
   if(in_device == nullptr){
      pvInfo() << "Auto assigning GPUs\n";
      device = getAutoGPUDevice();
   }
   else{
      std::vector <int> deviceVec;
      std::stringstream ss(in_device);
      std::string stoken;
      //Grabs strings from ss into item, seperated by commas
      while(std::getline(ss, stoken, ',')){
         //Convert stoken to integer
         for(auto& ch : stoken) {
            if(!isdigit(ch)) {
               pvError().printf("Device specification error: %s contains unrecognized characters. Must be comma separated integers greater or equal to 0 with no other characters allowed (including spaces).\n", in_device);
            }
         }
         deviceVec.push_back(atoi(stoken.c_str()));
      }
      //Check length of deviceVec
      //Allowed cases are 1 device specified or greater than or equal to number of mpi processes devices specified
      if(deviceVec.size() == 1){
         device = deviceVec[0];
      }
      else if(deviceVec.size() >= numMpi){
         device = deviceVec[mCommunicator->globalCommRank()];
      }
      else{
         pvError().printf("Device specification error: Number of devices specified (%zu) must be either 1 or >= than number of mpi processes (%d).\n", deviceVec.size(), numMpi);
      }
      pvInfo() << "Global MPI Process " << mCommunicator->globalCommRank() << " using device " << device << "\n";
   }

#ifdef PV_USE_CUDA
   mCudaDevice = new PVCuda::CudaDevice(device);
#endif
   return 0;
}

#ifdef PV_USE_CUDA
int HyPerCol::finalizeThreads()
{
   delete mCudaDevice;
   //if(mGpuGroupConns){
   //   free(mGpuGroupConns);
   //}
   for(auto iterator = mGpuGroupConns.begin(); iterator != mGpuGroupConns.end();)
   {
      delete *iterator;
      iterator = mGpuGroupConns.erase(iterator);
   }
   return 0;
}

void HyPerCol::addGpuGroup(BaseConnection* conn, int gpuGroupIdx){
   //default gpuGroupIdx is -1, so do nothing if this is the case
   if(gpuGroupIdx < 0){
      return;
   }
   mGpuGroupConns.reserve(gpuGroupIdx);
   if(mGpuGroupConns.at(gpuGroupIdx) == nullptr) {
      mGpuGroupConns.at(gpuGroupIdx) = conn;
   }
   ////Resize buffer if not big enough
   //if(gpuGroupIdx >= mNumGpuGroup){
   //   int oldNumGpuGroup = mNumGpuGroup;
   //   mNumGpuGroup = gpuGroupIdx + 1;
   //   mGpuGroupConns = (BaseConnection**) realloc(mGpuGroupConns, mNumGpuGroup * sizeof(BaseConnection*));
   //   //Initialize newly allocated part to nullptr
   //   for(int i = oldNumGpuGroup; i < mNumGpuGroup; i++){
   //      mGpuGroupConns[i] = nullptr;
   //   }
   //}
   ////If empty, fill
   //if(mGpuGroupConns[gpuGroupIdx] == nullptr){
   //   mGpuGroupConns[gpuGroupIdx] = conn;
   //}
   ////Otherwise, do nothing
   //
   //return;
}
#endif //PV_USE_CUDA

int HyPerCol::insertProbe(ColProbe * p)
{
   mColProbes.push_back(p);
   return mColProbes.size(); //Other insert functions return the index of the inserted object. Is this correct here?
}

void HyPerCol::addObject(BaseObject * obj) {
   bool succeeded = mObjectHierarchy.addObject(obj->getName(), obj);
   if (!succeeded) {
      if (columnId()==0) {
          pvError() << "Adding " << obj->getDescription() << "failed.\n";
      }
      MPI_Barrier(getCommunicator()->communicator());
      exit(PV_FAILURE);
   }
}

// BaseProbes include layer probes, connection probes, and column probes.
int HyPerCol::addBaseProbe(BaseProbe * p) {
   addObject(p);
   mBaseProbes.push_back(p);
   return mBaseProbes.size();
}

int HyPerCol::outputState(double time)
{
   for( int n = 0; n < mColProbes.size(); n++ ) {
       mColProbes.at(n)->outputStateWrapper(time, mDeltaTime);
   }
   return PV_SUCCESS;
}


HyPerLayer * HyPerCol::getLayerFromName(const char * layerName) {
   if (layerName==nullptr) { return nullptr; }
   int n = numberOfLayers();
   for( int i=0; i<n; i++ ) {
      HyPerLayer * curLayer = getLayer(i);
      assert(curLayer);
      const char * curLayerName = curLayer->getName();
      assert(curLayerName);
      if( !strcmp( curLayer->getName(), layerName) ) return curLayer;
   }
   return nullptr;
}

BaseConnection * HyPerCol::getConnFromName(const char * connName) {
   if( connName == nullptr ) return nullptr;
   int n = numberOfConnections();
   for( int i=0; i<n; i++ ) {
      BaseConnection * curConn = getConnection(i);
      assert(curConn);
      const char * curConnName = curConn->getName();
      assert(curConnName);
      if( !strcmp( curConn->getName(), connName) ) return curConn;
   }
   return nullptr;
}

NormalizeBase * HyPerCol::getNormalizerFromName(const char * normalizerName) {
   if( normalizerName == nullptr ) return nullptr;
   int n = numberOfNormalizers();
   for( int i=0; i<n; i++ ) {
      NormalizeBase * curNormalizer = getNormalizer(i);
      assert(curNormalizer);
      const char * curNormalizerName = curNormalizer->getName();
      assert(curNormalizerName);
      if( !strcmp(curNormalizer->getName(), normalizerName) ) return curNormalizer;
   }
   return nullptr;
}

ColProbe * HyPerCol::getColProbeFromName(const char * probeName) {
   if (probeName == nullptr) return nullptr;
   ColProbe * p = nullptr;
   int n = numberOfProbes();
   for (int i=0; i<n; i++) {
      ColProbe * curColProbe = getColProbe(i);
      const char * curName = curColProbe->getName();
      assert(curName);
      if (!strcmp(curName, probeName)) {
         p = curColProbe;
         break;
      }
   }
   return p;
}

BaseProbe * HyPerCol::getBaseProbeFromName(const char * probeName) {
   if (probeName == nullptr) { return nullptr; }
   BaseProbe * p = nullptr;
   int n = numberOfBaseProbes();
   for (int i=0; i<n; i++) {
      BaseProbe * curBaseProbe = getBaseProbe(i);
      const char * curName = curBaseProbe->getName();
      assert(curName);
      if (!strcmp(curName, probeName)) {
         p = curBaseProbe;
         break;
      }
   }
   return p;
}

unsigned int HyPerCol::seedRandomFromWallClock() {
   unsigned long t = 0UL;
   int rootproc = 0;
   if (columnId()==rootproc) {
       t = time((time_t *) nullptr);
   }
   MPI_Bcast(&t, 1, MPI_UNSIGNED, rootproc, mCommunicator->communicator());
   return t;
}

template <typename T>
int HyPerCol::writeScalarToFile(const char * cp_dir, const char * group_name, const char * val_name, T val) {
   return writeArrayToFile(cp_dir, group_name, val_name, &val, 1);
}

// Declare the instantiations of writeScalarToFile that occur in other .cpp files; otherwise you'll get linker errors.
template int HyPerCol::writeScalarToFile<int>(char const * cpDir, const char * group_name, char const * val_name, int val);
template int HyPerCol::writeScalarToFile<long>(char const * cpDir, const char * group_name, char const * val_name, long val);
template int HyPerCol::writeScalarToFile<float>(char const * cpDir, const char * group_name, char const * val_name, float val);
template int HyPerCol::writeScalarToFile<double>(char const * cpDir, const char * group_name, char const * val_name, double val);

template <typename T>
int HyPerCol::writeArrayToFile(const char * cp_dir, const char * group_name, const char * val_name, T* val, size_t count) {
   int status = PV_SUCCESS;
   if (columnId()==0)  {
      char filename[PV_PATH_MAX];
      int chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_%s.bin", cp_dir, group_name, val_name);
      if (chars_needed >= PV_PATH_MAX) {
         pvError().printf("writeArrayToFile error: path %s/%s_%s.bin is too long.\n", cp_dir, group_name, val_name);
      }
      PV_Stream * pvstream = PV_fopen(filename, "w", getVerifyWrites());
      if (pvstream==nullptr) {
         pvError().printf("writeArrayToFile error: unable to open path %s for writing.\n", filename);
      }
      int num_written = PV_fwrite(val, sizeof(T), count, pvstream);
      if (num_written != count) {
         pvError().printf("writeArrayToFile error while writing to %s.\n", filename);
      }
      PV_fclose(pvstream);
      chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_%s.txt", cp_dir, group_name, val_name);
      assert(chars_needed < PV_PATH_MAX);
      std::ofstream fs;
      fs.open(filename);
      if (!fs) {
         pvError().printf("writeArrayToFile error: unable to open path %s for writing.\n", filename);
      }
      for(int i = 0; i < count; i++){
         fs << val[i];
         fs << std::endl; // Can write as fs << val << std::endl, but eclipse flags that as an error 'Invalid overload of std::endl'
      }
      fs.close();
   }
   return status;
}
// Declare the instantiations of writeArrayToFile that occur in other .cpp files; otherwise you'll get linker errors.
template int HyPerCol::writeArrayToFile<int>(char const * cpDir, const char * group_name, char const * val_name, int* val, size_t count);
template int HyPerCol::writeArrayToFile<long>(char const * cpDir, const char * group_name, char const * val_name, long* val, size_t count);
template int HyPerCol::writeArrayToFile<float>(char const * cpDir, const char * group_name, char const * val_name, float* val, size_t count);
template int HyPerCol::writeArrayToFile<double>(char const * cpDir, const char * group_name, char const * val_name, double* val, size_t count);

template <typename T>
int HyPerCol::readScalarFromFile(const char * cp_dir, const char * group_name, const char * val_name, T * val, T default_value) {
   return readArrayFromFile(cp_dir, group_name, val_name, val, 1, default_value);
}

// Declare the instantiations of readScalarToFile that occur in other .cpp files; otherwise you'll get linker errors.
template int HyPerCol::readScalarFromFile<int>(char const * cpDir, const char * group_name, char const * val_name, int * val, int default_value);
template int HyPerCol::readScalarFromFile<long>(char const * cpDir, const char * group_name, char const * val_name, long * val, long default_value);
template int HyPerCol::readScalarFromFile<float>(char const * cpDir, const char * group_name, char const * val_name, float * val, float default_value);
template int HyPerCol::readScalarFromFile<double>(char const * cpDir, const char * group_name, char const * val_name, double * val, double default_value);

template <typename T>
int HyPerCol::readArrayFromFile(const char * cp_dir, const char * group_name, const char * val_name, T * val, size_t count, T default_value) {
   int status = PV_SUCCESS;
   if( columnId() == 0 ) {
      char filename[PV_PATH_MAX];
      int chars_needed;
      chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_%s.bin", cp_dir, group_name, val_name); // Could use pathInCheckpoint if not for the .bin
      if(chars_needed >= PV_PATH_MAX) {
         pvError().printf("HyPerLayer::readArrayFloat error: path %s/%s_%s.bin is too long.\n", cp_dir, group_name, val_name);
      }
      PV_Stream * pvstream = PV_fopen(filename, "r", getVerifyWrites());
      for(int i = 0; i < count; i++){
         val[i] = default_value;
      }
      if (pvstream==nullptr) {
         pvWarn() << "readArrayFromFile: unable to open path \"" << filename << "\" for reading.  Value used will be " << *val << std::endl;
      }
      else {
         int num_read = PV_fread(val, sizeof(T), count, pvstream);
         if (num_read != count) {
            pvWarn() << "readArrayFromFile: unable to read from \"" << filename << "\".  Value used will be " << *val << std::endl;
         }
         PV_fclose(pvstream);
      }
   }
   MPI_Bcast(val, sizeof(T)*count, MPI_CHAR, 0, getCommunicator()->communicator());

   return status;
}
// Declare the instantiations of readArrayToFile that occur in other .cpp files; otherwise you'll get linker errors.
template int HyPerCol::readArrayFromFile<int>(char const * cpDir, const char * group_name, char const * val_name, int * val, size_t count, int default_value);
template int HyPerCol::readArrayFromFile<long>(char const * cpDir, const char * group_name, char const * val_name, long * val, size_t count, long default_value);
template int HyPerCol::readArrayFromFile<float>(char const * cpDir, const char * group_name, char const * val_name, float * val, size_t count, float default_value);
template int HyPerCol::readArrayFromFile<double>(char const * cpDir, const char * group_name, char const * val_name, double * val, size_t count, double default_value);

HyPerCol * createHyPerCol(PV_Init * pv_initObj) {
   PVParams * params = pv_initObj->getParams();
   if (params==nullptr) {
      pvErrorNoExit() << "createHyPerCol called without having set params.\n";
      return nullptr;
   }
   int numGroups = params->numberOfGroups();
   if (numGroups==0) {
      pvErrorNoExit() << "Params \"" << pv_initObj->getParamsFile() << "\" does not define any groups.\n";
      return nullptr;
   }
   if( strcmp(params->groupKeywordFromIndex(0), "HyPerCol") ) {
      pvErrorNoExit() << "First group in the params \"" << pv_initObj->getParamsFile() << "\" does not define a HyPerCol.\n";
      return nullptr;
   }
   char const * colName = params->groupNameFromIndex(0);

   HyPerCol * hc = new HyPerCol(colName, pv_initObj);
   for (int k=0; k<numGroups; k++) {
      const char * kw = params->groupKeywordFromIndex(k);
      const char * name = params->groupNameFromIndex(k);
      if (!strcmp(kw, "HyPerCol")) {
         if (k==0) { continue; }
         else {
            if (hc->columnId()==0) {
               pvErrorNoExit() << "Group " << k+1 << " in params file (\"" << pv_initObj->getParamsFile() << "\") is a HyPerCol; only the first group can be a HyPercol.\n";
            }
            delete hc;
            return nullptr;
         }
      }
      else {
         BaseObject * addedObject = Factory::instance()->createByKeyword(kw, name, hc);
         if (addedObject==nullptr) {
            if (hc->globalRank()==0) {
               pvErrorNoExit().printf("Unable to create %s \"%s\".\n", kw, name);
            }
            delete hc;
            return nullptr;
         }
      }
   }

   return hc;
}

} // PV namespace

