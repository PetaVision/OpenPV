/*
 * Checkpointer.cpp
 *
 *  Created on Sep 28, 2016
 *      Author: Pete Schultz
 */

#include "Checkpointer.hpp"
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstring>
#include <fts.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace PV {

Checkpointer::Checkpointer(std::string const &name, Communicator *comm)
      : mName(name), mCommunicator(comm) {
   initialize();
}

Checkpointer::~Checkpointer() {
   free(mCheckpointWriteDir);
   free(mCheckpointWriteTriggerModeString);
   free(mCheckpointWriteWallclockUnit);
   free(mLastCheckpointDir);
   free(mInitializeFromCheckpointDir);
   delete mCheckpointTimer;
}

void Checkpointer::initialize() {
   mTimeInfoCheckpointEntry = std::make_shared<CheckpointEntryData<Checkpointer::TimeInfo>>(
         std::string("timeinfo"), getCommunicator(), &mTimeInfo, (size_t)1, true /*broadcast*/);
   // This doesn't get put into mCheckpointRegistry because we handle the timeinfo separately.
   mCheckpointTimer = new Timer(mName.c_str(), "column", "checkpoint");
   registerTimer(mCheckpointTimer);
}

void Checkpointer::ioParamsFillGroup(enum ParamsIOFlag ioFlag, PVParams *params) {
   ioParam_checkpointWrite(ioFlag, params);
   ioParam_checkpointWriteDir(ioFlag, params);
   ioParam_checkpointWriteTriggerMode(ioFlag, params);
   ioParam_checkpointWriteStepInterval(ioFlag, params);
   ioParam_checkpointWriteTimeInterval(ioFlag, params);
   ioParam_checkpointWriteClockInterval(ioFlag, params);
   ioParam_checkpointWriteClockUnit(ioFlag, params);
   ioParam_checkpointIndexWidth(ioFlag, params);
   ioParam_suppressNonplasticCheckpoints(ioFlag, params);
   ioParam_deleteOlderCheckpoints(ioFlag, params);
   ioParam_numCheckpointsKept(ioFlag, params);
   ioParam_lastCheckpointDir(ioFlag, params);
   ioParam_initializeFromCheckpointDir(ioFlag, params);
   ioParam_defaultInitializeFromCheckpointFlag(ioFlag, params);
}

void Checkpointer::ioParam_checkpointWrite(enum ParamsIOFlag ioFlag, PVParams *params) {
   params->ioParamValue(
         ioFlag, mName.c_str(), "checkpointWrite", &mCheckpointWriteFlag, mCheckpointWriteFlag);
}

void Checkpointer::ioParam_checkpointWriteDir(enum ParamsIOFlag ioFlag, PVParams *params) {
   pvAssert(!params->presentAndNotBeenRead(mName.c_str(), "checkpointWrite"));
   if (mCheckpointWriteFlag) {
      params->ioParamStringRequired(
            ioFlag, mName.c_str(), "checkpointWriteDir", &mCheckpointWriteDir);
      if (ioFlag == PARAMS_IO_READ) {
         ensureDirExists(getCommunicator(), mCheckpointWriteDir);
      }
   }
}

void Checkpointer::ioParam_checkpointWriteTriggerMode(enum ParamsIOFlag ioFlag, PVParams *params) {
   pvAssert(!params->presentAndNotBeenRead(mName.c_str(), "checkpointWrite"));
   if (mCheckpointWriteFlag) {
      params->ioParamString(
            ioFlag,
            mName.c_str(),
            "checkpointWriteTriggerMode",
            &mCheckpointWriteTriggerModeString,
            "step");
      if (ioFlag == PARAMS_IO_READ) {
         pvAssert(mCheckpointWriteTriggerModeString);
         if (!strcmp(mCheckpointWriteTriggerModeString, "step")
             || !strcmp(mCheckpointWriteTriggerModeString, "Step")
             || !strcmp(mCheckpointWriteTriggerModeString, "STEP")) {
            mCheckpointWriteTriggerMode = STEP;
            registerCheckpointData(
                  mName,
                  std::string("nextCheckpointStep"),
                  &mNextCheckpointStep,
                  (std::size_t)1,
                  true /*broadcast*/);
         }
         else if (
               !strcmp(mCheckpointWriteTriggerModeString, "time")
               || !strcmp(mCheckpointWriteTriggerModeString, "Time")
               || !strcmp(mCheckpointWriteTriggerModeString, "TIME")) {
            mCheckpointWriteTriggerMode = SIMTIME;
            registerCheckpointData(
                  mName,
                  std::string("nextCheckpointTime"),
                  &mNextCheckpointSimtime,
                  (std::size_t)1,
                  true /*broadcast*/);
         }
         else if (
               !strcmp(mCheckpointWriteTriggerModeString, "clock")
               || !strcmp(mCheckpointWriteTriggerModeString, "Clock")
               || !strcmp(mCheckpointWriteTriggerModeString, "CLOCK")) {
            mCheckpointWriteTriggerMode = WALLCLOCK;
         }
         else {
            if (mCommunicator->globalCommRank() == 0) {
               ErrorLog() << "Parameter group \"" << mName << "\" checkpointWriteTriggerMode \""
                          << mCheckpointWriteTriggerModeString << "\" is not recognized.\n";
            }
            MPI_Barrier(mCommunicator->globalCommunicator());
            exit(EXIT_FAILURE);
         }
      }
   }
}

void Checkpointer::ioParam_checkpointWriteStepInterval(enum ParamsIOFlag ioFlag, PVParams *params) {
   pvAssert(!params->presentAndNotBeenRead(mName.c_str(), "checkpointWrite"));
   if (mCheckpointWriteFlag) {
      pvAssert(!params->presentAndNotBeenRead(mName.c_str(), "checkpointWriteTriggerMode"));
      if (mCheckpointWriteTriggerMode == STEP) {
         params->ioParamValue(
               ioFlag,
               mName.c_str(),
               "checkpointWriteStepInterval",
               &mCheckpointWriteStepInterval,
               mCheckpointWriteStepInterval);
      }
   }
}

void Checkpointer::ioParam_checkpointWriteTimeInterval(enum ParamsIOFlag ioFlag, PVParams *params) {
   pvAssert(!params->presentAndNotBeenRead(mName.c_str(), "checkpointWrite"));
   if (mCheckpointWriteFlag) {
      pvAssert(!params->presentAndNotBeenRead(mName.c_str(), "checkpointWriteTriggerMode"));
      if (mCheckpointWriteTriggerMode == SIMTIME) {
         params->ioParamValue(
               ioFlag,
               mName.c_str(),
               "checkpointWriteTimeInterval",
               &mCheckpointWriteSimtimeInterval,
               mCheckpointWriteSimtimeInterval);
      }
   }
}

void Checkpointer::ioParam_checkpointWriteClockInterval(
      enum ParamsIOFlag ioFlag,
      PVParams *params) {
   assert(!params->presentAndNotBeenRead(mName.c_str(), "checkpointWrite"));
   if (mCheckpointWriteFlag) {
      pvAssert(!params->presentAndNotBeenRead(mName.c_str(), "checkpointWriteTriggerMode"));
      if (mCheckpointWriteTriggerMode == WALLCLOCK) {
         params->ioParamValueRequired(
               ioFlag,
               mName.c_str(),
               "checkpointWriteClockInterval",
               &mCheckpointWriteWallclockInterval);
      }
   }
}

void Checkpointer::ioParam_checkpointWriteClockUnit(enum ParamsIOFlag ioFlag, PVParams *params) {
   pvAssert(!params->presentAndNotBeenRead(mName.c_str(), "checkpointWrite"));
   if (mCheckpointWriteFlag) {
      pvAssert(!params->presentAndNotBeenRead(mName.c_str(), "checkpointWriteTriggerMode"));
      if (mCheckpointWriteTriggerMode == WALLCLOCK) {
         assert(
               !params->presentAndNotBeenRead(
                     mName.c_str(), "checkpointWriteTriggerClockInterval"));
         params->ioParamString(
               ioFlag,
               mName.c_str(),
               "checkpointWriteClockUnit",
               &mCheckpointWriteWallclockUnit,
               "seconds");
         if (ioFlag == PARAMS_IO_READ) {
            pvAssert(mCheckpointWriteWallclockUnit);
            for (size_t n = 0; n < strlen(mCheckpointWriteWallclockUnit); n++) {
               mCheckpointWriteWallclockUnit[n] = tolower(mCheckpointWriteWallclockUnit[n]);
            }
            if (!strcmp(mCheckpointWriteWallclockUnit, "second")
                || !strcmp(mCheckpointWriteWallclockUnit, "seconds")
                || !strcmp(mCheckpointWriteWallclockUnit, "sec")
                || !strcmp(mCheckpointWriteWallclockUnit, "s")) {
               free(mCheckpointWriteWallclockUnit);
               mCheckpointWriteWallclockUnit            = strdup("seconds");
               mCheckpointWriteWallclockIntervalSeconds = mCheckpointWriteWallclockInterval;
            }
            else if (
                  !strcmp(mCheckpointWriteWallclockUnit, "minute")
                  || !strcmp(mCheckpointWriteWallclockUnit, "minutes")
                  || !strcmp(mCheckpointWriteWallclockUnit, "min")
                  || !strcmp(mCheckpointWriteWallclockUnit, "m")) {
               free(mCheckpointWriteWallclockUnit);
               mCheckpointWriteWallclockUnit = strdup("minutes");
               mCheckpointWriteWallclockIntervalSeconds =
                     mCheckpointWriteWallclockInterval * (time_t)60;
            }
            else if (
                  !strcmp(mCheckpointWriteWallclockUnit, "hour")
                  || !strcmp(mCheckpointWriteWallclockUnit, "hours")
                  || !strcmp(mCheckpointWriteWallclockUnit, "hr")
                  || !strcmp(mCheckpointWriteWallclockUnit, "h")) {
               free(mCheckpointWriteWallclockUnit);
               mCheckpointWriteWallclockUnit = strdup("hours");
               mCheckpointWriteWallclockIntervalSeconds =
                     mCheckpointWriteWallclockInterval * (time_t)3600;
            }
            else if (
                  !strcmp(mCheckpointWriteWallclockUnit, "day")
                  || !strcmp(mCheckpointWriteWallclockUnit, "days")) {
               free(mCheckpointWriteWallclockUnit);
               mCheckpointWriteWallclockUnit = strdup("days");
               mCheckpointWriteWallclockIntervalSeconds =
                     mCheckpointWriteWallclockInterval * (time_t)86400;
            }
            else {
               if (getCommunicator()->globalCommRank() == 0) {
                  ErrorLog().printf(
                        "checkpointWriteClockUnit \"%s\" is unrecognized.  Use \"seconds\", "
                        "\"minutes\", \"hours\", or \"days\".\n",
                        mCheckpointWriteWallclockUnit);
               }
               MPI_Barrier(getCommunicator()->globalCommunicator());
               exit(EXIT_FAILURE);
            }
            FatalIf(
                  mCheckpointWriteWallclockUnit == nullptr,
                  "Error in global rank %d process converting checkpointWriteClockUnit: %s\n",
                  getCommunicator()->globalCommRank(),
                  strerror(errno));
         }
      }
   }
}

void Checkpointer::ioParam_deleteOlderCheckpoints(enum ParamsIOFlag ioFlag, PVParams *params) {
   assert(!params->presentAndNotBeenRead(mName.c_str(), "checkpointWrite"));
   if (mCheckpointWriteFlag) {
      params->ioParamValue(
            ioFlag,
            mName.c_str(),
            "deleteOlderCheckpoints",
            &mDeleteOlderCheckpoints,
            false /*default value*/);
   }
}

void Checkpointer::ioParam_numCheckpointsKept(enum ParamsIOFlag ioFlag, PVParams *params) {
   pvAssert(!params->presentAndNotBeenRead(mName.c_str(), "checkpointWrite"));
   if (mCheckpointWriteFlag) {
      pvAssert(!params->presentAndNotBeenRead(mName.c_str(), "deleteOlderCheckpoints"));
      if (mDeleteOlderCheckpoints) {
         params->ioParamValue(ioFlag, mName.c_str(), "numCheckpointsKept", &mNumCheckpointsKept, 1);
         if (ioFlag == PARAMS_IO_READ && mNumCheckpointsKept <= 0) {
            if (getCommunicator()->commRank() == 0) {
               ErrorLog() << "HyPerCol \"" << mName
                          << "\": numCheckpointsKept must be positive (value was "
                          << mNumCheckpointsKept << ")\n";
            }
            MPI_Barrier(mCommunicator->communicator());
            exit(EXIT_FAILURE);
         }
         if (ioFlag == PARAMS_IO_READ) {
            if (mNumCheckpointsKept < 0) {
               if (getCommunicator()->globalCommRank() == 0) {
                  ErrorLog() << "HyPerCol \"" << mName
                             << "\": numCheckpointsKept must be positive (value was "
                             << mNumCheckpointsKept << ")\n";
               }
               MPI_Barrier(mCommunicator->globalCommunicator());
               exit(EXIT_FAILURE);
            }
            if (mOldCheckpointDirectories.size() != 0) {
               WarnLog() << "ioParamsFillGroup called after list of old checkpoint directories was "
                            "created.  Reinitializing.\n";
            }
            mOldCheckpointDirectories.resize(mNumCheckpointsKept, "");
            mOldCheckpointDirectoriesIndex = 0;
         }
      }
   }
}

void Checkpointer::ioParam_checkpointIndexWidth(enum ParamsIOFlag ioFlag, PVParams *params) {
   assert(!params->presentAndNotBeenRead(mName.c_str(), "checkpointWrite"));
   if (mCheckpointWriteFlag) {
      params->ioParamValue(
            ioFlag,
            mName.c_str(),
            "checkpointIndexWidth",
            &mCheckpointIndexWidth,
            mCheckpointIndexWidth);
   }
}

void Checkpointer::ioParam_suppressNonplasticCheckpoints(
      enum ParamsIOFlag ioFlag,
      PVParams *params) {
   assert(!params->presentAndNotBeenRead(mName.c_str(), "checkpointWrite"));
   if (mCheckpointWriteFlag) {
      params->ioParamValue(
            ioFlag,
            mName.c_str(),
            "suppressNonplasticCheckpoints",
            &mSuppressNonplasticCheckpoints,
            mSuppressNonplasticCheckpoints);
   }
}

void Checkpointer::ioParam_lastCheckpointDir(enum ParamsIOFlag ioFlag, PVParams *params) {
   assert(!params->presentAndNotBeenRead(mName.c_str(), "checkpointWrite"));
   if (!mCheckpointWriteFlag) {
      params->ioParamStringRequired(
            ioFlag, mName.c_str(), "lastCheckpointDir", &mLastCheckpointDir);
   }
}

void Checkpointer::ioParam_initializeFromCheckpointDir(enum ParamsIOFlag ioFlag, PVParams *params) {
   params->ioParamString(
         ioFlag,
         mName.c_str(),
         "initializeFromCheckpointDir",
         &mInitializeFromCheckpointDir,
         "",
         true);
   mDefaultInitializeFromCheckpointFlag =
         mInitializeFromCheckpointDir != nullptr and mInitializeFromCheckpointDir[0] != '\0';
}

// defaultInitializeFromCheckpointFlag was made obsolete Dec 18, 2016.
void Checkpointer::ioParam_defaultInitializeFromCheckpointFlag(
      enum ParamsIOFlag ioFlag,
      PVParams *params) {
   assert(!params->presentAndNotBeenRead(mName.c_str(), "initializeFromCheckpointDir"));
   if (mInitializeFromCheckpointDir != nullptr && mInitializeFromCheckpointDir[0] != '\0') {
      if (params->present(mName.c_str(), "defaultInitializeFromCheckpointFlag")) {
         if (getCommunicator()->commRank() == 0) {
            ErrorLog() << mName << ": defaultInitializeFromCheckpointFlag is obsolete.\n"
                       << "If initializeFromCheckpointDir is non-empty, the objects in the\n"
                       << "HyPerCol will initialize from checkpoint unless they set their\n"
                       << "individual initializeFromCheckpointFlag parameter to false.\n";
         }
         MPI_Barrier(getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   }
}

void Checkpointer::provideFinalStep(long int finalStep) {
   if (mCheckpointIndexWidth < 0) {
      mWidthOfFinalStepNumber = (int)std::floor(std::log10((float)finalStep)) + 1;
   }
}

void Checkpointer::addObserver(Observer *observer, BaseMessage const &message) {
   mObserverTable.addObject(observer->getDescription(), observer);
}

bool Checkpointer::registerCheckpointEntry(
      std::shared_ptr<CheckpointEntry> checkpointEntry,
      bool constantEntireRun) {
   if (mSuppressNonplasticCheckpoints && constantEntireRun) {
      return true;
   }
   std::string const &name = checkpointEntry->getName();
   for (auto &c : mCheckpointRegistry) {
      if (c->getName() == checkpointEntry->getName()) {
         return false;
      }
   }
   mCheckpointRegistry.push_back(checkpointEntry);
   return true;
}

void Checkpointer::registerTimer(Timer const *timer) { mTimers.push_back(timer); }

void Checkpointer::readNamedCheckpointEntry(
      std::string const &objName,
      std::string const &dataName) {
   std::string checkpointEntryName(objName);
   if (!(objName.empty() || dataName.empty())) {
      checkpointEntryName.append("_");
   }
   checkpointEntryName.append(dataName);
   readNamedCheckpointEntry(checkpointEntryName);
}

void Checkpointer::readNamedCheckpointEntry(std::string const &checkpointEntryName) {
   if (mInitializeFromCheckpointDir == nullptr or mInitializeFromCheckpointDir[0] == '\0') {
      return;
   }
   std::string checkpointDirectory = generateDirectory(mInitializeFromCheckpointDir);
   for (auto &c : mCheckpointRegistry) {
      if (c->getName() == checkpointEntryName) {
         double timestamp = 0.0; // not used
         c->read(checkpointDirectory, &timestamp);
         return;
      }
   }
   Fatal() << "initializeFromCheckpoint failed to find checkpointEntryName " << checkpointEntryName
           << "\n";
}

void Checkpointer::findWarmStartDirectory() {
   char warmStartDirectoryBuffer[PV_PATH_MAX];
   if (mCommunicator->commRank() == 0) {
      if (mCheckpointWriteFlag) {
         // Look for largest indexed Checkpointnnnnnn directory in checkpointWriteDir
         pvAssert(mCheckpointWriteDir);
         std::string cpDirString = mCheckpointWriteDir;
         if (cpDirString.c_str()[cpDirString.length() - 1] != '/') {
            cpDirString += "/";
         }
         struct stat statbuf;
         int statstatus = PV_stat(cpDirString.c_str(), &statbuf);
         if (statstatus == 0) {
            if (statbuf.st_mode & S_IFDIR) {
               char *dirs[]     = {mCheckpointWriteDir, nullptr};
               FTS *fts         = fts_open(dirs, FTS_LOGICAL, nullptr);
               FTSENT *ftsent   = fts_read(fts);
               bool found       = false;
               long int cpIndex = LONG_MIN;
               std::string indexedDir;
               for (ftsent = fts_children(fts, 0); ftsent != nullptr; ftsent = ftsent->fts_link) {
                  if (ftsent->fts_statp->st_mode & S_IFDIR) {
                     long int x;
                     int k = sscanf(ftsent->fts_name, "Checkpoint%ld", &x);
                     if (x > cpIndex) {
                        cpIndex    = x;
                        indexedDir = ftsent->fts_name;
                        found      = true;
                     }
                  }
               }
               FatalIf(
                     !found,
                     "restarting but checkpointWriteFlag is set and "
                     "checkpointWriteDir directory \"%s\" does not have any "
                     "checkpoints\n",
                     mCheckpointWriteDir);
               mCheckpointReadDirectory = cpDirString;
               mCheckpointReadDirectory.append(indexedDir);
            }
            else {
               Fatal().printf(
                     "checkpoint read directory \"%s\" is "
                     "not a directory.\n",
                     mCheckpointWriteDir);
            }
         }
         else if (errno == ENOENT) {
            Fatal().printf(
                  "restarting but neither Last nor checkpointWriteDir "
                  "directory \"%s\" exists.\n",
                  mCheckpointWriteDir);
         }
      }
      else {
         pvAssert(mLastCheckpointDir);
         FatalIf(
               mLastCheckpointDir[0] = '\0',
               "Restart flag set, but unable to determine restart directory.\n");
         mCheckpointReadDirectory = strdup(mLastCheckpointDir);
      }
      FatalIf(
            mCheckpointReadDirectory.size() >= PV_PATH_MAX,
            "Restart flag set, but inferred checkpoint read directory is too long (%zu "
            "bytes).\n",
            mCheckpointReadDirectory.size());
      memcpy(
            warmStartDirectoryBuffer,
            mCheckpointReadDirectory.c_str(),
            mCheckpointReadDirectory.size());
      warmStartDirectoryBuffer[mCheckpointReadDirectory.size()] = '\0';
   }
   MPI_Bcast(warmStartDirectoryBuffer, PV_PATH_MAX, MPI_CHAR, 0, mCommunicator->communicator());
   if (mCommunicator->commRank() != 0) {
      mCheckpointReadDirectory = warmStartDirectoryBuffer;
   }
}

void Checkpointer::setCheckpointReadDirectory() {
   if (mCheckpointWriteFlag) {
      findWarmStartDirectory();
   }
   else {
      mCheckpointReadDirectory = mLastCheckpointDir;
   }
}

void Checkpointer::setCheckpointReadDirectory(std::string const &checkpointReadDir) {
   std::vector<std::string> checkpointReadDirs;
   checkpointReadDirs.reserve(mCommunicator->numCommBatches());
   std::size_t dirStart = (std::size_t)0;
   while (dirStart < checkpointReadDir.size()) {
      std::size_t dirStop = checkpointReadDir.find(':', dirStart);
      if (dirStop == std::string::npos) {
         dirStop = checkpointReadDir.size();
      }
      checkpointReadDirs.push_back(checkpointReadDir.substr(dirStart, dirStop - dirStart));
      FatalIf(
            checkpointReadDirs.size() > (std::size_t)mCommunicator->numCommBatches(),
            "Checkpoint read parsing error: Too many colon separated "
            "checkpoint read directories. "
            "Only specify %d checkpoint directories.\n",
            mCommunicator->numCommBatches());
      dirStart = dirStop + 1;
   }
   // Make sure number matches up
   int const count = (int)checkpointReadDirs.size();
   FatalIf(
         count != mCommunicator->numCommBatches() && count != 1,
         "Checkpoint read parsing error: Not enough colon separated "
         "checkpoint read directories. "
         "Running with %d batch MPIs but only %d colon separated checkpoint "
         "directories.\n",
         mCommunicator->numCommBatches(),
         count);
   // Grab the directory for this rank and use as mCheckpointReadDir
   int const checkpointIndex = count == 1 ? 0 : mCommunicator->commBatch();
   std::string dirString     = expandLeadingTilde(checkpointReadDirs[checkpointIndex].c_str());
   mCheckpointReadDirectory  = strdup(dirString.c_str());
   pvAssert(!mCheckpointReadDirectory.empty());

   InfoLog().printf(
         "Global Rank %d process setting checkpointReadDir to %s.\n",
         mCommunicator->globalCommRank(),
         mCheckpointReadDirectory.c_str());
}

void Checkpointer::checkpointRead(double *simTimePointer, long int *currentStepPointer) {
   std::string checkpointReadDirectory = generateDirectory(mCheckpointReadDirectory);
   double readTime;
   for (auto &c : mCheckpointRegistry) {
      c->read(checkpointReadDirectory, &readTime);
   }
   mTimeInfoCheckpointEntry->read(checkpointReadDirectory.c_str(), &readTime);
   if (simTimePointer) {
      *simTimePointer = mTimeInfo.mSimTime;
   }
   if (currentStepPointer) {
      *currentStepPointer = mTimeInfo.mCurrentCheckpointStep;
   }
   notify(
         mObserverTable,
         std::make_shared<ProcessCheckpointReadMessage const>(checkpointReadDirectory),
         getCommunicator()->commRank() == 0 /*printFlag*/);
}

void Checkpointer::checkpointWrite(double simTime) {
   mTimeInfo.mSimTime = simTime;
   // set mSimTime here so that it is available in routines called by checkpointWrite.
   if (!mCheckpointWriteFlag) {
      return;
   }
   if (checkpointWriteSignal()) {
      InfoLog().printf(
            "Global rank %d: checkpointing in response to SIGUSR1 at time %f.\n",
            getCommunicator()->globalCommRank(),
            simTime);
      mCheckpointSignal = 0;
      checkpointNow();
   }
   else {
      switch (mCheckpointWriteTriggerMode) {
         case NONE:
            pvAssert(0);
            break; // Only NONE if checkpointWrite is off, in which case this method should have
         // returned above.
         case STEP: checkpointWriteStep(); break;
         case SIMTIME: checkpointWriteSimtime(); break;
         case WALLCLOCK: checkpointWriteWallclock(); break;
         default: pvAssert(0); break;
      }
   }
   mTimeInfo.mCurrentCheckpointStep++;
}

bool Checkpointer::checkpointWriteSignal() {
   if (getCommunicator()->globalCommRank() == 0) {
      int sigstatus = PV_SUCCESS;
      sigset_t pollusr1;

      sigstatus = sigpending(&pollusr1);
      assert(sigstatus == 0);
      mCheckpointSignal = sigismember(&pollusr1, SIGUSR1);
      assert(mCheckpointSignal == 0 || mCheckpointSignal == 1);
      if (mCheckpointSignal) {
         sigstatus = sigemptyset(&pollusr1);
         assert(sigstatus == 0);
         sigstatus = sigaddset(&pollusr1, SIGUSR1);
         assert(sigstatus == 0);
         int result = 0;
         sigwait(&pollusr1, &result);
         assert(result == SIGUSR1);
      }
   }
   MPI_Bcast(&mCheckpointSignal, 1 /*count*/, MPI_INT, 0, mCommunicator->globalCommunicator());
   bool signaled = (mCheckpointSignal != 0);
   if (signaled) {
      InfoLog().printf(
            "Global rank %d: checkpointing in response to SIGUSR1 at time %f.\n",
            getCommunicator()->globalCommRank(),
            mTimeInfo.mSimTime);
      mCheckpointSignal = 0;
      checkpointNow();
   }
   return signaled;
}

void Checkpointer::checkpointWriteStep() {
   pvAssert(mCheckpointWriteStepInterval > 0);
   if (mTimeInfo.mCurrentCheckpointStep % mCheckpointWriteStepInterval == 0) {
      mNextCheckpointStep = mTimeInfo.mCurrentCheckpointStep + mCheckpointWriteStepInterval;
      // We don't use mNextCheckpointStep for anything.  It's here because HyPerCol checkpointed
      // it and we can test whether nothing broke during the refactor by comparing directories.
      checkpointNow();
   }
}

void Checkpointer::checkpointWriteSimtime() {
   if (mTimeInfo.mSimTime >= mNextCheckpointSimtime) {
      checkpointNow();
      mNextCheckpointSimtime += mCheckpointWriteSimtimeInterval;
   }
}

void Checkpointer::checkpointWriteWallclock() {
   std::time_t currentTime = std::time(nullptr);
   if (currentTime == (std::time_t)(-1)) {
      throw;
   }
   double elapsed = std::difftime(currentTime, mLastCheckpointWallclock);
   if (elapsed >= mCheckpointWriteWallclockInterval) {
      checkpointNow();
      mLastCheckpointWallclock = currentTime;
   }
}

void Checkpointer::checkpointNow() {
   std::stringstream checkpointDirStream;
   checkpointDirStream << mCheckpointWriteDir << "/Checkpoint";
   int fieldWidth = mCheckpointIndexWidth < 0 ? mWidthOfFinalStepNumber : mCheckpointIndexWidth;
   checkpointDirStream.fill('0');
   checkpointDirStream.width(fieldWidth);
   checkpointDirStream << mTimeInfo.mCurrentCheckpointStep;
   std::string checkpointDirectory = checkpointDirStream.str();
   if (checkpointDirectory != mCheckpointReadDirectory) {
      /* Note: the strcmp isn't perfect, since there are multiple ways to specify a path that
       * points to the same directory.  Should use realpath, but that breaks under OS X. */
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog() << "Checkpointing to \"" << checkpointDirectory
                   << "\", simTime = " << mTimeInfo.mSimTime << "\n";
      }
   }
   else {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "Skipping checkpoint to \"%s\","
               " which would clobber the checkpointRead checkpoint.\n",
               checkpointDirectory.c_str());
      }
      return;
   }
   checkpointToDirectory(checkpointDirectory);
   if (mCommunicator->commRank() == 0) {
      InfoLog().printf("checkpointWrite complete. simTime = %f\n", mTimeInfo.mSimTime);
   }

   if (mDeleteOlderCheckpoints) {
      rotateOldCheckpoints(checkpointDirectory);
   }
}

void Checkpointer::checkpointToDirectory(std::string const &directory) {
   std::string checkpointDirectory = generateDirectory(directory);
   mCheckpointTimer->start();
   if (getCommunicator()->commRank() == 0) {
      InfoLog() << "Checkpointing to directory \"" << checkpointDirectory
                << "\" at simTime = " << mTimeInfo.mSimTime << "\n";
      struct stat timeinfostat;
      std::string timeinfoFilename(checkpointDirectory);
      timeinfoFilename.append("/timeinfo.bin");
      int statstatus = stat(timeinfoFilename.c_str(), &timeinfostat);
      if (statstatus == 0) {
         WarnLog() << "Checkpoint directory \"" << checkpointDirectory
                   << "\" has existing timeinfo.bin, which is now being deleted.\n";
         mTimeInfoCheckpointEntry->remove(checkpointDirectory);
      }
   }
   notify(
         mObserverTable,
         std::make_shared<PrepareCheckpointWriteMessage const>(checkpointDirectory),
         getCommunicator()->commRank() == 0 /*printFlag*/);
   ensureDirExists(getCommunicator(), checkpointDirectory.c_str());
   for (auto &c : mCheckpointRegistry) {
      c->write(checkpointDirectory, mTimeInfo.mSimTime, mVerifyWritesFlag);
   }
   mTimeInfoCheckpointEntry->write(checkpointDirectory, mTimeInfo.mSimTime, mVerifyWritesFlag);
   mCheckpointTimer->stop();
   mCheckpointTimer->start();
   writeTimers(checkpointDirectory);
   mCheckpointTimer->stop();
}

void Checkpointer::finalCheckpoint(double simTime) {
   mTimeInfo.mSimTime = simTime;
   if (mCheckpointWriteFlag) {
      checkpointNow();
   }
   else if (mLastCheckpointDir != nullptr && mLastCheckpointDir[0] != '\0') {
      checkpointToDirectory(std::string(mLastCheckpointDir));
   }
}

void Checkpointer::rotateOldCheckpoints(std::string const &newCheckpointDirectory) {
   std::string &oldestCheckpointDir = mOldCheckpointDirectories[mOldCheckpointDirectoriesIndex];
   if (!oldestCheckpointDir.empty()) {
      if (mCommunicator->commRank() == 0) {
         std::string targetDirectory = generateDirectory(oldestCheckpointDir);
         struct stat lcp_stat;
         int statstatus = stat(targetDirectory.c_str(), &lcp_stat);
         if (statstatus != 0 || !(lcp_stat.st_mode & S_IFDIR)) {
            if (statstatus == 0) {
               ErrorLog().printf(
                     "Failed to delete older checkpoint: failed to stat \"%s\": %s.\n",
                     targetDirectory.c_str(),
                     strerror(errno));
            }
            else {
               ErrorLog().printf(
                     "Deleting older checkpoint: \"%s\" exists but is not a directory.\n",
                     targetDirectory.c_str());
            }
         }
         sync();
         std::string rmrf_string("");
         rmrf_string     = rmrf_string + "rm -r '" + targetDirectory + "'";
         int rmrf_result = system(rmrf_string.c_str());
         if (rmrf_result != 0) {
            WarnLog().printf(
                  "unable to delete older checkpoint \"%s\": rm command returned %d\n",
                  targetDirectory.c_str(),
                  WEXITSTATUS(rmrf_result));
         }
      }
      MPI_Barrier(mCommunicator->globalCommunicator());
      if (mCommunicator->globalCommRank() == 0) {
         sync();
         struct stat oldcp_stat;
         int statstatus = stat(oldestCheckpointDir.c_str(), &oldcp_stat);
         if (statstatus == 0 && (oldcp_stat.st_mode & S_IFDIR)) {
            int rmdirstatus = rmdir(oldestCheckpointDir.c_str());
            if (rmdirstatus) {
               ErrorLog().printf(
                     "Unable to delete older checkpoint \"%s\": rmdir command returned %d "
                     "(%s)\n",
                     oldestCheckpointDir.c_str(),
                     errno,
                     std::strerror(errno));
            }
         }
      }
   }
   mOldCheckpointDirectories[mOldCheckpointDirectoriesIndex] = newCheckpointDirectory;
   mOldCheckpointDirectoriesIndex++;
   if (mOldCheckpointDirectoriesIndex == mNumCheckpointsKept) {
      mOldCheckpointDirectoriesIndex = 0;
   }
}

void Checkpointer::writeTimers(PrintStream &stream) const {
   for (auto timer : mTimers) {
      timer->fprint_time(stream);
   }
}

std::string Checkpointer::generateDirectory(std::string const &baseDirectory) {
   std::string path(baseDirectory);
   int batchWidth = getCommunicator()->numCommBatches();
   if (batchWidth > 1) {
      path.append("/batchsweep_");
      std::size_t lengthLargestBatchIndex = std::to_string(batchWidth - 1).size();
      std::string batchIndexAsString      = std::to_string(getCommunicator()->commBatch());
      std::size_t lengthBatchIndex        = batchIndexAsString.size();
      if (lengthBatchIndex < lengthLargestBatchIndex) {
         path.append(lengthLargestBatchIndex - lengthBatchIndex, '0');
      }
      path.append(batchIndexAsString);
   }
   ensureDirExists(getCommunicator(), path.c_str());
   return path;
}

void Checkpointer::writeTimers(std::string const &directory) {
   if (getCommunicator()->commRank() == 0) {
      std::string timerpathstring = directory;
      timerpathstring += "/";
      timerpathstring += "timers.txt";

      const char *timerpath = timerpathstring.c_str();
      FileStream timerstream(timerpath, std::ios_base::out, mVerifyWritesFlag);
      writeTimers(timerstream);
   }
}

std::string const Checkpointer::mDefaultOutputPath = "output";

} // namespace PV
