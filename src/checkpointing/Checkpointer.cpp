/*
 * Checkpointer.cpp
 *
 *  Created on Sep 28, 2016
 *      Author: Pete Schultz
 */

#include "Checkpointer.hpp"

#include "checkpointing/CheckpointingMessages.hpp"
#include "utils/ExpandLeadingTilde.hpp"
#include <cerrno>
#include <climits>
#include <cmath>
#include <fts.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>

namespace PV {

Checkpointer::Checkpointer(
      std::string const &name,
      Communicator const *communicator,
      std::shared_ptr<Arguments const> arguments)
      : mName(name) {
   Subject::initializeTable(name.c_str());
   mMPIBlock = communicator->getIOMPIBlock();
   initBlockDirectoryName();

   mWarmStart               = arguments->getBooleanArgument("Restart");
   mCheckpointReadDirectory = arguments->getStringArgument("CheckpointReadDirectory");
   if (!mCheckpointReadDirectory.empty()) {
      extractCheckpointReadDirectory();
   }

   mTimeInfoCheckpointEntry = std::make_shared<CheckpointEntryData<Checkpointer::TimeInfo>>(
         std::string("timeinfo"), &mTimeInfo, (size_t)1, true /*broadcast*/);
   // This doesn't get put into mCheckpointRegistry because we handle the timeinfo separately.
   mCheckpointTimer = new Timer(mName.c_str(), "column", "checkpoint");
   registerTimer(mCheckpointTimer);
}

Checkpointer::~Checkpointer() {
   free(mCheckpointWriteDir);
   free(mCheckpointWriteTriggerModeString);
   free(mCheckpointWriteWallclockUnit);
   free(mLastCheckpointDir);
   free(mInitializeFromCheckpointDir);

   // Don't delete the objects in the ObserverComponentTable; Checkpointer doesn't own them.
   mTable->clear();
   delete mCheckpointTimer;
}

void Checkpointer::initBlockDirectoryName() {
   mBlockDirectoryName.clear();
   if (mMPIBlock->getGlobalNumRows() != mMPIBlock->getNumRows()
       or mMPIBlock->getGlobalNumColumns() != mMPIBlock->getNumColumns()
       or mMPIBlock->getGlobalBatchDimension() != mMPIBlock->getBatchDimension()) {
      int const blockColumnIndex = mMPIBlock->getStartColumn() / mMPIBlock->getNumColumns();
      int const blockRowIndex    = mMPIBlock->getStartRow() / mMPIBlock->getNumRows();
      int const blockBatchIndex  = mMPIBlock->getStartBatch() / mMPIBlock->getBatchDimension();
      mBlockDirectoryName.append("block_");
      mBlockDirectoryName.append("col" + std::to_string(blockColumnIndex));
      mBlockDirectoryName.append("row" + std::to_string(blockRowIndex));
      mBlockDirectoryName.append("elem" + std::to_string(blockBatchIndex));
   }
}

void Checkpointer::ioParams(enum ParamsIOFlag ioFlag, PVParams *params) {
   ioParamsFillGroup(ioFlag, params);

   // If WarmStart is set and CheckpointWrite is false, CheckpointReadDirectory
   // is LastCheckpointDir. If WarmStart is set and CheckpointWrite is true, we
   // CheckpointReadDirectory is the last checkpoint in CheckpointWriteDir.
   // Hence we cannot set CheckpointReadDirectory until we've read the params.
   if (mWarmStart and ioFlag == PARAMS_IO_READ) {
      // Arguments class should prevent -r and -c from both being set.
      pvAssert(mCheckpointReadDirectory.empty());
      // Set mCheckpointReadDirectory to the last checkpoint in the CheckpointWrite directory.
      findWarmStartDirectory();
   }
}

void Checkpointer::ioParamsFillGroup(enum ParamsIOFlag ioFlag, PVParams *params) {
   ioParam_verifyWrites(ioFlag, params);
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
}

void Checkpointer::ioParam_verifyWrites(enum ParamsIOFlag ioFlag, PVParams *params) {
   params->ioParamValue(ioFlag, mName.c_str(), "verifyWrites", &mVerifyWrites, mVerifyWrites);
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
         ensureDirExists(mMPIBlock, mCheckpointWriteDir);
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
                  true /*broadcast*/,
                  false /*not constant entire run*/);
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
                  true /*broadcast*/,
                  false /*not constant entire run*/);
         }
         else if (
               !strcmp(mCheckpointWriteTriggerModeString, "clock")
               || !strcmp(mCheckpointWriteTriggerModeString, "Clock")
               || !strcmp(mCheckpointWriteTriggerModeString, "CLOCK")) {
            mCheckpointWriteTriggerMode = WALLCLOCK;
         }
         else {
            if (mMPIBlock->getRank() == 0) {
               ErrorLog() << "Parameter group \"" << mName << "\" checkpointWriteTriggerMode \""
                          << mCheckpointWriteTriggerModeString << "\" is not recognized.\n";
            }
            MPI_Barrier(mMPIBlock->getComm());
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
               if (mMPIBlock->getRank() == 0) {
                  ErrorLog().printf(
                        "checkpointWriteClockUnit \"%s\" is unrecognized.  Use \"seconds\", "
                        "\"minutes\", \"hours\", or \"days\".\n",
                        mCheckpointWriteWallclockUnit);
               }
               MPI_Barrier(mMPIBlock->getComm());
               exit(EXIT_FAILURE);
            }
            FatalIf(
                  mCheckpointWriteWallclockUnit == nullptr,
                  "Error in global rank %d process converting checkpointWriteClockUnit: %s\n",
                  mMPIBlock->getRank(),
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
            if (mMPIBlock->getRank() == 0) {
               ErrorLog() << "HyPerCol \"" << mName
                          << "\": numCheckpointsKept must be positive (value was "
                          << mNumCheckpointsKept << ")\n";
            }
            MPI_Barrier(mMPIBlock->getComm());
            exit(EXIT_FAILURE);
         }
         if (ioFlag == PARAMS_IO_READ) {
            if (mNumCheckpointsKept < 0) {
               if (mMPIBlock->getRank() == 0) {
                  ErrorLog() << "HyPerCol \"" << mName
                             << "\": numCheckpointsKept must be positive (value was "
                             << mNumCheckpointsKept << ")\n";
               }
               MPI_Barrier(mMPIBlock->getComm());
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
   if (ioFlag == PARAMS_IO_READ and mInitializeFromCheckpointDir != nullptr
       and mInitializeFromCheckpointDir[0] != '\0') {
      verifyCheckpointDirectory(mInitializeFromCheckpointDir, "InitializeFromCheckpointDir.\n");
      mInitializeFromCheckpointFileManager =
            std::make_shared<FileManager>(getMPIBlock(), mInitializeFromCheckpointDir);
   }
}

void Checkpointer::provideFinalStep(long int finalStep) {
   if (mCheckpointIndexWidth < 0) {
      mWidthOfFinalStepNumber = (int)std::floor(std::log10((float)finalStep)) + 1;
   }
}

bool Checkpointer::registerCheckpointEntry(
      std::shared_ptr<CheckpointEntry> checkpointEntry,
      bool constantEntireRun) {
   if (mSuppressNonplasticCheckpoints && constantEntireRun) {
      return true;
   }
   std::string const &name = checkpointEntry->getName();
   for (auto &c : mCheckpointRegistry) {
      if (c->getName() == name) {
         return false;
      }
   }
   mCheckpointRegistry.push_back(checkpointEntry);
   return true;
}

void Checkpointer::registerTimer(Timer const *timer) { mTimers.push_back(timer); }

void Checkpointer::readNamedCheckpointEntry(
      std::string const &objName,
      std::string const &dataName,
      bool constantEntireRun) {
   std::string checkpointEntryName(objName);
   if (!(objName.empty() || dataName.empty())) {
      checkpointEntryName.append("_");
   }
   checkpointEntryName.append(dataName);
   readNamedCheckpointEntry(checkpointEntryName, constantEntireRun);
}

void Checkpointer::readNamedCheckpointEntry(
      std::string const &checkpointEntryName,
      bool constantEntireRun) {
   if (mSuppressNonplasticCheckpoints and constantEntireRun) {
      return;
   }
   for (auto &c : mCheckpointRegistry) {
      if (c->getName() == checkpointEntryName) {
         double timestamp = 0.0; // not used
         c->read(mInitializeFromCheckpointFileManager, &timestamp);
         return;
      }
   }
   Fatal() << "initializeFromCheckpoint failed to find checkpointEntryName " << checkpointEntryName
           << "\n";
}

void Checkpointer::findWarmStartDirectory() {
   char warmStartDirectoryBuffer[PV_PATH_MAX];
   if (mMPIBlock->getRank() == 0) {
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
               char *dirs[]              = {mCheckpointWriteDir, nullptr};
               FTS *fts                  = fts_open(dirs, FTS_LOGICAL, nullptr);
               FTSENT *ftsent            = fts_read(fts);
               bool found                = false;
               long unsigned int cpIndex = 0UL;
               std::string latestCheckpoint;
               for (ftsent = fts_children(fts, 0); ftsent != nullptr; ftsent = ftsent->fts_link) {
                  if (ftsent->fts_statp->st_mode & S_IFDIR) {
                     long unsigned int x;
                     int scanstatus = sscanf(ftsent->fts_name, "Checkpoint%lu", &x);
                     if (scanstatus == 1 and (x > cpIndex or latestCheckpoint.empty())) {
                        std::string candidateCheckpoint = cpDirString + ftsent->fts_name;
                        if (isCompleteCheckpoint(candidateCheckpoint)) {
                           cpIndex          = x;
                           found            = true;
                           latestCheckpoint = candidateCheckpoint;
                        }
                     }
                  }
               }
               FatalIf(
                     !found,
                     "restarting but checkpointWriteFlag is set and "
                     "checkpointWriteDir directory \"%s\" does not have any "
                     "checkpoints\n",
                     mCheckpointWriteDir);
               mCheckpointReadDirectory = latestCheckpoint;
            }
            else {
               Fatal().printf(
                     "restart flag was set but checkpointWriteDir \"%s\" is not a directory.\n",
                     mCheckpointWriteDir);
            }
         }
         else if (errno == ENOENT) {
            Fatal().printf(
                  "restart flag was set but checkpointWriteDir directory \"%s\" does not exist.\n",
                  mCheckpointWriteDir);
         }
         else {
            Fatal().printf(
                  "restart flag was set but opening checkpointWriteDir failed: %s\n",
                  strerror(errno));
         }
      }
      else {
         pvAssert(mLastCheckpointDir);
         FatalIf(
               mLastCheckpointDir[0] == '\0',
               "Restart flag set, but unable to determine restart directory.\n");
         FatalIf(
               !isCompleteCheckpoint(std::string(mLastCheckpointDir)),
               "Restart flag set but \"%s\" does not appear to be a complete checkpoint",
               mLastCheckpointDir);
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
   MPI_Bcast(warmStartDirectoryBuffer, PV_PATH_MAX, MPI_CHAR, 0, mMPIBlock->getComm());
   if (mMPIBlock->getRank() == 0) {
      InfoLog() << "Restarting from checkpoint \"" << mCheckpointReadDirectory << "\"\n";
   }
   else {
      mCheckpointReadDirectory = warmStartDirectoryBuffer;
   }
}

bool Checkpointer::isCompleteCheckpoint(std::string const &candidateCheckpoint) const {
   pvAssert(mMPIBlock->getRank() == 0); // should only be called by root process of the MPIBlock
   std::string blockDirectory(candidateCheckpoint);
   if (!mBlockDirectoryName.empty()) {
      blockDirectory.append("/").append(mBlockDirectoryName);
   }
   bool checkpointComplete = false;
   errno                   = 0;
   struct stat pathstat;
   int statstatus = stat(blockDirectory.c_str(), &pathstat);
   if (statstatus != 0) {
      ErrorLog().printf("Failed to stat \"%s\": %s\n", blockDirectory.c_str(), strerror(errno));
      return false;
   }
   if (statstatus == 0 and S_ISDIR(pathstat.st_mode)) {
      std::string timeinfopath = blockDirectory + "/timeinfo.bin";
      statstatus               = stat(timeinfopath.c_str(), &pathstat);
      if (statstatus) {
         if (errno == ENOENT) {
            WarnLog() << "Directory \"" << candidateCheckpoint << "\""
                      << " exists but does not appear to be a complete checkpoint.\n";
         }
         else {
            WarnLog() << "Attempting to stat \"" << timeinfopath << "\" failed: " << strerror(errno)
                      << "\n";
         }
         return false;
      }
      return true;
   }
   return false;
}

void Checkpointer::readStateFromCheckpoint() {
   if (getInitializeFromCheckpointDir() and getInitializeFromCheckpointDir()[0]) {
      notify(
            std::make_shared<ReadStateFromCheckpointMessage<Checkpointer>>(this),
            mMPIBlock->getRank() == 0 /*printFlag*/);
   }
}

void Checkpointer::extractCheckpointReadDirectory() {
   std::vector<std::string> checkpointReadDirs;
   checkpointReadDirs.reserve(mMPIBlock->getBatchDimension());
   std::size_t dirStart = (std::size_t)0;
   while (dirStart < mCheckpointReadDirectory.size()) {
      std::size_t dirStop = mCheckpointReadDirectory.find(':', dirStart);
      if (dirStop == std::string::npos) {
         dirStop = mCheckpointReadDirectory.size();
      }
      checkpointReadDirs.push_back(mCheckpointReadDirectory.substr(dirStart, dirStop - dirStart));
      FatalIf(
            checkpointReadDirs.size() > (std::size_t)mMPIBlock->getBatchDimension(),
            "Checkpoint read parsing error: Too many colon separated "
            "checkpoint read directories. "
            "Only specify %d checkpoint directories.\n",
            mMPIBlock->getBatchDimension());
      dirStart = dirStop + 1;
   }
   // Make sure number matches up
   int const count = (int)checkpointReadDirs.size();
   FatalIf(
         count != mMPIBlock->getBatchDimension() && count != 1,
         "Checkpoint read parsing error: Not enough colon separated "
         "checkpoint read directories. "
         "Running with %d batch MPIs but only %d colon separated checkpoint "
         "directories.\n",
         mMPIBlock->getBatchDimension(),
         count);
   // Grab the directory for this rank and use as mCheckpointReadDir
   int const checkpointIndex = count == 1 ? 0 : mMPIBlock->getBatchIndex();
   std::string dirString     = expandLeadingTilde(checkpointReadDirs[checkpointIndex].c_str());
   mCheckpointReadDirectory  = dirString.c_str();
   pvAssert(!mCheckpointReadDirectory.empty());

   if (getMPIBlock()->getGlobalRank() == 0) {
      InfoLog().printf(
            "Global rank %d process setting CheckpointReadDirectory to %s.\n",
            mMPIBlock->getGlobalRank(),
            mCheckpointReadDirectory.c_str());
   }
}

void Checkpointer::checkpointRead(double *simTimePointer, long int *currentStepPointer) {
   verifyCheckpointDirectory(mCheckpointReadDirectory.c_str(), "CheckpointReadDirectory");
   auto cpreadFileManager =
         std::make_shared<FileManager>(getMPIBlock(), getCheckpointReadDirectory());
   double readTime;
   for (auto &c : mCheckpointRegistry) {
      c->read(cpreadFileManager, &readTime);
   }
   mTimeInfoCheckpointEntry->read(cpreadFileManager, &readTime);
   if (simTimePointer) {
      *simTimePointer = mTimeInfo.mSimTime;
   }
   if (currentStepPointer) {
      *currentStepPointer = mTimeInfo.mCurrentCheckpointStep;
   }
   notify(
         std::make_shared<ProcessCheckpointReadMessage const>(cpreadFileManager),
         getMPIBlock()->getRank() == 0 /*printFlag*/);
}

void Checkpointer::checkpointWrite(double simTime) {
   mTimeInfo.mSimTime = simTime;
   // set mSimTime here so that it is available in routines called by checkpointWrite.
   bool isScheduled = scheduledCheckpoint(); // Is a checkpoint scheduled to occur here?
   // If there is both a signal and scheduled checkpoint, we call checkpointWriteSignal
   // but not checkpointNow, because signal-generated checkpoints shouldn't be deleted.
   int receivedSignal = retrieveSignal();
   assert(receivedSignal == 0 || receivedSignal == SIGUSR1 || receivedSignal == SIGUSR2);
   if (receivedSignal) {
      checkpointWriteSignal(receivedSignal);
   }
   else if (isScheduled) {
      checkpointNow();
   }
   mTimeInfo.mCurrentCheckpointStep++;
   // increment step number here so that initial conditions correspond to step zero, etc.
}

void Checkpointer::checkpointDelete(std::shared_ptr<FileManager const> fileManager) {
   sync();
   mTimeInfoCheckpointEntry->remove(fileManager);
   fileManager->deleteFile(std::string("timers.txt"));

   for (auto &c : mCheckpointRegistry) {
      c->remove(fileManager);
   }
   fileManager->deleteDirectory(std::string(""));
}

int Checkpointer::retrieveSignal() {
   int checkpointSignal;
   if (mMPIBlock->getGlobalRank() == 0) {
      sigset_t sigpendingset;

      int sigstatus = sigpending(&sigpendingset);
      FatalIf(sigstatus, "Signal handling routine sigpending() failed. %s\n", strerror(errno));
      // If somehow both SIGUSR1 and SIGUSR2 are pending, the checkpoint-and-exit signal
      // should take priority.
      checkpointSignal = sigismember(&sigpendingset, SIGUSR2)
                               ? SIGUSR2
                               : sigismember(&sigpendingset, SIGUSR1) ? SIGUSR1 : 0;

      if (checkpointSignal) {
         sigstatus = sigemptyset(&sigpendingset);
         assert(sigstatus == 0);
         sigstatus = sigaddset(&sigpendingset, checkpointSignal);
         assert(sigstatus == 0);
         int result = 0;
         sigwait(&sigpendingset, &result);
         assert(result == checkpointSignal);
      }
   }
   MPI_Bcast(&checkpointSignal, 1 /*count*/, MPI_INT, 0, mMPIBlock->getGlobalComm());
   return (checkpointSignal);
}

bool Checkpointer::scheduledCheckpoint() {
   bool isScheduled = false;
   if (mCheckpointWriteFlag) {
      switch (mCheckpointWriteTriggerMode) {
         case NONE:
            // Only NONE if checkpointWrite is off, in which case this method should not get called
            pvAssert(0);
            break;
         case STEP: isScheduled      = scheduledStep(); break;
         case SIMTIME: isScheduled   = scheduledSimTime(); break;
         case WALLCLOCK: isScheduled = scheduledWallclock(); break;
         default: pvAssert(0); break;
      }
   }
   return isScheduled;
}

bool Checkpointer::scheduledStep() {
   bool isScheduled = false;
   pvAssert(mCheckpointWriteStepInterval > 0);
   if (mTimeInfo.mCurrentCheckpointStep % mCheckpointWriteStepInterval == 0) {
      mNextCheckpointStep = mTimeInfo.mCurrentCheckpointStep + mCheckpointWriteStepInterval;
      isScheduled         = true;
   }
   return isScheduled;
}

bool Checkpointer::scheduledSimTime() {
   bool isScheduled = false;
   if (mTimeInfo.mSimTime >= mNextCheckpointSimtime) {
      mNextCheckpointSimtime += mCheckpointWriteSimtimeInterval;
      isScheduled = true;
   }
   return isScheduled;
}

bool Checkpointer::scheduledWallclock() {
   bool isScheduled = false;
   std::time_t currentTime;
   if (mMPIBlock->getGlobalRank() == 0) {
      currentTime = std::time(nullptr);
   }
   MPI_Bcast(&currentTime, sizeof(currentTime), MPI_CHAR, 0, mMPIBlock->getComm());
   if (currentTime == (std::time_t)(-1)) {
      throw;
   }
   double elapsed = std::difftime(currentTime, mLastCheckpointWallclock);
   if (elapsed >= mCheckpointWriteWallclockInterval) {
      isScheduled              = true;
      mLastCheckpointWallclock = currentTime;
   }
   return isScheduled;
}

void Checkpointer::checkpointWriteSignal(int checkpointSignal) {
   char const *signalName = nullptr;
   switch (checkpointSignal) {
      case SIGUSR1: signalName = "SIGUSR1 (checkpoint and continue)"; break;
      case SIGUSR2: signalName = "SIGUSR2 (checkpoint and exit)"; break;
      default:
         pvAssert(0);
         // checkpointWriteSignal should not be called unless the signal is one of the above
         break;
   }
   InfoLog().printf(
         "Global rank %d: checkpointing in response to %s at time %f.\n",
         mMPIBlock->getGlobalRank(),
         signalName,
         mTimeInfo.mSimTime);
   std::string checkpointDirectory = makeCheckpointDirectoryFromCurrentStep();
   checkpointToDirectory(checkpointDirectory);
   if (checkpointSignal == SIGUSR2) {
      if (mMPIBlock->getGlobalRank() == 0) {
         InfoLog() << "Exiting.\n";
      }
      MPI_Finalize();
      exit(EXIT_SUCCESS);
   }
}

std::string Checkpointer::makeCheckpointDirectoryFromCurrentStep() {
   std::string checkpointDirectory;
   if (mCheckpointWriteFlag) {
      std::stringstream checkpointDirStream;
      checkpointDirStream << mCheckpointWriteDir << "/Checkpoint";
      int fieldWidth = mCheckpointIndexWidth < 0 ? mWidthOfFinalStepNumber : mCheckpointIndexWidth;
      checkpointDirStream.fill('0');
      checkpointDirStream.width(fieldWidth);
      checkpointDirStream << mTimeInfo.mCurrentCheckpointStep;
      checkpointDirectory = checkpointDirStream.str();
   }
   else {
      checkpointDirectory = mLastCheckpointDir;
   }
   return checkpointDirectory;
}

void Checkpointer::checkpointNow() {
   std::string checkpointDirectory = makeCheckpointDirectoryFromCurrentStep();
   if (checkpointDirectory == mCheckpointReadDirectory) {
      /* Note: the equality comparison isn't perfect, since there are multiple ways to specify a
       * path that points to the same directory. Should use realpath, but that breaks under OS X.
       */
      if (mMPIBlock->getGlobalRank() == 0) {
         InfoLog().printf(
               "Skipping checkpoint to \"%s\","
               " which would clobber the checkpointRead checkpoint.\n",
               checkpointDirectory.c_str());
      }
      return;
   }
   checkpointToDirectory(checkpointDirectory);

   if (mDeleteOlderCheckpoints) {
      rotateOldCheckpoints(checkpointDirectory);
   }
}

void Checkpointer::checkpointToDirectory(std::string const &directory) {
   mCheckpointTimer->start();
   auto fileManager = std::make_shared<FileManager>(getMPIBlock(), directory);
   if (mMPIBlock->getRank() == 0) {
      InfoLog() << "Checkpointing to directory \"" << directory
                << "\" at simTime = " << mTimeInfo.mSimTime << "\n";
      struct stat timeinfostat;
      int statstatus = fileManager->stat(std::string("timeinfo.bin"), timeinfostat);
      if (statstatus == 0) {
         WarnLog() << "Checkpoint directory \"" << directory
                   << "\" has existing timeinfo.bin, which is now being deleted.\n";
         mTimeInfoCheckpointEntry->remove(fileManager);
      }
   }
   fileManager->ensureDirectoryExists(std::string("."));
   notify(
         std::make_shared<PrepareCheckpointWriteMessage const>(),
         mMPIBlock->getRank() == 0 /*printFlag*/);
   for (auto &c : mCheckpointRegistry) {
      c->write(fileManager, mTimeInfo.mSimTime, mVerifyWrites);
   }
   mCheckpointTimer->stop();
   mCheckpointTimer->start();
   writeTimers(fileManager);

   // mTimeInfoCheckpointEntry should be the last thing that gets written to the checkpoint,
   // since its presence is used by isCompleteCheckpoint to confirm that a checkpoint is complete.
   mTimeInfoCheckpointEntry->write(fileManager, mTimeInfo.mSimTime, mVerifyWrites);
   mCheckpointTimer->stop();
   if (mMPIBlock->getRank() == 0) {
      InfoLog().printf("checkpointWrite complete. simTime = %f\n", mTimeInfo.mSimTime);
      InfoLog().flush();
   }
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
      auto fileManager = std::make_shared<FileManager>(getMPIBlock(), oldestCheckpointDir);
      if (fileManager->isRoot()) {
         struct stat lcp_stat;
         int statstatus = fileManager->stat(std::string("."), lcp_stat);
         if (statstatus) {
            ErrorLog().printf(
                  "Failed to delete older checkpoint: failed to stat \"%s\": %s.\n",
                  oldestCheckpointDir.c_str(),
                  strerror(errno));
         }
         else if (!(lcp_stat.st_mode & S_IFDIR)) {
            ErrorLog().printf(
                  "Failed to delete older checkpoint: \"%s\" exists but is not a directory.\n",
                  oldestCheckpointDir.c_str());
         }
         else {
            sync();
            mTimeInfoCheckpointEntry->remove(fileManager);
            fileManager->deleteFile(std::string("timers.txt"));

            for (auto &c : mCheckpointRegistry) {
               c->remove(fileManager);
            }
            fileManager->deleteDirectory(std::string(""));
         }
      }
   }
   mOldCheckpointDirectories[mOldCheckpointDirectoriesIndex] = newCheckpointDirectory;
   mOldCheckpointDirectoriesIndex++;
   if (mOldCheckpointDirectoriesIndex == mNumCheckpointsKept) {
      mOldCheckpointDirectoriesIndex = 0;
   }
}

void Checkpointer::writeTimers(std::shared_ptr<PrintStream> stream) const {
   if (!stream) { return; }
   for (auto timer : mTimers) {
      timer->fprint_time(*stream);
   }
}

std::string Checkpointer::generateBlockPath(std::string const &baseDirectory) {
   std::string path(baseDirectory);
   if (!mBlockDirectoryName.empty()) {
      path.append("/").append(mBlockDirectoryName);
   }
   return path;
}

void Checkpointer::verifyCheckpointDirectory(
      char const *directory,
      std::string const &description) {
   int status = PV_SUCCESS;
   if (mMPIBlock->getRank() == 0) {
      if (directory == nullptr || directory[0] == '\0') {
         ErrorLog() << "Checkpointer \"" << mName << "\": " << description << " is not set.\n";
         status = PV_FAILURE;
      }
      if (!isCompleteCheckpoint(std::string(directory))) {
         ErrorLog() << "Checkpoint \"" << mName << "\": \"" << directory << "\""
                    << " is not a checkpoint directory\n";
         status = PV_FAILURE;
      }
      if (status) {
         exit(EXIT_FAILURE);
      }
   }
}

void Checkpointer::writeTimers(std::shared_ptr<FileManager const> fileManager) {
   if (fileManager->isRoot()) {
      std::string timersfilename("timers.txt");
      auto timerstream = fileManager->open(timersfilename, std::ios_base::out, mVerifyWrites);
      writeTimers(timerstream);
   }
}

std::string const Checkpointer::mDefaultOutputPath = "output";
} // namespace PV
