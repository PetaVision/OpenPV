/*
 * Checkpointer.cpp
 *
 *  Created on Sep 28, 2016
 *      Author: Pete Schultz
 */

#include "Checkpointer.hpp"
#include <cmath>
#include <signal.h>

namespace PV {

Checkpointer::Checkpointer(std::string const &name, Communicator *comm)
      : mName(name), mCommunicator(comm) {
   initialize();
}

Checkpointer::~Checkpointer() {
   free(mOutputPath);
   free(mCheckpointWriteDir);
   free(mCheckpointWriteTriggerModeString);
   free(mCheckpointWriteWallclockUnit);
   delete mCheckpointTimer;
}

void Checkpointer::initialize() {
   mTimeInfoCheckpointEntry = std::make_shared<CheckpointEntryData<Checkpointer::TimeInfo>>(
         std::string("timeinfo"), getCommunicator(), &mTimeInfo, (size_t)1, true /*broadcast*/);
   // This doesn't get put into mCheckpointRegistry because we handle the timeinfo separately.
   mCheckpointTimer = new Timer(mName.c_str(), "column", "checkpoint");
   registerTimer(mCheckpointTimer);
}

void Checkpointer::setOutputPath(std::string const &outputPath) {
   if (mOutputPath) {
      WarnLog(changingOutputPath);
      changingOutputPath << "\"" << mName << "\": changing output path from \"" << mOutputPath
                         << "\" to ";
      if (!outputPath.empty()) {
         changingOutputPath << "\"" << outputPath << "\".\n";
      }
      else {
         changingOutputPath << "null.\n";
      }
   }
   if (!outputPath.empty()) {
      mOutputPath = strdup(expandLeadingTilde(outputPath.c_str()).c_str());
      FatalIf(mOutputPath == nullptr, "Checkpointer::setOutputPath unable to copy output path.\n");
   }
   else {
      mOutputPath = nullptr;
   }
}

void Checkpointer::ioParamsFillGroup(enum ParamsIOFlag ioFlag, PVParams *params) {
   ioParam_verifyWrites(ioFlag, params);
   ioParam_outputPath(ioFlag, params);
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
   ioParam_suppressLastOutput(ioFlag, params);
}

void Checkpointer::ioParam_verifyWrites(enum ParamsIOFlag ioFlag, PVParams *params) {
   params->ioParamValue(
         ioFlag, mName.c_str(), "verifyWrites", &mVerifyWritesFlag, mVerifyWritesFlag);
}

void Checkpointer::ioParam_outputPath(enum ParamsIOFlag ioFlag, PVParams *params) {
   switch (ioFlag) {
      case PARAMS_IO_READ:
         // To use make use of the -o option on the command line, call Checkpointer::setOutputPath()
         // before Checkpointer::ioParamsFillGroup(), as HyPerCol::initialize() does.
         if (mOutputPath == nullptr) {
            if (!params->stringPresent(mName.c_str(), "outputPath")) {
               WarnLog() << "Output path specified neither in command line nor in params file.\n";
            }
            params->ioParamString(
                  ioFlag,
                  mName.c_str(),
                  "outputPath",
                  &mOutputPath,
                  mDefaultOutputPath.c_str(),
                  true);
         }
         else {
            if (params->stringPresent(mName.c_str(), "outputPath")) {
               InfoLog() << "Output path \"" << mOutputPath
                         << "\" specified on command line; value in params file will be ignored.\n";
            }
         }
         break;
      case PARAMS_IO_WRITE: params->writeParamString("outputPath", mOutputPath); break;
      default: pvAssert(0); break;
   }
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

void Checkpointer::ioParam_suppressLastOutput(enum ParamsIOFlag ioFlag, PVParams *params) {
   assert(!params->presentAndNotBeenRead(mName.c_str(), "checkpointWrite"));
   if (!mCheckpointWriteFlag) {
      params->ioParamValue(
            ioFlag, mName.c_str(), "suppressLastOutput", &mSuppressLastOutput, mSuppressLastOutput);
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

void Checkpointer::checkpointRead(
      std::string const &checkpointReadDir,
      double *simTimePointer,
      long int *currentStepPointer) {
   mCheckpointReadDirectory = checkpointReadDir;
   double readTime;
   for (auto &c : mCheckpointRegistry) {
      c->read(checkpointReadDir, &readTime);
   }
   mTimeInfoCheckpointEntry->read(checkpointReadDir.c_str(), &readTime);
   if (simTimePointer) {
      *simTimePointer = mTimeInfo.mSimTime;
   }
   if (currentStepPointer) {
      *currentStepPointer = mTimeInfo.mCurrentCheckpointStep;
   }
   notify(mObserverTable, std::make_shared<ProcessCheckpointReadMessage const>(checkpointReadDir));
}

void Checkpointer::checkpointWrite(double simTime) {
   std::string checkpointWriteDir;
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
   if (strcmp(checkpointDirectory.c_str(), mCheckpointReadDirectory.c_str())) {
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
   mCheckpointTimer->start();
   if (getCommunicator()->commRank() == 0) {
      InfoLog() << "Checkpointing to directory \"" << directory
                << "\" at simTime = " << mTimeInfo.mSimTime << "\n";
      struct stat timeinfostat;
      std::string timeinfoFilename(directory);
      timeinfoFilename.append("/timeinfo.bin");
      int statstatus = stat(timeinfoFilename.c_str(), &timeinfostat);
      if (statstatus == 0) {
         WarnLog() << "Checkpoint directory \"" << directory
                   << "\" has existing timeinfo.bin, which is now being deleted.\n";
         mTimeInfoCheckpointEntry->remove(timeinfoFilename);
      }
   }
   notify(mObserverTable, std::make_shared<PrepareCheckpointWriteMessage const>(directory));
   ensureDirExists(getCommunicator(), directory.c_str());
   for (auto &c : mCheckpointRegistry) {
      c->write(directory, mTimeInfo.mSimTime, mVerifyWritesFlag);
   }
   mTimeInfoCheckpointEntry->write(directory, mTimeInfo.mSimTime, mVerifyWritesFlag);
   mCheckpointTimer->stop();
   mCheckpointTimer->start();
   writeTimers(directory);
   mCheckpointTimer->stop();
}

void Checkpointer::finalCheckpoint(double simTime) {
   mTimeInfo.mSimTime = simTime;
   if (mCheckpointWriteFlag) {
      checkpointNow();
   }
   else if (!mSuppressLastOutput) {
      std::string finalCheckpointDir{mOutputPath};
      finalCheckpointDir.append("/Last");
      checkpointToDirectory(finalCheckpointDir);
   }
}

void Checkpointer::rotateOldCheckpoints(std::string const &newCheckpointDirectory) {
   std::string &oldestCheckpointDir = mOldCheckpointDirectories[mOldCheckpointDirectoriesIndex];
   if (!oldestCheckpointDir.empty()) {
      if (mCommunicator->commRank() == 0) {
         struct stat lcp_stat;
         int statstatus = stat(oldestCheckpointDir.c_str(), &lcp_stat);
         if (statstatus != 0 || !(lcp_stat.st_mode & S_IFDIR)) {
            if (statstatus == 0) {
               ErrorLog().printf(
                     "Failed to delete older checkpoint: failed to stat \"%s\": %s.\n",
                     oldestCheckpointDir.c_str(),
                     strerror(errno));
            }
            else {
               ErrorLog().printf(
                     "Deleting older checkpoint: \"%s\" exists but is not a directory.\n",
                     oldestCheckpointDir.c_str());
            }
         }
         sync();
         std::string rmrf_string("");
         rmrf_string     = rmrf_string + "rm -r '" + oldestCheckpointDir + "'";
         int rmrf_result = system(rmrf_string.c_str());
         if (rmrf_result != 0) {
            WarnLog().printf(
                  "unable to delete older checkpoint \"%s\": rm command returned %d\n",
                  oldestCheckpointDir.c_str(),
                  WEXITSTATUS(rmrf_result));
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

namespace TextOutput {

template <>
void print(Checkpointer::TimeInfo const *dataPointer, size_t numValues, PrintStream &stream) {
   for (size_t n = 0; n < numValues; n++) {
      stream << "time = " << dataPointer[n].mSimTime << "\n";
      stream << "timestep = " << dataPointer[n].mCurrentCheckpointStep << "\n";
   }
} // print()

} // namespace TextOutput

} // namespace PV
