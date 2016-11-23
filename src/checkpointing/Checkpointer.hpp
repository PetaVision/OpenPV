/*
 * Checkpointer.hpp
 *
 *  Created on Sep 28, 2016
 *      Author: Pete Schultz
 */

#ifndef CHECKPOINTER_HPP_
#define CHECKPOINTER_HPP_

#include "checkpointing/CheckpointEntry.hpp"
#include "checkpointing/CheckpointEntryData.hpp"
#include "checkpointing/CheckpointingMessages.hpp"
#include "columns/Communicator.hpp"
#include "io/PVParams.hpp"
#include "io/io.hpp"
#include "observerpattern/Subject.hpp"
#include "utils/Timer.hpp"
#include <ctime>
#include <map>
#include <memory>

namespace PV {

class Checkpointer : public Subject {
  private:
   /**
    * List of parameters needed from the Checkpointer class
    * @name Checkpointer Parameters
    * @{
    */

   void ioParam_verifyWrites(enum ParamsIOFlag ioFlag, PVParams *params);
   void ioParam_checkpointWrite(enum ParamsIOFlag ioFlag, PVParams *params);
   void ioParam_checkpointWriteDir(enum ParamsIOFlag ioFlag, PVParams *params);
   void ioParam_checkpointWriteTriggerMode(enum ParamsIOFlag ioFlag, PVParams *params);
   void ioParam_checkpointWriteStepInterval(enum ParamsIOFlag ioFlag, PVParams *params);
   void ioParam_checkpointWriteTimeInterval(enum ParamsIOFlag ioFlag, PVParams *params);
   void ioParam_checkpointWriteClockInterval(enum ParamsIOFlag ioFlag, PVParams *params);
   void ioParam_checkpointWriteClockUnit(enum ParamsIOFlag ioFlag, PVParams *params);

   /**
    * @brief If checkpointWrite is true, checkpointIndexWidth specifies the
    * minimum width for the
    * step number appearing in the checkpoint directory.
    * @details If the step number needs fewer digits than checkpointIndexWidth,
    * it is padded with
    * zeroes.  If the step number needs more, the full
    * step number is still printed.  Hence, setting checkpointWrite to zero means
    * that there are
    * never any padded zeroes.
    * If set to a negative number, the width will be inferred from startTime,
    * stopTime and dt.
    * The default value is -1 (infer the width).
    */
   void ioParam_checkpointIndexWidth(enum ParamsIOFlag ioFlag, PVParams *params);

   /**
    * If checkpointWrite is true and this flag is true,
    * connections will only checkpoint if plasticityFlag=true.
    */
   void ioParam_suppressNonplasticCheckpoints(enum ParamsIOFlag ioFlag, PVParams *params);

   /**
    * @brief deleteOlderCheckpoints: If checkpointWrite, specifies if the run
    * should delete older checkpoints when writing new ones.
    */
   virtual void ioParam_deleteOlderCheckpoints(enum ParamsIOFlag ioFlag, PVParams *params);

   /**
    * @brief mNumCheckpointsKept: If mDeleteOlderCheckpoints is set,
    * keep this many checkpoints before deleting the checkpoint.
    * Default is 1 (delete a checkpoint when a newer checkpoint is written.)
    */
   virtual void ioParam_numCheckpointsKept(enum ParamsIOFlag ioFlag, PVParams *params);

   /**
    * @brief initializeFromCheckpointDir: Sets directory used by
    * Checkpointer::initializeFromCheckpoint(). Layers and connections use this
    * directory if they set their initializeFromCheckpointFlag parameter.
    */
   virtual void ioParam_initializeFromCheckpointDir(enum ParamsIOFlag ioFlag, PVParams *params);

   /**
    * @brief defaultInitializeFromCheckpointFlag: Flag to set the default for
    * layers and connections.
    * @details Sets the default for layers and connections to use for initialize
    * from checkpoint based off of initializeFromCheckpointDir. Only used if
    * initializeFromCheckpointDir is set.
    */
   virtual void
   ioParam_defaultInitializeFromCheckpointFlag(enum ParamsIOFlag ioFlag, PVParams *params);

   /**
    * @brief lastCheckpointDir: If checkpointWrite is not set, this required parameter specifies
    * the directory to write a final written checkpoint at the end of the run.
    * Writing the last checkpoint can be suppressed by setting this string to the empty string.
    * Relative paths are relative to the working directory.
    */
   void ioParam_lastCheckpointDir(enum ParamsIOFlag ioFlag, PVParams *params);
   /** @} */

   enum CheckpointWriteTriggerMode { NONE, STEP, SIMTIME, WALLCLOCK };
   enum WallClockUnit { SECOND, MINUTE, HOUR, DAY };

  public:
   struct TimeInfo {
      double mSimTime                 = 0.0;
      long int mCurrentCheckpointStep = 0L;
   };
   Checkpointer(std::string const &name, Communicator *comm);
   ~Checkpointer();

   void ioParamsFillGroup(enum ParamsIOFlag ioFlag, PVParams *params);
   void provideFinalStep(long int finalStep);

   template <typename T>
   bool registerCheckpointData(
         std::string const &objName,
         std::string const &dataName,
         T *dataPointer,
         size_t numValues,
         bool broadcast);

   bool registerCheckpointEntry(
         std::shared_ptr<CheckpointEntry> checkpointEntry,
         bool constantEntireRun = false);
   void registerTimer(Timer const *timer);
   virtual void addObserver(Observer *observer, BaseMessage const &message) override;

   void setCheckpointReadDirectory();
   void setCheckpointReadDirectory(std::string const &checkpointReadDirectory);
   void readNamedCheckpointEntry(std::string const &objName, std::string const &dataName) const;
   void readNamedCheckpointEntry(std::string const &checkpointEntryName) const;
   void checkpointRead(double *simTimePointer, long int *currentStepPointer);
   void checkpointWrite(double simTime);
   void finalCheckpoint(double simTime);
   void writeTimers(PrintStream &stream) const;

   Communicator *getCommunicator() { return mCommunicator; }
   bool doesVerifyWrites() { return mVerifyWritesFlag; }
   bool getCheckpointWriteFlag() const { return mCheckpointWriteFlag; }
   char const *getcheckpointWriteDir() const { return mCheckpointWriteDir; }
   enum CheckpointWriteTriggerMode getCheckpointWriteTriggerMode() const {
      return mCheckpointWriteTriggerMode;
   }
   long int getCheckpointWriteStepInterval() const { return mCheckpointWriteStepInterval; }
   double getCheckpointWriteSimtimeInterval() const { return mCheckpointWriteSimtimeInterval; }
   bool getSuppressNonplasticCheckpoints() const { return mSuppressNonplasticCheckpoints; }
   std::string const &getCheckpointReadDirectory() const { return mCheckpointReadDirectory; }
   char const *getLastCheckpointDir() const { return mLastCheckpointDir; }
   char const *getInitializeFromCheckpointDir() const { return mInitializeFromCheckpointDir; }
   bool getDefaultInitializeFromCheckpointFlag() const {
      return mDefaultInitializeFromCheckpointFlag;
   }

  private:
   void initialize();
   void findWarmStartDirectory();
   bool checkpointWriteSignal();
   void checkpointWriteStep();
   void checkpointWriteSimtime();
   void checkpointWriteWallclock();
   void checkpointNow();
   void checkpointToDirectory(std::string const &checkpointDirectory);
   void rotateOldCheckpoints(std::string const &newCheckpointDirectory);
   void writeTimers(std::string const &directory);

  private:
   std::string mName;
   Communicator *mCommunicator = nullptr;
   std::vector<std::shared_ptr<CheckpointEntry>> mCheckpointRegistry; // Needs to be a vector so
   // that each MPI process
   // iterates over the entries
   // in the same order.
   ObserverTable mObserverTable;
   TimeInfo mTimeInfo;
   std::shared_ptr<CheckpointEntryData<TimeInfo>> mTimeInfoCheckpointEntry = nullptr;
   bool mVerifyWritesFlag                                                  = true;
   bool mCheckpointWriteFlag                                               = false;
   char *mCheckpointWriteDir                                               = nullptr;
   char *mCheckpointWriteTriggerModeString                                 = nullptr;
   enum CheckpointWriteTriggerMode mCheckpointWriteTriggerMode             = NONE;
   long int mCheckpointWriteStepInterval                                   = 1L;
   double mCheckpointWriteSimtimeInterval                                  = 1.0;
   std::time_t mCheckpointWriteWallclockInterval                           = 1L;
   char *mCheckpointWriteWallclockUnit                                     = nullptr;
   std::time_t mCheckpointWriteWallclockIntervalSeconds                    = 1L;
   int mCheckpointIndexWidth                                               = -1;
   bool mSuppressNonplasticCheckpoints                                     = false;
   bool mDeleteOlderCheckpoints                                            = false;
   int mNumCheckpointsKept                                                 = 2;
   char *mLastCheckpointDir                                                = nullptr;
   char *mInitializeFromCheckpointDir                                      = nullptr;
   bool mDefaultInitializeFromCheckpointFlag                               = false;
   std::string mCheckpointReadDirectory;
   int mCheckpointSignal                = 0;
   long int mNextCheckpointStep         = 0L; // kept only for consistency with HyPerCol
   double mNextCheckpointSimtime        = 0.0;
   std::time_t mLastCheckpointWallclock = (std::time_t)0;
   std::time_t mNextCheckpointWallclock = (std::time_t)0;
   int mWidthOfFinalStepNumber          = 0;
   int mOldCheckpointDirectoriesIndex =
         0; // A pointer to the oldest checkpoint in the mOldCheckpointDirectories vector.
   std::vector<std::string> mOldCheckpointDirectories; // A ring buffer of existing checkpoints,
   // used if mDeleteOlderCheckpoints is true.
   std::vector<Timer const *> mTimers;
   Timer *mCheckpointTimer = nullptr;

   static std::string const mDefaultOutputPath;
};

/**
 * CheckpointerDataInterface provides a virtual method intended for interfacing
 * with Checkpointer register methods.  An object that does checkpointing should
 * derive from CheckpointerDataInterface, and override registerData to call
 * Checkpointer::registerCheckpointEntry once for each piece of checkpointable
 * data.
 *
 * BaseObject derives from CheckpointerDataInterface, and calls registerData
 * when it receives a RegisterDataMessage (which HyPerCol::run calls after
 * AllocateDataMessage and before InitializeStateMessage).
 */
class CheckpointerDataInterface {
  public:
   virtual int registerData(Checkpointer *checkpointer, std::string const &objName) {
      return PV_SUCCESS;
   }
};

template <typename T>
bool Checkpointer::registerCheckpointData(
      std::string const &objName,
      std::string const &dataName,
      T *dataPointer,
      std::size_t numValues,
      bool broadcast) {
   return registerCheckpointEntry(
         std::make_shared<CheckpointEntryData<T>>(
               objName, dataName, getCommunicator(), dataPointer, numValues, broadcast));
}

} // namespace PV

#endif // CHECKPOINTER_HPP_
