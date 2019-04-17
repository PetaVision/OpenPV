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
#include "io/PVParams.hpp"
#include "observerpattern/Subject.hpp"
#include "structures/MPIBlock.hpp"
#include "utils/Timer.hpp"
#include <ctime>

namespace PV {

/**
 * A class to handle checkpointing tasks.
 */
class Checkpointer : public Subject {
  private:
   /**
    * List of parameters needed from the Checkpointer class
    * @name Checkpointer Parameters
    * @{
    */

   /**
    * @brief verifyWrites: If true, calls to FileStream::write are checked by
    * opening the file in read mode and reading back the data and comparing it
    * to the data just written.
    */
   virtual void ioParam_verifyWrites(enum ParamsIOFlag ioFlag, PVParams *params);

   /**
    * @brief mOutputPath: Specifies the absolute or relative output path of the
    * run
    */
   virtual void ioParam_outputPath(enum ParamsIOFlag ioFlag, PVParams *params);

   /**
    * @brief checkpointWrite: Flag to determine if the run writes checkpoints.
    */
   void ioParam_checkpointWrite(enum ParamsIOFlag ioFlag, PVParams *params);

   /**
    * @brief checkpointWriteDir: If checkpointWrite is set, specifies the output
    * checkpoint
    * directory.
    */
   void ioParam_checkpointWriteDir(enum ParamsIOFlag ioFlag, PVParams *params);

   /**
    * @brief mCheckpointWriteTriggerMode: If checkpointWrite is set, specifies
    * the method to
    * checkpoint.
    * @details Possible choices include
    * - step: Checkpoint off of timesteps
    * - time: Checkpoint off of simulation time
    * - clock: Checkpoint off of clock time. Not implemented yet.
    */
   void ioParam_checkpointWriteTriggerMode(enum ParamsIOFlag ioFlag, PVParams *params);

   /**
    * @brief checkpointWriteStepInterval: If checkpointWrite on step, specifies
    * the number of steps between checkpoints.
    */
   void ioParam_checkpointWriteStepInterval(enum ParamsIOFlag ioFlag, PVParams *params);

   /**
    * @brief checkpointWriteTimeInteval: If checkpointWrite on time, specifies
    * the amount of simulation time between checkpoints.
    */
   void ioParam_checkpointWriteTimeInterval(enum ParamsIOFlag ioFlag, PVParams *params);

   /**
    * @brief checkpointWriteClockInteval: If checkpointWrite on clock, specifies
    * the amount of clock
    * time between checkpoints.  The units are specified using the parameter
    * checkpointWriteClockUnit
    */
   void ioParam_checkpointWriteClockInterval(enum ParamsIOFlag ioFlag, PVParams *params);

   /**
    * @brief checkpointWriteClockInteval: If checkpointWrite on clock, specifies
    * the units used in checkpointWriteClockInterval.
    */
   void ioParam_checkpointWriteClockUnit(enum ParamsIOFlag ioFlag, PVParams *params);

   /**
    * @brief If checkpointWrite is true, checkpointIndexWidth specifies the
    * minimum width for the step number appearing in the checkpoint directory.
    * @details If the step number needs fewer digits than checkpointIndexWidth,
    * it is padded with zeroes.  If the step number needs more, the full
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
   void ioParam_deleteOlderCheckpoints(enum ParamsIOFlag ioFlag, PVParams *params);

   /**
    * @brief mNumCheckpointsKept: If mDeleteOlderCheckpoints is set,
    * keep this many checkpoints before deleting the checkpoint.
    * Default is 1 (delete a checkpoint when a newer checkpoint is written.)
    */
   void ioParam_numCheckpointsKept(enum ParamsIOFlag ioFlag, PVParams *params);

   /**
    * @brief initializeFromCheckpointDir: Sets directory used by
    * Checkpointer::initializeFromCheckpoint(). Layers and connections use this
    * directory if they set their initializeFromCheckpointFlag parameter.
    */
   void ioParam_initializeFromCheckpointDir(enum ParamsIOFlag ioFlag, PVParams *params);

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
   Checkpointer(
         std::string const &name,
         MPIBlock const *globalMPIBlock,
         Arguments const *arguments);
   ~Checkpointer();

   /**
    * Given a relative path, returns a full path consisting of the effective
    * output directory for the process's checkpoint cell, followed by "/",
    * followed by the given relative path. It is a fatal error for the path to
    * be an absolute path (i.e. starting with '/').
    */
   std::string makeOutputPathFilename(std::string const &path);

   void ioParams(enum ParamsIOFlag ioFlag, PVParams *params);
   void provideFinalStep(long int finalStep);

   template <typename T>
   bool registerCheckpointData(
         std::string const &objName,
         std::string const &dataName,
         T *dataPointer,
         size_t numValues,
         bool broadcast,
         bool constantEntireRun);

   bool registerCheckpointEntry(
         std::shared_ptr<CheckpointEntry> checkpointEntry,
         bool constantEntireRun);

   void registerTimer(Timer const *timer);

   void readNamedCheckpointEntry(
         std::string const &objName,
         std::string const &dataName,
         bool constantEntireRun);
   void
   readNamedCheckpointEntry(std::string const &checkpointEntryName, bool constantEntireRun = false);
   void readStateFromCheckpoint();
   void checkpointRead(double *simTimePointer, long int *currentStepPointer);
   void checkpointWrite(double simTime);
   void finalCheckpoint(double simTime);
   void writeTimers(PrintStream &stream) const;

   MPIBlock const *getMPIBlock() { return mMPIBlock; }
   bool doesVerifyWrites() { return mVerifyWrites; }
   std::string const &getOutputPath() { return mOutputPath; }
   bool getCheckpointWriteFlag() const { return mCheckpointWriteFlag; }
   char const *getCheckpointWriteDir() const { return mCheckpointWriteDir; }
   enum CheckpointWriteTriggerMode getCheckpointWriteTriggerMode() const {
      return mCheckpointWriteTriggerMode;
   }
   long int getCheckpointWriteStepInterval() const { return mCheckpointWriteStepInterval; }
   double getCheckpointWriteSimtimeInterval() const { return mCheckpointWriteSimtimeInterval; }
   bool getSuppressNonplasticCheckpoints() const { return mSuppressNonplasticCheckpoints; }
   std::string const &getCheckpointReadDirectory() const { return mCheckpointReadDirectory; }
   char const *getLastCheckpointDir() const { return mLastCheckpointDir; }
   char const *getInitializeFromCheckpointDir() const { return mInitializeFromCheckpointDir; }
   std::string const &getBlockDirectoryName() const { return mBlockDirectoryName; }

  private:
   void initMPIBlock(MPIBlock const *globalMPIBlock, Arguments const *arguments);
   void initBlockDirectoryName();
   void ioParamsFillGroup(enum ParamsIOFlag ioFlag, PVParams *params);

   /**
    * If called when mCheckpointReadDirectory is a colon-separated list of
    * paths, extracts the entry corresponding to the process's batch index
    * and replaces mCheckpointReadDirectory with that entry. Called by
    * configCheckpointReadDirectory. If mCheckpointReadDirectory is not a
    * colon-separated list, it is left unchanged. If it is a colon-separated
    * list, the number of entries must agree with
    *
    */
   void extractCheckpointReadDirectory();
   void findWarmStartDirectory();
   std::string makeCheckpointDirectoryFromCurrentStep();

   /**
     * If a SIGUSR signal has been received by the global root process, clears the signal
     * and returns true. Otherwise returns false.
     */
   bool receivedSignal();

   /**
     * Returns true if the params file settings indicate a checkpoint should occur at this
     * timestep. It also advances the appropriate data member for the trigger mode
     * to the next scheduled checkpoint. Returns false otherwise.
     */
   bool scheduledCheckpoint();

   /**
     * Called by scheduledCheckpoint if checkpointWriteTriggerMode is "step". If the
     * step number is an integral multiple of checkpointWriteStepInterval, it advances
     * mNextCheckpointStep by the step interval and returns true. Otherwise it returns false.
     */
   bool scheduledStep();

   /**
     * Called by scheduledCheckpoint if checkpointWriteTriggerMode is "time". If the
     * simTime is >= the current value of mNextCheckpointSimtime, it advances
     * mNextCheckpointSimtime by the time interval and returns true. Otherwise it returns false.
     */
   bool scheduledSimTime();

   /**
     * Called by scheduledCheckpoint if checkpointWriteTriggerMode is "clock". If the
     * elapsed time between the wall clock time and mLastCheckpointWallclock exceeds
     * mCheckpointWriteWallclockInterval, it sets mLastCheckpointWallclock to the current
     * wall clock time and returns true. Otherwise it returns false.
     */
   bool scheduledWallclock();

   /**
    * Called by checkpointWrite if a SIGUSR1 signal was sent (as reported by receivedSignal).
    * It writes a checkpoint, indexed by the current timestep. If the deleteOlderCheckpoints param
    * was set, it does not cause a checkpoint to be deleted, and does not rotate the checkpoint
    * into the list of directories that will be deleted.
    */
   void checkpointWriteSignal();

   /**
    * Called by checkpointWrite() (if there was not a SIGUSR1 signal pending) and finalCheckpoint().
    * Writes a checkpoint, indexed by the current timestep. If the deleteOlderCheckpoints param
    * was set, and the number of checkpoints exceeds numCheckpointsKept, the oldest checkpoint is
    * deleted, and the just-written checkpoint is rotated onto the list of checkpoints that will
    * be deleted.
    */
   void checkpointNow();

   /**
    * Creates a checkpoint based at the given directory. If the checkpoint directory already exists,
    * it issues a warning, and deletes the timeinfo.bin file in the checkpooint. This way, the
    * presence of the timeinfo.bin file indicates that the checkpoint is complete.
    */
   void checkpointToDirectory(std::string const &checkpointDirectory);

   /**
    * Called if deleteOlderCheckpoints is true. It deletes the oldest checkpoint in the list of
    * old checkpoint directories, and adds the new checkpoint directory to the list.
    */
   void rotateOldCheckpoints(std::string const &newCheckpointDirectory);
   void writeTimers(std::string const &directory);
   std::string generateBlockPath(std::string const &baseDirectory);
   void verifyDirectory(char const *directory, std::string const &description);

  private:
   std::string mName;
   MPIBlock *mMPIBlock = nullptr;
   std::string mBlockDirectoryName;
   std::vector<std::shared_ptr<CheckpointEntry>> mCheckpointRegistry; // Needs to be a vector so
   // that each MPI process
   // iterates over the entries
   // in the same order.
   TimeInfo mTimeInfo;
   std::shared_ptr<CheckpointEntryData<TimeInfo>> mTimeInfoCheckpointEntry = nullptr;
   bool mWarmStart                                                         = false;
   bool mVerifyWrites                                                      = true;
   std::string mOutputPath                                                 = "";
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
   std::string mCheckpointReadDirectory;
   long int mNextCheckpointStep         = 0L; // kept only for consistency with HyPerCol
   double mNextCheckpointSimtime        = 0.0;
   std::time_t mLastCheckpointWallclock = (std::time_t)0;
   int mWidthOfFinalStepNumber          = 0;
   int mOldCheckpointDirectoriesIndex =
         0; // A pointer to the oldest checkpoint in the mOldCheckpointDirectories vector.
   std::vector<std::string> mOldCheckpointDirectories; // A ring buffer of existing checkpoints,
   // used if mDeleteOlderCheckpoints is true.
   std::vector<Timer const *> mTimers;
   Timer *mCheckpointTimer = nullptr;

   static std::string const mDefaultOutputPath;
};

} // namespace PV

#include "Checkpointer.tpp"

#endif // CHECKPOINTER_HPP_
