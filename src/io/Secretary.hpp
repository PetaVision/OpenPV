/*
 * Secretary.hpp
 *
 *  Created on Sep 28, 2016
 *      Author: Pete Schultz
 */

#ifndef SECRETARY_HPP_
#define SECRETARY_HPP_

#include "columns/Communicator.hpp"
#include "io/CheckpointEntry.hpp"
#include "io/HyPerCheckpoint.hpp"
#include "io/PVParams.hpp"
#include "io/io.hpp"
#include "observerpattern/BaseMessage.hpp"
#include "observerpattern/Subject.hpp"
#include "utils/Timer.hpp"
#include <ctime>
#include <map>
#include <memory>

namespace PV {

class Secretary : public Subject {
  private:
   /**
    * List of parameters needed from the Secretary class
    * @name Secretary Parameters
    * @{
    */

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
   /** @} */

   enum CheckpointWriteTriggerMode { NONE, STEP, SIMTIME, WALLCLOCK };
   enum WallClockUnit { SECOND, MINUTE, HOUR, DAY };

  public:
   struct TimeInfo {
      double mSimTime                 = 0.0;
      long int mCurrentCheckpointStep = 0L;
   };
   class ProcessCheckpointReadMessage : public BaseMessage {
     public:
      ProcessCheckpointReadMessage() { setMessageType("ProcessCheckpointReadMessage"); }
   };
   class PrepareCheckpointWriteMessage : public BaseMessage {
     public:
      PrepareCheckpointWriteMessage() { setMessageType("ProcessCheckpointWriteMessage"); }
   };
   Secretary(std::string const &name, Communicator *comm);
   Secretary(HyPerCheckpoint *cp, std::string const &name, Communicator *comm);
   ~Secretary();

   void setOutputPath(std::string const &outputPath);
   void ioParamsFillGroup(enum ParamsIOFlag ioFlag, PVParams *params);
   void provideFinalStep(long int finalStep);

   template <typename T>
   bool registerCheckpointData(
         std::string const &objName,
         std::string const &dataName,
         T *dataPointer,
         size_t numValues,
         bool broadcast);

   bool registerCheckpointEntry(std::shared_ptr<CheckpointEntry> checkpointEntry);
   void registerTimer(Timer const *timer);
   virtual void addObserver(Observer *observer, BaseMessage const &message) override;
   void checkpointRead(
         std::string const &checkpointReadDir,
         double *simTimePointer,
         long int *currentStepPointer);
   void checkpointWrite(double simTime);
   void finalCheckpoint(double simTime);
   void writeTimers(PrintStream &stream) const;

   Communicator *getCommunicator() { return mCommunicator; }
   bool doesVerifyWrites() { return mVerifyWritesFlag; }
   char const *getOutputPath() const { return mOutputPath; }
   bool getCheckpointWriteFlag() const { return mCheckpointWriteFlag; }
   char const *getcheckpointWriteDir() const { return mCheckpointWriteDir; }
   enum CheckpointWriteTriggerMode getCheckpointWriteTriggerMode() const {
      return mCheckpointWriteTriggerMode;
   }
   long int getCheckpointWriteStepInterval() const { return mCheckpointWriteStepInterval; }
   double getCheckpointWriteSimtimeInterval() const { return mCheckpointWriteSimtimeInterval; }
   int getCheckpointIndexWidth() const { return mCheckpointIndexWidth; }
   bool getSuppressNonplasticCheckpoints() const { return mSuppressNonplasticCheckpoints; }
   bool getSuppressLastOutput() const { return mSuppressLastOutput; }

  private:
   void initialize();
   void ioParam_verifyWrites(enum ParamsIOFlag ioFlag, PVParams *params);
   void ioParam_outputPath(enum ParamsIOFlag ioFlag, PVParams *params);
   void ioParam_checkpointWrite(enum ParamsIOFlag ioFlag, PVParams *params);
   void ioParam_checkpointWriteDir(enum ParamsIOFlag ioFlag, PVParams *params);
   void ioParam_checkpointWriteTriggerMode(enum ParamsIOFlag ioFlag, PVParams *params);
   void ioParam_checkpointWriteStepInterval(enum ParamsIOFlag ioFlag, PVParams *params);
   void ioParam_checkpointWriteTimeInterval(enum ParamsIOFlag ioFlag, PVParams *params);
   void ioParam_checkpointWriteClockInterval(enum ParamsIOFlag ioFlag, PVParams *params);
   void ioParam_checkpointWriteClockUnit(enum ParamsIOFlag ioFlag, PVParams *params);
   void ioParam_checkpointIndexWidth(enum ParamsIOFlag ioFlag, PVParams *params);
   void ioParam_suppressNonplasticCheckpoints(enum ParamsIOFlag ioFlag, PVParams *params);
   void ioParam_suppressLastOutput(enum ParamsIOFlag ioFlag, PVParams *params);
   bool checkpointWriteSignal();
   void checkpointWriteStep();
   void checkpointWriteSimtime();
   void checkpointWriteWallclock();
   void checkpointNow();
   void checkpointToDirectory(std::string const &checkpointDirectory);
   void rotateOldCheckpoints(std::string const &newCheckpointDirectory);
   void writeTimers(std::string const &directory);

  private:
   HyPerCheckpoint *mHyPerCheckpoint = nullptr;
   // Note: the HyPerCheckpoint is a convenience in moving checkpointing functions from HyPerCol
   // to Secretary. Once the refactor is complete, the mHyPerCheckpoint will be removed.

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
   char *mOutputPath                                                       = nullptr;
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
   bool mSuppressLastOutput                                                = false;
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
 * SecretaryDataInterface provides a virtual method intended for interfacing
 * with Secretary register methods.  An object that does checkpointing should
 * derive from SecretaryDataInterface, and override registerData to call
 * Secretary::registerCheckpointEntry once for each piece of checkpointable
 * data.
 *
 * BaseObject derives from SecretaryDataInterface, and calls registerData
 * when it receives a RegisterDataMessage (which HyPerCol::run calls after
 * AllocateDataMessage and before InitializeStateMessage).
 */
class SecretaryDataInterface {
  public:
   virtual int registerData(Secretary *secretary, std::string const &objName) { return PV_SUCCESS; }
};

template <typename T>
bool Secretary::registerCheckpointData(
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

#endif // SECRETARY_HPP_
